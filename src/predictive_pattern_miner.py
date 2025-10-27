#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Template Gray Miner -- SLA-anchored ranking of Drain templates (End-to-End)
==========================================================================

Implements:
  - Step 4: Count per window set (incident vs baseline)
  - Step 5: Score templates (log-odds z, lift, clarity, lead/lag)
  - Step 6: Multiple testing control (BH-FDR) + stability filters
  - Step 7: Review output (ranked DataFrame + YAML) + diagnostics + prod rules

Designed for large datasets (~20M+ rows):
  - Chunked two-pass pipeline (counts+clarity, then top-K lead/lag)
  - Parallel lead/lag via ProcessPoolExecutor
  - Memory optimizations (categoricals, observed=True)
  - Windows 11 & Linux compatible (spawn-safe)

Dependencies: pandas, numpy, (optional) scipy (for xcorr), (optional) PyYAML, tqdm
"""

from __future__ import annotations

import math
import os
import warnings
import typing as T
from dataclasses import dataclass
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

# --- Optional extras (handled gracefully) ---
try:
    from scipy import signal  # used for normalized cross-correlation
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

try:
    import yaml  # for YAML export
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

try:
    from tqdm import tqdm
except Exception:  # fallback no-op if tqdm not installed
    def tqdm(it, **kwargs):  # type: ignore
        return it


# ====================== Config & Diagnostics Types ======================

@dataclass
class MinerConfig:
    """Configuration for TemplateGrayMiner with validation."""

    # Column names
    time_col: str = "timestamp"
    template_col: str = "log_template_id"
    sla_start_col: str = "sla_breach_start"
    sla_end_col: str = "sla_breach_end"
    window_start_col: str = "from_time"
    window_end_col: str = "end_time"
    context_cols: T.Optional[T.List[str]] = None
    template_text_col: T.Optional[str] = None

    # Timing parameters
    freq: str = "1min"
    lead_lag_horizon: str = "10min"

    # Scoring parameters
    z_weight: float = 0.35
    beta_lead: float = 0.3
    clarity_power: float = 1.0
    max_lift: float = 10.0

    # Filtering parameters
    min_support_incident: int = 5
    min_template_occurrences: int = 10
    alpha_fdr: T.Optional[float] = 0.05

    # Performance parameters
    compute_lead_lag: bool = True
    correlation_method: str = "xcorr"  # "xcorr" (SciPy) or "pearson"
    n_jobs: int = 1
    top_k_leadlag: T.Optional[int] = 2000  # compute lead/lag only for top-K after prelim score
    chunk_size: T.Optional[int] = None     # if None, single-pass; else two-pass chunked
    show_progress: bool = True

    # Context weights for clarity calculation (per column)
    context_weights: T.Optional[T.Dict[str, float]] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        if not (0 <= self.beta_lead <= 1):
            raise ValueError(f"beta_lead must be in [0,1], got {self.beta_lead}")
        if self.z_weight <= 0:
            raise ValueError(f"z_weight must be positive, got {self.z_weight}")
        if self.clarity_power <= 0:
            raise ValueError(f"clarity_power must be positive, got {self.clarity_power}")
        if self.min_support_incident < 1:
            raise ValueError(f"min_support_incident must be >= 1, got {self.min_support_incident}")
        if self.n_jobs < 1:
            raise ValueError(f"n_jobs must be >= 1, got {self.n_jobs}")
        if self.correlation_method not in ("xcorr", "pearson"):
            raise ValueError(f"correlation_method must be 'xcorr' or 'pearson', got {self.correlation_method}")


@dataclass
class TemplateDiagnostics:
    """Diagnostic information for a ranked template."""
    template_id: T.Any
    template_text: T.Optional[str]
    score_breakdown: T.Dict[str, float]
    temporal_pattern: T.Dict[str, T.Any]
    example_contexts: T.List[T.Dict[str, T.Any]]
    recommendations: T.List[str]
    risk_level: str


# ============================ Math Utilities ============================

def _ensure_datetime_utc(df: pd.DataFrame, cols: T.Iterable[str]) -> None:
    """Ensure datetime columns are parsed and UTC."""
    for c in cols:
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    if df[list(cols)].isna().any().any():
        bad = df[list(cols)].isna().sum()
        raise ValueError(f"NaT values found in datetime columns: {bad[bad > 0].to_dict()}")


def _logit_series(p: pd.Series) -> pd.Series:
    """Safe logit transform."""
    eps = 1e-12
    p = p.clip(eps, 1 - eps)
    return np.log(p / (1.0 - p))


def _log_comb(n: int, k: int) -> float:
    """Log binomial coefficient via lgamma."""
    if k > n or k < 0:
        return -np.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _fisher_one_sided(N: int, K: int, n: int, k: int) -> float:
    """One-sided Fisher exact test p-value: P[X >= k], X ~ Hypergeom(N, K, n)."""
    if k > min(n, K):
        return 0.0
    k_max = min(n, K)
    log_denom = _log_comb(N, n)
    logs = []
    for x in range(k, k_max + 1):
        log_pmf = _log_comb(K, x) + _log_comb(N - K, n - x) - log_denom
        if np.isfinite(log_pmf):
            logs.append(log_pmf)
    if not logs:
        return 1.0
    m = max(logs)
    s = sum(math.exp(v - m) for v in logs)
    return float(min(1.0, math.exp(m) * s))


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction; returns q-values."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, m + 1)
    q = p * m / ranks
    q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
    q_out = np.empty_like(q)
    q_out[order] = np.clip(q_sorted, 0, 1)
    return q_out


# ---------------- Normalized cross-correlation (SciPy path) -------------

def _normalized_xcorr(x: np.ndarray, y: np.ndarray, max_lag_bins: int) -> T.Tuple[np.ndarray, np.ndarray]:
    """
    Normalized cross-correlation between x (template) and y (SLA).
    We return only lags in [-max_lag_bins, 0]; negative lags mean template leads.
    """
    # z-score normalize
    x = (x - x.mean()) / (x.std() + 1e-10)
    y = (y - y.mean()) / (y.std() + 1e-10)

    # correlate x against y
    corr = signal.correlate(x, y, mode='full')
    lags = signal.correlation_lags(len(x), len(y), mode='full')

    # normalize by overlap length for comparability
    overlap = (np.minimum(len(x), len(y)) - np.abs(lags)).astype(float)
    overlap[overlap <= 0] = np.nan
    corr = corr / overlap

    mask = (lags >= -max_lag_bins) & (lags <= 0)
    return corr[mask], lags[mask]


# ----------------------- Lead/Lag Worker (parallel) ---------------------

def _leadlag_worker_enhanced(
    tpl_id: T.Any,
    tpl_times_ns: np.ndarray,
    tpl_wids: np.ndarray,
    windows_table: np.ndarray,
    freq_seconds: int,
    max_lag_bins: int,
    correlation_method: str = "xcorr"
) -> T.Tuple[T.Any, float, int, int]:
    """
    Compute best positive correlation at negative lags (template leads).
    Returns: (tpl_id, r_weighted, lag_seconds_weighted, windows_contributed)
    """
    if tpl_times_ns.size == 0:
        return tpl_id, 0.0, 0, 0

    freq_ns = np.int64(freq_seconds) * np.int64(1_000_000_000)

    r_vals: T.List[float] = []
    lag_vals: T.List[int] = []
    weights: T.List[float] = []

    unique_wids = np.unique(tpl_wids)

    for row in windows_table:
        wid = np.int64(row[0])
        if wid not in unique_wids:
            continue

        w_start_ns, w_end_ns, sla_start_ns, sla_end_ns = map(np.int64, row[1:5])
        # Extract timestamps for this window
        mask = (tpl_wids == wid)
        ts_w = tpl_times_ns[mask]
        if ts_w.size == 0:
            continue
        ts_w = ts_w[(ts_w >= w_start_ns) & (ts_w < w_end_ns)]
        if ts_w.size == 0:
            continue

        bins = int((w_end_ns - w_start_ns) // freq_ns)
        if bins <= 1:
            continue

        # Template activity series x
        idx = ((ts_w - w_start_ns) // freq_ns).astype(np.int64)
        idx = idx[(idx >= 0) & (idx < bins)]
        if idx.size == 0:
            continue
        x = np.bincount(idx, minlength=bins).astype(np.float64)

        # SLA indicator series y
        y = np.zeros(bins, dtype=np.float64)
        s0 = max(0, int((sla_start_ns - w_start_ns) // freq_ns))
        s1 = min(bins, int((sla_end_ns - w_start_ns) // freq_ns))
        if s1 <= s0:
            continue
        y[s0:s1] = 1.0

        # Skip if no variance
        if x.std() == 0.0 or y.std() == 0.0:
            continue

        if correlation_method == "xcorr" and _HAS_SCIPY:
            corrs, lags = _normalized_xcorr(x, y, max_lag_bins)
            if corrs.size:
                best_idx = int(np.nanargmax(corrs))
                best_r = float(corrs[best_idx])
                best_lag_bins = int(lags[best_idx])  # <= 0 (negative => lead)
            else:
                best_r, best_lag_bins = 0.0, 0
        else:
            # Pearson scan over negative lags (template leads SLA)
            best_r, best_lag_bins = 0.0, 0
            for k in range(1, max_lag_bins + 1):
                if k >= bins:
                    break
                a = x[:-k]  # template shifted left by k bins
                b = y[k:]   # SLA aligned
                asd, bsd = a.std(), b.std()
                if asd > 0 and bsd > 0:
                    r = float(np.corrcoef(a, b)[0, 1])
                    if not np.isnan(r) and r > best_r:
                        best_r, best_lag_bins = r, -k

        if best_r > 0.0:
            r_vals.append(best_r)
            lag_vals.append(best_lag_bins * freq_seconds)  # negative seconds = lead
            weights.append(float(x.sum()))

    if not r_vals:
        return tpl_id, 0.0, 0, 0

    w = np.asarray(weights, dtype=np.float64)
    w = w / (w.sum() if w.sum() > 0 else 1.0)
    r_weighted = float(np.dot(w, np.asarray(r_vals)))
    lag_weighted = int(round(float(np.dot(w, np.asarray(lag_vals)))))
    return tpl_id, r_weighted, lag_weighted, len(r_vals)


# ============================ Main Miner Class ============================

class TemplateGrayMiner:
    """Template-level miner that ranks Drain templates against SLA-anchored windows."""

    def __init__(self, config: T.Optional[MinerConfig] = None, **kwargs):
        self.config = config or MinerConfig(**kwargs)

        # Auto-downgrade correlation method if SciPy is unavailable
        if self.config.correlation_method == "xcorr" and not _HAS_SCIPY:
            warnings.warn("SciPy not found; falling back to 'pearson' correlation.", RuntimeWarning)
            self.config.correlation_method = "pearson"

        self.cards_: T.Optional[pd.DataFrame] = None
        self._df: T.Optional[pd.DataFrame] = None  # stored for diagnostics
        self._diagnostics: T.Dict[T.Any, TemplateDiagnostics] = {}

    # ----------------------------- Public API -----------------------------

    def fit(self, df: pd.DataFrame) -> "TemplateGrayMiner":
        """Run Steps 4-7 on the provided dataframe and populate self.cards_."""
        self._validate_input(df)

        if self.config.chunk_size and len(df) > self.config.chunk_size:
            self._fit_chunked(df)
        else:
            self._fit_single(df)

        return self

    def get_ranked(self, top_k: T.Optional[int] = None) -> pd.DataFrame:
        """Return ranked DataFrame with all metrics and provenance."""
        if self.cards_ is None:
            raise RuntimeError("Call fit(df) first.")
        return self.cards_.head(top_k) if top_k else self.cards_.copy()

    def to_yaml(self, top_k: int = 200) -> str:
        """Produce a review-friendly YAML string with provenance fields."""
        if self.cards_ is None:
            raise RuntimeError("Call fit(df) first.")
        data = []
        for _, r in self.cards_.head(top_k).iterrows():
            entry = {
                "template_id": r[self.config.template_col],
                "template_text": r.get("template_text", None),
                "score": float(r["score"]),
                "z": float(r["z"]),
                "lift": float(r["lift"]),
                "clarity": float(r["clarity"]),
                "lead_lag_sec": int(r.get("lead_lag_seconds", 0)),
                "lead_corr_pos": float(r.get("lead_corr_pos", 0.0)),
                "support_incident": int(r["c_i"]),
                "support_baseline": int(r["c_b"]),
                "N_incident": int(r["N_i"]),
                "N_baseline": int(r["N_b"]),
                "last_seen": str(r["last_seen"]),
                "first_seen": str(r["first_seen"]),
                "approved": False,
            }
            if "p_value" in r and not pd.isna(r["p_value"]):
                entry["p_value"] = float(r["p_value"])
                entry["q_value"] = float(r.get("q_value", 1.0))
                entry["significant_fdr"] = bool(r.get("significant_fdr", False))
            data.append(entry)

        if _HAS_YAML:
            return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
        else:
            lines = []
            for e in data:
                lines.append("-")
                for k, v in e.items():
                    lines.append(f"  {k}: {v}")
            return "\n".join(lines)

    def to_production_rules(self, top_k: int = 50) -> T.List[T.Dict[str, T.Any]]:
        """Generate production-ready alerting rules from ranked templates."""
        if self.cards_ is None:
            raise RuntimeError("Call fit(df) first.")
        rules = []
        for _, row in self.cards_.head(top_k).iterrows():
            historical_rate = row["c_i"] / max(row["N_i"], 1)
            threshold_occurrences = max(1, int(round(historical_rate * 100)))  # per 100 events
            rule = {
                "id": f"sla_predictor_{row[self.config.template_col]}",
                "template": row.get("template_text", row[self.config.template_col]),
                "condition": {
                    "template_id": row[self.config.template_col],
                    "min_occurrences": threshold_occurrences,
                    "window_minutes": 10,
                    "rate_threshold": historical_rate * 2.0,  # alert if rate doubles vs incident baseline
                },
                "alert": {
                    "severity": self._determine_severity(row),
                    "expected_sla_breach_minutes": abs(int(row.get("lead_lag_seconds", 0))) // 60,
                    "confidence": float(min(1.0, row["score"] / 10.0)),
                    "historical_accuracy": {
                        "true_positive_rate": float(row["c_i"] / max(row["c_i"] + row["c_b"], 1))
                    },
                },
                "metadata": {
                    "z_score": float(row["z"]),
                    "lift": float(row["lift"]),
                    "clarity": float(row["clarity"]),
                    "last_updated": datetime.utcnow().isoformat() + "Z",
                },
            }
            rules.append(rule)
        return rules

    def diagnose_template(self, template_id: T.Any) -> TemplateDiagnostics:
        """Provide detailed diagnostics for a specific template_id."""
        if self.cards_ is None:
            raise RuntimeError("Call fit(df) first.")
        if template_id in self._diagnostics:
            return self._diagnostics[template_id]

        row = self.cards_[self.cards_[self.config.template_col] == template_id]
        if row.empty:
            raise ValueError(f"Template {template_id} not found in results")
        r = row.iloc[0]

        diag = TemplateDiagnostics(
            template_id=template_id,
            template_text=r.get("template_text"),
            score_breakdown={
                "z_score": float(r["z"]),
                "lift": float(r["lift"]),
                "clarity": float(r["clarity"]),
                "lead_correlation": float(r.get("lead_corr_pos", 0.0)),
                "final_score": float(r["score"]),
            },
            temporal_pattern={
                "avg_lead_time_seconds": int(r.get("lead_lag_seconds", 0)),
                "incident_count": int(r["c_i"]),
                "baseline_count": int(r["c_b"]),
                "first_seen": str(r["first_seen"]),
                "last_seen": str(r["last_seen"]),
            },
            example_contexts=self._get_example_contexts(template_id),
            recommendations=self._generate_recommendations(r),
            risk_level=self._determine_risk_level(r),
        )
        self._diagnostics[template_id] = diag
        return diag

    # --------------------------- Internal: Fit paths ---------------------------

    def _fit_single(self, df: pd.DataFrame) -> None:
        """Single-pass fit (sufficient memory)."""
        self._df = self._prepare_df(df)

        counts_df, N_i, N_b = self._compute_counts(self._df)
        metrics_df = self._compute_metrics_full(self._df, counts_df, N_i, N_b)
        final_df = self._apply_filters_and_fdr(metrics_df)

        self.cards_ = final_df.sort_values("score", ascending=False).reset_index(drop=True)

    def _fit_chunked(self, df: pd.DataFrame) -> None:
        """
        Two-pass chunked fit for very large DataFrames:
          Pass A: aggregate counts & clarity data across chunks
          Rank preliminarily; select top-K for lead/lag
          Pass B: gather timestamps for top-K and compute lead/lag in parallel
        """
        chunk = int(self.config.chunk_size)
        n = len(df)
        n_chunks = (n + chunk - 1) // chunk
        if self.config.show_progress:
            print(f"Processing {n:,} rows in {n_chunks} chunks of {chunk:,}")

        # --- Pass A: aggregate counts + context distributions ---
        agg_tpl: T.Dict[T.Any, T.Dict[str, T.Any]] = {}
        agg_ctx_dim: T.Dict[str, T.Dict[T.Tuple[T.Any, str], int]] = defaultdict(lambda: defaultdict(int))
        N_i_total = 0
        N_total = 0

        have_context = bool(self.config.context_cols and all(c in df.columns for c in self.config.context_cols))

        for i in tqdm(range(n_chunks), desc="Pass A: counts & contexts", disable=not self.config.show_progress):
            part = df.iloc[i*chunk : min((i+1)*chunk, n)]
            part = self._prepare_df(part)

            # per-template counts
            g = part.groupby(self.config.template_col, observed=True)
            ci = g["is_incident"].sum()
            sz = g.size()
            first = g[self.config.time_col].min()
            last = g[self.config.time_col].max()

            for tpl_id in sz.index:
                c_i = int(ci.get(tpl_id, 0))
                total = int(sz.loc[tpl_id])
                s = agg_tpl.get(tpl_id)
                if s is None:
                    agg_tpl[tpl_id] = dict(
                        c_i=c_i, total=total,
                        first_seen=first.loc[tpl_id], last_seen=last.loc[tpl_id]
                    )
                else:
                    s["c_i"] += c_i
                    s["total"] += total
                    s["first_seen"] = min(s["first_seen"], first.loc[tpl_id])
                    s["last_seen"]  = max(s["last_seen"],  last.loc[tpl_id])

            N_i_total += int(part["is_incident"].sum())
            N_total   += int(len(part))

            # per-(template, context dimension) counts
            if have_context:
                for c in self.config.context_cols or []:
                    if c not in part.columns:
                        continue
                    counts_c = part.groupby([self.config.template_col, c], observed=True).size()
                    for (tpl_id, ctx_val), cnt in counts_c.items():
                        agg_ctx_dim[c][(tpl_id, str(ctx_val))] += int(cnt)
            else:
                # fallback: use SLA window id as context
                counts_w = part.groupby([self.config.template_col, "sla_window_id"], observed=True).size()
                for (tpl_id, wid), cnt in counts_w.items():
                    agg_ctx_dim["sla_window_id"][(tpl_id, str(int(wid)))] += int(cnt)

        # Build counts_df
        rows = []
        for tpl_id, s in agg_tpl.items():
            rows.append({
                self.config.template_col: tpl_id,
                "c_i": int(s["c_i"]),
                "c_b": int(s["total"] - s["c_i"]),
                "N_i": int(N_i_total),
                "N_b": int(N_total - N_i_total),
                "first_seen": s["first_seen"],
                "last_seen": s["last_seen"],
            })
        counts_df = pd.DataFrame(rows)

        # Build clarity from per-dimension context counts
        clarity_series = self._clarity_from_aggregates(agg_ctx_dim, counts_df)

        # Compute preliminary z, lift, score (no lead/lag yet)
        prelim = self._metrics_from_counts_only(counts_df, clarity_series)
        prelim = prelim.sort_values("score", ascending=False)

        # Choose top-K for lead/lag
        topK = set(prelim.head(self.config.top_k_leadlag or 2000)[self.config.template_col])

        # --- Pass B: gather timestamps for top-K and compute lead/lag ---
        if self.config.compute_lead_lag and len(topK) > 0:
            tpl_times: T.Dict[T.Any, T.List[np.ndarray]] = defaultdict(list)
            tpl_wids: T.Dict[T.Any, T.List[np.ndarray]] = defaultdict(list)
            windows_seen = []

            wcols = [self.config.window_start_col, self.config.window_end_col,
                     self.config.sla_start_col, self.config.sla_end_col, "sla_window_id"]

            for i in tqdm(range(n_chunks), desc="Pass B: gather lead/lag data", disable=not self.config.show_progress):
                part = df.iloc[i*chunk : min((i+1)*chunk, n)]
                part = self._prepare_df(part)
                windows_seen.append(part[wcols].drop_duplicates("sla_window_id"))

                sub = part[part[self.config.template_col].isin(topK)]
                if sub.empty:
                    continue
                sub = sub.sort_values(self.config.time_col)
                ts = sub[self.config.time_col].astype("int64")
                wid = sub["sla_window_id"].astype("int64")
                for tpl_id, g in sub.groupby(self.config.template_col, observed=True):
                    tpl_times[tpl_id].append(g[self.config.time_col].astype("int64").to_numpy())
                    tpl_wids[tpl_id].append(g["sla_window_id"].astype("int64").to_numpy())

            tpl_groups = {
                tpl: (np.concatenate(tpl_times[tpl]), np.concatenate(tpl_wids[tpl]))
                for tpl in tpl_times
            }
            windows = pd.concat(windows_seen).drop_duplicates("sla_window_id").sort_values("sla_window_id")
            windows_table = np.column_stack([
                windows["sla_window_id"].to_numpy(np.int64),
                windows[self.config.window_start_col].astype("int64").to_numpy(),
                windows[self.config.window_end_col].astype("int64").to_numpy(),
                windows[self.config.sla_start_col].astype("int64").to_numpy(),
                windows[self.config.sla_end_col].astype("int64").to_numpy(),
            ])

            lead_corr_s, lead_lag_s, win_contrib_s = self._leadlag_from_groups(tpl_groups, windows_table)

            # Merge lead/lag back into prelim and recompute final score
            prelim["lead_corr_pos"] = prelim[self.config.template_col].map(lead_corr_s).fillna(0.0).values
            prelim["lead_lag_seconds"] = prelim[self.config.template_col].map(lead_lag_s).fillna(0).astype(int).values
            prelim["windows_with_leadlag"] = prelim[self.config.template_col].map(win_contrib_s).fillna(0).astype(int).values
            prelim["score"] = self._score_enhanced(prelim["z"], prelim["lift"], prelim["clarity"], prelim["lead_corr_pos"])

        final_df = self._apply_filters_and_fdr(prelim)
        self.cards_ = final_df.sort_values("score", ascending=False).reset_index(drop=True)

    # --------------------------- Internal: Building blocks ---------------------------

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Basic input checks."""
        if len(df) == 0:
            raise ValueError("Empty dataframe provided.")
        req_cols = [
            self.config.time_col, self.config.template_col,
            self.config.sla_start_col, self.config.sla_end_col,
            self.config.window_start_col, self.config.window_end_col,
        ]
        missing = [c for c in req_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        template_coverage = df[self.config.template_col].notna().mean()
        if template_coverage < 0.5:
            warnings.warn(f"Low template coverage: {template_coverage:.1%} of rows have templates.", RuntimeWarning)

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse datetimes, validate windows, set categoricals, create window id & incident flag."""
        df = df.copy()

        _ensure_datetime_utc(df, [
            self.config.time_col, self.config.sla_start_col, self.config.sla_end_col,
            self.config.window_start_col, self.config.window_end_col
        ])

        # Validate window containment and order; drop invalid rows
        invalid = (
            (df[self.config.sla_start_col] < df[self.config.window_start_col]) |
            (df[self.config.sla_end_col] > df[self.config.window_end_col]) |
            (df[self.config.sla_start_col] > df[self.config.sla_end_col]) |
            (df[self.config.window_start_col] > df[self.config.window_end_col])
        )
        if invalid.any():
            nbad = int(invalid.sum())
            warnings.warn(f"Dropping {nbad} rows with invalid window definitions.", RuntimeWarning)
            df = df.loc[~invalid].copy()

        # Memory: categories
        df[self.config.template_col] = df[self.config.template_col].astype("category")
        if self.config.context_cols:
            for c in self.config.context_cols:
                if c in df.columns:
                    df[c] = df[c].astype("category")

        # SLA window ID (factorize 4 timestamps)
        wid_cols = [self.config.window_start_col, self.config.window_end_col,
                    self.config.sla_start_col, self.config.sla_end_col]
        df["sla_window_id"] = pd.factorize(df[wid_cols].astype(str).agg("|".join, axis=1))[0].astype(np.int64)

        # Incident flag
        t = df[self.config.time_col]
        df["is_incident"] = (t >= df[self.config.sla_start_col]) & (t < df[self.config.sla_end_col])

        return df

    # ---- Step 4: counts ----
    def _compute_counts(self, df: pd.DataFrame) -> T.Tuple[pd.DataFrame, int, int]:
        N_i = int(df["is_incident"].sum())
        N_b = int(len(df) - N_i)
        if N_i == 0 or N_b == 0:
            raise ValueError(f"Insufficient class balance: N_i={N_i}, N_b={N_b}")

        grp = df.groupby(self.config.template_col, observed=True)
        c_i = grp["is_incident"].sum().astype(np.int64)
        total = grp.size().astype(np.int64)
        c_b = total - c_i

        first_seen = grp[self.config.time_col].min()
        last_seen = grp[self.config.time_col].max()

        out = pd.DataFrame({
            self.config.template_col: c_i.index,
            "c_i": c_i.values,
            "c_b": c_b.values,
            "N_i": N_i,
            "N_b": N_b,
            "first_seen": first_seen.values,
            "last_seen": last_seen.values,
        })

        if self.config.template_text_col and self.config.template_text_col in df.columns:
            text_map = df.dropna(subset=[self.config.template_text_col]).drop_duplicates(self.config.template_col)[
                [self.config.template_col, self.config.template_text_col]
            ]
            out = out.merge(text_map, on=self.config.template_col, how="left")
        return out, N_i, N_b

    # ---- Step 5: clarity (weighted) ----
    def _compute_clarity_weighted(self, df: pd.DataFrame) -> pd.Series:
        """
        Clarity = weighted sum of (1 - normalized entropy) across context dimensions.
        If no context cols, fall back to SLA window concentration.
        """
        if not (self.config.context_cols and all(c in df.columns for c in self.config.context_cols)):
            # Fallback: SLA window concentration
            tmp = df[[self.config.template_col, "sla_window_id"]].rename(columns={
                self.config.template_col: "tpl", "sla_window_id": "ctx"
            })
            counts = tmp.groupby(["tpl", "ctx"], observed=True).size().rename("cnt").reset_index()
            tot = counts.groupby("tpl")["cnt"].sum().rename("tot")
            counts = counts.merge(tot, on="tpl", how="left")
            p = counts["cnt"] / counts["tot"].replace(0, np.nan)
            ent = (-p * np.log(p + 1e-12)).groupby(counts["tpl"]).sum()
            K = counts.groupby("tpl")["ctx"].nunique()
            Hn = (ent / np.log(K.replace(0, np.nan))).replace([np.inf, -np.inf, np.nan], 0.0)
            clarity = (1.0 - Hn).clip(0.0, 1.0)
            # penalize rare templates
            occ = df[self.config.template_col].value_counts()
            rare = occ[occ < self.config.min_template_occurrences].index
            clarity.loc[clarity.index.intersection(rare)] = np.maximum(
                clarity.loc[clarity.index.intersection(rare)] - 0.1, 0.0
            )
            return clarity

        # Weighted per-dimension clarity
        dims = [c for c in (self.config.context_cols or []) if c in df.columns]
        weights = self.config.context_weights or {}
        w = np.array([max(0.0, float(weights.get(c, 1.0))) for c in dims], dtype=float)
        if w.sum() == 0:
            w[:] = 1.0
        w = w / w.sum()

        clarity_parts = []
        for c, wc in zip(dims, w):
            sub = df[[self.config.template_col, c]].rename(columns={self.config.template_col: "tpl", c: "ctx"})
            counts = sub.groupby(["tpl", "ctx"], observed=True).size().rename("cnt").reset_index()
            tot = counts.groupby("tpl")["cnt"].sum().rename("tot")
            counts = counts.merge(tot, on="tpl", how="left")
            p = counts["cnt"] / counts["tot"].replace(0, np.nan)
            ent = (-p * np.log(p + 1e-12)).groupby(counts["tpl"]).sum()
            K = counts.groupby("tpl")["ctx"].nunique()
            Hn = (ent / np.log(K.replace(0, np.nan))).replace([np.inf, -np.inf, np.nan], 0.0)
            clarity_parts.append(wc * (1.0 - Hn))

        # sum weighted parts; align across all templates
        all_tpls = df[self.config.template_col].cat.categories
        clarity = sum(cp.reindex(all_tpls, fill_value=0.5) for cp in clarity_parts).clip(0.0, 1.0)

        # penalize very rare templates
        occ = df[self.config.template_col].value_counts()
        rare = occ[occ < self.config.min_template_occurrences].index
        clarity.loc[clarity.index.intersection(rare)] = np.maximum(
            clarity.loc[clarity.index.intersection(rare)] - 0.1, 0.0
        )
        return clarity

    def _clarity_from_aggregates(
        self,
        agg_ctx_dim: T.Dict[str, T.Dict[T.Tuple[T.Any, str], int]],
        counts_df: pd.DataFrame
    ) -> pd.Series:
        """
        Build clarity from aggregated per-dimension context counts (chunked pass A).
        """
        # If contexts present, compute weighted clarity across dims
        dims = list(agg_ctx_dim.keys())
        if not dims:
            # No contexts aggregated -> default 0.5
            return pd.Series(0.5, index=counts_df[self.config.template_col])

        weights = self.config.context_weights or {}
        w = np.array([max(0.0, float(weights.get(c, 1.0))) for c in dims], dtype=float)
        if w.sum() == 0:
            w[:] = 1.0
        w = w / w.sum()

        clarity_parts = []
        for c, wc in zip(dims, w):
            items = agg_ctx_dim[c]
            if not items:
                continue
            ctx_df = pd.DataFrame(
                [(tpl, ctx, cnt) for (tpl, ctx), cnt in items.items()],
                columns=["tpl", "ctx", "cnt"]
            )
            tot = ctx_df.groupby("tpl")["cnt"].sum().rename("tot")
            ctx_df = ctx_df.merge(tot, on="tpl", how="left")
            p = ctx_df["cnt"] / ctx_df["tot"].replace(0, np.nan)
            ent = (-p * np.log(p + 1e-12)).groupby(ctx_df["tpl"]).sum()
            K = ctx_df.groupby("tpl")["ctx"].nunique()
            Hn = (ent / np.log(K.replace(0, np.nan))).replace([np.inf, -np.inf, np.nan], 0.0)
            clarity_parts.append(wc * (1.0 - Hn))

        tpl_idx = counts_df[self.config.template_col]
        clarity = sum(cp.reindex(tpl_idx, fill_value=0.5) for cp in clarity_parts).clip(0.0, 1.0)

        # rare template penalty
        total_occ = counts_df["c_i"] + counts_df["c_b"]
        rare_mask = total_occ < self.config.min_template_occurrences
        clarity.loc[rare_mask] = np.maximum(clarity.loc[rare_mask] - 0.1, 0.0)
        return clarity

    # ---- Step 5: z, lift, lead/lag, final score ----
    def _score_enhanced(self, z: pd.Series, lift: pd.Series, clarity: pd.Series, lead_corr_pos: pd.Series) -> pd.Series:
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))
        z_contrib = sigmoid(self.config.z_weight * z)
        lift_contrib = np.minimum(lift.fillna(self.config.max_lift), self.config.max_lift)
        clarity_contrib = np.power(clarity.clip(0, 1), self.config.clarity_power)
        lead_bonus = 1.0 + self.config.beta_lead * lead_corr_pos.clip(0, 1)
        return z_contrib * lift_contrib * clarity_contrib * lead_bonus

    def _metrics_from_counts_only(self, counts_df: pd.DataFrame, clarity_series: pd.Series) -> pd.DataFrame:
        """Compute z, lift, preliminary score from counts + clarity (no lead/lag)."""
        ci = counts_df["c_i"].astype(float)
        cb = counts_df["c_b"].astype(float)
        N_i = float(counts_df["N_i"].iloc[0]) if len(counts_df) else 0.0
        N_b = float(counts_df["N_b"].iloc[0]) if len(counts_df) else 0.0

        p_i = (ci + 0.5) / (N_i + 1.0)
        p_b = (cb + 0.5) / (N_b + 1.0)
        z = (_logit_series(p_i) - _logit_series(p_b)) / np.sqrt(1.0/(ci + 0.5) + 1.0/(cb + 0.5))

        with np.errstate(divide='ignore', invalid='ignore'):
            lift = (ci / max(N_i, 1.0)) / (cb / max(N_b, 1.0))
            lift = lift.replace([np.inf, -np.inf], np.nan).fillna(self.config.max_lift)

        out = counts_df.copy()
        out["z"] = z.values
        out["lift"] = np.minimum(lift, self.config.max_lift)
        out["clarity"] = clarity_series.reindex(out[self.config.template_col]).fillna(0.5).values
        out["lead_corr_pos"] = 0.0
        out["lead_lag_seconds"] = 0
        out["windows_with_leadlag"] = 0
        out["score"] = self._score_enhanced(out["z"], out["lift"], out["clarity"], out["lead_corr_pos"])
        return out

    def _compute_metrics_full(self, df: pd.DataFrame, counts_df: pd.DataFrame, N_i: int, N_b: int) -> pd.DataFrame:
        """Compute z, lift, clarity, lead/lag, and final score (single-pass variant)."""
        ci = counts_df["c_i"].astype(float)
        cb = counts_df["c_b"].astype(float)

        p_i = (ci + 0.5) / (N_i + 1.0)
        p_b = (cb + 0.5) / (N_b + 1.0)
        z = (_logit_series(p_i) - _logit_series(p_b)) / np.sqrt(1.0/(ci + 0.5) + 1.0/(cb + 0.5))

        with np.errstate(divide='ignore', invalid='ignore'):
            lift = (ci / max(N_i, 1.0)) / (cb / max(N_b, 1.0))
            lift = lift.replace([np.inf, -np.inf], np.nan).fillna(self.config.max_lift)

        clarity = self._compute_clarity_weighted(df)

        # Lead/lag (all templates)
        if self.config.compute_lead_lag and pd.Timedelta(self.config.lead_lag_horizon) > pd.Timedelta(0):
            lead_corr, lead_lag_sec, win_count = self._compute_lead_lag_parallel(df, counts_df)
        else:
            idx = counts_df[self.config.template_col]
            lead_corr = pd.Series(0.0, index=idx)
            lead_lag_sec = pd.Series(0, index=idx, dtype=int)
            win_count = pd.Series(0, index=idx, dtype=int)

        score = self._score_enhanced(z, lift,
                                     clarity.reindex(counts_df[self.config.template_col]).fillna(0.5),
                                     lead_corr)

        out = counts_df.copy()
        out["z"] = z.values
        out["lift"] = np.minimum(lift, self.config.max_lift)
        out["clarity"] = clarity.reindex(out[self.config.template_col]).fillna(0.5).values
        out["lead_corr_pos"] = lead_corr.reindex(out[self.config.template_col]).fillna(0.0).values
        out["lead_lag_seconds"] = lead_lag_sec.reindex(out[self.config.template_col]).fillna(0).astype(int).values
        out["windows_with_leadlag"] = win_count.reindex(out[self.config.template_col]).fillna(0).astype(int).values
        out["score"] = score.values
        return out

    # ---- Lead/lag orchestration ----
    def _compute_lead_lag_parallel(self, df: pd.DataFrame, counts_df: pd.DataFrame) -> T.Tuple[pd.Series, pd.Series, pd.Series]:
        """Compute lead/lag correlations for all templates in df."""
        # Prepare windows table
        wcols = [self.config.window_start_col, self.config.window_end_col,
                 self.config.sla_start_col, self.config.sla_end_col, "sla_window_id"]
        windows = df[wcols].drop_duplicates("sla_window_id").sort_values("sla_window_id")
        windows_table = np.column_stack([
            windows["sla_window_id"].to_numpy(np.int64),
            windows[self.config.window_start_col].astype("int64").to_numpy(),
            windows[self.config.window_end_col].astype("int64").to_numpy(),
            windows[self.config.sla_start_col].astype("int64").to_numpy(),
            windows[self.config.sla_end_col].astype("int64").to_numpy(),
        ])

        # Minimal per-template payload
        df_min = df[[self.config.template_col, self.config.time_col, "sla_window_id"]].copy()
        df_min[self.config.time_col] = df_min[self.config.time_col].astype("int64")
        df_min["sla_window_id"] = df_min["sla_window_id"].astype("int64")

        tpl_groups: T.Dict[T.Any, T.Tuple[np.ndarray, np.ndarray]] = {}
        for tpl_id, g in tqdm(df_min.groupby(self.config.template_col, observed=True),
                              disable=not self.config.show_progress,
                              desc="Preparing lead/lag payload"):
            g = g.sort_values(self.config.time_col)
            tpl_groups[tpl_id] = (
                g[self.config.time_col].to_numpy(copy=False),
                g["sla_window_id"].to_numpy(copy=False),
            )

        return self._leadlag_from_groups(tpl_groups, windows_table)

    def _leadlag_from_groups(
        self,
        tpl_groups: T.Dict[T.Any, T.Tuple[np.ndarray, np.ndarray]],
        windows_table: np.ndarray
    ) -> T.Tuple[pd.Series, pd.Series, pd.Series]:
        """Run lead/lag workers over prepared per-template payloads."""
        freq_td = pd.Timedelta(self.config.freq)
        freq_seconds = int(freq_td.total_seconds())
        max_lag_bins = max(1, int(pd.Timedelta(self.config.lead_lag_horizon) / freq_td))

        lead_corr: T.Dict[T.Any, float] = {}
        lead_lag_sec: T.Dict[T.Any, int] = {}
        win_count: T.Dict[T.Any, int] = {}

        if self.config.n_jobs == 1:
            it = tpl_groups.items()
            if self.config.show_progress:
                it = tqdm(it, desc="Lead/lag (1 process)")
            for tpl_id, (ts_ns, wids) in it:
                _, r_w, lag_w, nwin = _leadlag_worker_enhanced(
                    tpl_id, ts_ns, wids, windows_table, freq_seconds, max_lag_bins, self.config.correlation_method
                )
                lead_corr[tpl_id] = r_w
                lead_lag_sec[tpl_id] = lag_w
                win_count[tpl_id] = nwin
        else:
            with ProcessPoolExecutor(max_workers=self.config.n_jobs) as ex:
                futures = []
                for tpl_id, (ts_ns, wids) in tpl_groups.items():
                    futures.append(ex.submit(
                        _leadlag_worker_enhanced,
                        tpl_id, ts_ns, wids, windows_table, freq_seconds, max_lag_bins, self.config.correlation_method
                    ))
                for fut in tqdm(as_completed(futures), total=len(futures),
                                disable=not self.config.show_progress,
                                desc=f"Lead/lag ({self.config.n_jobs} workers)"):
                    tpl_id, r_w, lag_w, nwin = fut.result()
                    lead_corr[tpl_id] = r_w
                    lead_lag_sec[tpl_id] = lag_w
                    win_count[tpl_id] = nwin

        lead_corr_s = pd.Series(lead_corr, dtype=float)
        lead_lag_s = pd.Series(lead_lag_sec, dtype=int)
        win_contrib_s = pd.Series(win_count, dtype=int)
        return lead_corr_s, lead_lag_s, win_contrib_s

    # ---- Step 6: filters + FDR ----
    def _apply_filters_and_fdr(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # Stability filter
        keep = out["c_i"] >= self.config.min_support_incident
        if self.config.show_progress and (~keep).sum() > 0:
            print(f"Filtered {int((~keep).sum())} templates with insufficient incident support (< {self.config.min_support_incident}).")
        out = out.loc[keep].copy()

        # Fisher + BH-FDR
        if self.config.alpha_fdr is not None and len(out) > 0:
            pvals = []
            iterator = out.itertuples(index=False)
            if self.config.show_progress:
                iterator = tqdm(iterator, total=len(out), desc="Fisher p-values")
            for r in iterator:
                # r fields by column order:
                # [template_id, c_i, c_b, N_i, N_b, ...]
                c_i = int(getattr(r, "c_i"))
                c_b = int(getattr(r, "c_b"))
                N_i = int(getattr(r, "N_i"))
                N_b = int(getattr(r, "N_b"))
                p = _fisher_one_sided(N=N_i + N_b, K=c_i + c_b, n=N_i, k=c_i)
                pvals.append(p)
            out["p_value"] = np.asarray(pvals, dtype=float)
            out["q_value"] = _bh_fdr(out["p_value"].values)
            out["significant_fdr"] = out["q_value"] <= float(self.config.alpha_fdr)
            if self.config.show_progress:
                print(f"FDR: {int(out['significant_fdr'].sum())}/{len(out)} significant at alpha={self.config.alpha_fdr}.")
        return out

    # --------------------------- Diagnostics helpers ---------------------------

    def _get_example_contexts(self, template_id: T.Any, max_examples: int = 6) -> T.List[T.Dict[str, T.Any]]:
        if self._df is None:
            return []
        sub = self._df[self._df[self.config.template_col] == template_id]
        if sub.empty:
            return []
        inc = sub[sub["is_incident"]].head(max_examples // 2)
        base = sub[~sub["is_incident"]].head(max_examples // 2)
        examples = []
        for lbl, rows in (("incident", inc), ("baseline", base)):
            for _, r in rows.iterrows():
                ctx = {"type": lbl, "timestamp": str(r[self.config.time_col])}
                if self.config.context_cols:
                    for c in self.config.context_cols:
                        if c in r:
                            ctx[c] = str(r[c])
                examples.append(ctx)
        return examples

    def _generate_recommendations(self, row: pd.Series) -> T.List[str]:
        rec: T.List[str] = []
        lead_secs = int(row.get("lead_lag_seconds", 0))
        score = float(row["score"])
        lift = float(row["lift"])
        clarity = float(row["clarity"])

        if score > 8:
            rec.append(f"HIGH PRIORITY: Create alert; expected lead ~{abs(lead_secs)//60} min")
        elif score > 5:
            rec.append("MEDIUM PRIORITY: Monitor closely; validate before paging")

        if lead_secs < -300:
            rec.append(f"Good early warning ({abs(lead_secs)//60} min)")
        elif lead_secs < 0:
            rec.append("Short lead time; rapid response needed")

        if clarity > 0.8:
            rec.append("Highly focused pattern; easy to route to owning team")
        elif clarity < 0.3:
            rec.append("Widely distributed; consider adding context gates")

        if lift > 10:
            rec.append(f"Extremely strong signal ({lift:.1f}x baseline)")

        return rec or ["Review historical incidents for validation"]

    def _determine_risk_level(self, row: pd.Series) -> str:
        score, lift = float(row["score"]), float(row["lift"])
        if score > 8 and lift > 10:
            return "CRITICAL"
        if score > 6 and lift > 5:
            return "HIGH"
        if score > 4:
            return "MEDIUM"
        if score > 2:
            return "LOW"
        return "INFO"

    def _determine_severity(self, row: pd.Series) -> str:
        score = float(row["score"])
        q = float(row.get("q_value", 1.0)) if "q_value" in row else 1.0
        if score > 8 and q < 0.01:
            return "critical"
        if score > 6 and q < 0.05:
            return "high"
        if score > 4:
            return "medium"
        return "low"


# =============================== Example Usage ===============================

if __name__ == "__main__":
    # Windows note: keep this guard when using n_jobs > 1
    # Example configuration (tune for your environment)
    cfg = MinerConfig(
        time_col="timestamp",
        template_col="log_template_id",
        sla_start_col="sla_breach_start",
        sla_end_col="sla_breach_end",
        window_start_col="from_time",
        window_end_col="end_time",
        context_cols=["provider", "component", "level"],   # optional but recommended
        template_text_col="template_text",                 # optional

        freq="1min",
        lead_lag_horizon="10min",

        z_weight=0.35,
        beta_lead=0.3,
        clarity_power=1.2,
        max_lift=10.0,

        min_support_incident=5,
        min_template_occurrences=10,
        alpha_fdr=0.05,

        compute_lead_lag=True,
        correlation_method=("xcorr" if _HAS_SCIPY else "pearson"),
        n_jobs=max(1, (os.cpu_count() or 2) // 2),
        top_k_leadlag=2000,       # compute lead/lag only for the most promising templates
        chunk_size=5_000_000,     # switch to two-pass when DataFrame > 5M rows
        show_progress=True,

        context_weights={"level": 2.0, "component": 1.5, "provider": 1.0}
    )

    # Instantiate miner
    miner = TemplateGrayMiner(cfg)

    # --- Example: expected DataFrame schema ---
    print("Expected DataFrame columns & dtypes:")
    print({
        "timestamp": "datetime64[ns] or str parseable to UTC",
        "log_template_id": "str/int (will be cast to category)",
        "template_text": "str (optional)",
        "sla_breach_start": "datetime64[ns] or str",
        "sla_breach_end": "datetime64[ns] or str",
        "from_time": "datetime64[ns] or str",
        "end_time": "datetime64[ns] or str",
        "provider": "str (optional)",
        "component": "str (optional)",
        "level": "str (optional)",
    })
    print("\n# Usage:")
    print("# df = pd.read_parquet('sla_breach_logs.parquet')  # or CSV, etc.")
    print("# miner.fit(df)")
    print("# top = miner.get_ranked(50)")
    print("# print(top[['template_text','score','z','lift','clarity','lead_lag_seconds']].head(10))")
    print("# yaml_blob = miner.to_yaml(top_k=200)")
    print("# rules = miner.to_production_rules(top_k=50)")
    print("# diag = miner.diagnose_template(top.iloc[0]['log_template_id'])")