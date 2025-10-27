#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anomaly Ranker Foundation — Statistical Ensemble Library
========================================================

This module provides reusable statistical anomaly detection utilities.
It is designed to be imported by anomaly_ranker_engine.py and other tools.

Features:
- 8 statistical detection methods (z-score, GLR, PMI, Cohen's d, CUSUM, etc.)
- Robust baseline building with hour-of-day patterns
- Parallel processing infrastructure for scalability
- Memory-efficient pivot table operations (float32)
- Optional DuckDB integration for fast raw log aggregation

Can be run standalone for lightweight statistical analysis:
  python anomaly_ranker_foundation.py \
    --input hourly_counts.csv \
    --incident-start 2025-08-12T20:00:00Z \
    --incident-end   2025-08-12T23:00:00Z \
    --workers 8 --parallel auto

Or imported as a library:
  from anomaly_ranker_foundation import rank_events_parallel, build_baselines

Primary users:
  - anomaly_ranker_engine.py (comprehensive analysis driver)
  - Custom analyzers (can import individual utilities)
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

try:
    import duckdb  # optional, speeds up raw -> hourly aggregation massively
    HAVE_DUCKDB = True
except Exception:
    HAVE_DUCKDB = False

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# ---------------- Utilities ----------------

def parse_dt(s: str) -> pd.Timestamp:
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    if pd.isna(ts):
        ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Could not parse datetime: {s}")
    return ts.tz_convert(None) if ts.tz is not None else ts

def as_hourly_index(ts_min: pd.Timestamp, ts_max: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(ts_min.floor("H"), ts_max.ceil("H"), freq="H")

def mad(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))

def iqr(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    q75, q25 = np.percentile(x, [75, 25])
    return q75 - q25

def robust_scale(x: np.ndarray) -> float:
    s = iqr(x)
    if not np.isnan(s) and s > 0:
        return s / 1.349
    s = mad(x)
    if not np.isnan(s) and s > 0:
        return s
    s = np.nanstd(x)
    return s if s > 0 else 1.0

def safe_div(a: float, b: float) -> float:
    if b == 0 or np.isnan(b):
        return 0.0
    return float(a) / float(b)

def rolling_runlen(mask: np.ndarray) -> int:
    max_run = cur = 0
    for v in mask:
        if v:
            cur += 1; max_run = max(max_run, cur)
        else:
            cur = 0
    return max_run

def detect_global_blackouts(piv: pd.DataFrame, frac_zero_thresh: float = 0.8) -> pd.Index:
    zero_share = (piv.fillna(0) == 0).sum(axis=1) / (piv.notna().sum(axis=1).replace(0, np.nan))
    zero_share = zero_share.fillna(0.0)
    return zero_share.index[zero_share >= frac_zero_thresh]

# ---------------- Baselines ----------------

@dataclass
class Baseline:
    mu_hod: pd.DataFrame
    sig_hod: pd.DataFrame
    mu_global: pd.Series
    sig_global: pd.Series

def build_baselines(piv: pd.DataFrame) -> Baseline:
    idx = piv.index
    by_hod_mu = []; by_hod_sig = []
    for h in range(24):
        sl = piv[idx.hour == h]
        by_hod_mu.append(sl.median(axis=0, skipna=True))
        by_hod_sig.append(sl.apply(lambda c: robust_scale(c.values), axis=0))
    mu_hod = pd.DataFrame(by_hod_mu); sig_hod = pd.DataFrame(by_hod_sig)
    mu_hod.index = range(24); sig_hod.index = range(24)
    mu_global = piv.median(axis=0, skipna=True)
    sig_global = piv.apply(lambda c: robust_scale(c.values), axis=0)
    sig_hod = sig_hod.replace(0, 1.0).fillna(1.0)
    sig_global = sig_global.replace(0, 1.0).fillna(1.0)
    return Baseline(mu_hod, sig_hod, mu_global, sig_global)

def baseline_for(piv: pd.DataFrame, bl: Baseline) -> Tuple[pd.DataFrame, pd.DataFrame]:
    hours = piv.index.hour
    mu = pd.DataFrame(index=piv.index, columns=piv.columns, dtype=np.float32)
    sig = pd.DataFrame(index=piv.index, columns=piv.columns, dtype=np.float32)
    for h in range(24):
        mask = hours == h
        if not mask.any(): continue
        mu.loc[mask] = bl.mu_hod.loc[h].reindex(piv.columns).values.astype(np.float32)
        sig.loc[mask] = bl.sig_hod.loc[h].reindex(piv.columns).values.astype(np.float32)
    mu = mu.fillna(bl.mu_global.astype(np.float32))
    sig = sig.fillna(bl.sig_global.astype(np.float32)).replace(0, 1.0)
    return mu, sig

# ---------------- Features ----------------

def poisson_glr(inc_counts: np.ndarray, base_counts: np.ndarray) -> float:
    inc_counts = inc_counts[~np.isnan(inc_counts)]
    base_counts = base_counts[~np.isnan(base_counts)]
    if inc_counts.size == 0 or base_counts.size == 0:
        return 0.0
    k1, k0 = inc_counts.sum(), base_counts.sum()
    t1, t0 = max(inc_counts.size,1), max(base_counts.size,1)
    lam1, lam0 = safe_div(k1,t1), safe_div(k0,t0)
    if lam0 <= 0 or lam1 <= 0: return 0.0
    llr = 2.0 * (k1 * np.log(lam1/lam0) - (lam1 - lam0) * t1)
    return max(0.0, float(llr))

def cohen_d(inc_vals: np.ndarray, ref_vals: np.ndarray) -> float:
    inc_vals = inc_vals[~np.isnan(inc_vals)]
    ref_vals = ref_vals[~np.isnan(ref_vals)]
    if inc_vals.size == 0 or ref_vals.size == 0: return 0.0
    m1, m0 = inc_vals.mean(), ref_vals.mean()
    v1, v0 = np.var(inc_vals, ddof=1), np.var(ref_vals, ddof=1)
    n1, n0 = max(inc_vals.size,1), max(ref_vals.size,1)
    sp = np.sqrt(((n1-1)*v1 + (n0-1)*v0) / max(n1+n0-2,1))
    if sp == 0 or np.isnan(sp): return 0.0
    return float((m1-m0)/sp)

def pmi_overrep(inc_counts: np.ndarray, all_counts: np.ndarray, inc_hours: int, total_hours: int) -> float:
    inc_sum, all_sum = np.nansum(inc_counts), np.nansum(all_counts)
    if all_sum <= 0 or inc_hours <= 0 or total_hours <= 0:
        return 0.0
    p_inc_event = safe_div(inc_sum, all_sum)
    p_inc_time = safe_div(inc_hours, total_hours)
    if p_inc_event <= 0 or p_inc_time <= 0:
        return 0.0
    return float(np.log(p_inc_event / p_inc_time))

def cusum_intensity(series: np.ndarray) -> float:
    series = series.astype(float)
    series = series[~np.isnan(series)]
    if series.size == 0: return 0.0
    med = np.median(series)
    dev = series - med
    cplus = np.maximum.accumulate(np.cumsum(np.maximum(dev, 0)))
    cminus = np.maximum.accumulate(np.cumsum(np.maximum(-dev, 0)))
    return float(max(cplus.max(initial=0.0), cminus.max(initial=0.0)))

def robust_z(values: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return (values - mu) / np.where(sig <= 0, 1.0, sig)

# ---------------- Parallel feature computation ----------------

def compute_features_for_events(args):
    # Unpack tuple for pickling friendliness
    (col_names, S, M, SG, idx_vals, inc_mask_vals, pre_mask_vals,
     same_hod_ref_vals, inc_end_pos, hours_var_window, inc_hours, total_hours) = args

    results = []
    for j, ev in enumerate(col_names):
        s = S[:, j]
        m = M[:, j]
        sg = SG[:, j]
        inc_vals = s[inc_mask_vals]
        base_vals = s[pre_mask_vals]

        z = robust_z(s, m, sg)
        inc_z = z[inc_mask_vals]
        z_pos = np.nanmax(inc_z) if inc_z.size else 0.0
        z_neg = np.nanmin(inc_z) if inc_z.size else 0.0
        z_neg_abs = abs(z_neg) if not np.isnan(z_neg) else 0.0

        glr = poisson_glr(inc_vals, base_vals)

        thr = m + 2.0 * sg
        above = (s > thr).astype(bool)
        runlen = 0
        cur = 0
        for flag in above[inc_mask_vals]:
            if flag:
                cur += 1; runlen = max(runlen, cur)
            else:
                cur = 0

        pmi = pmi_overrep(inc_vals, s, inc_hours, total_hours)

        ref_vals = s[same_hod_ref_vals]
        d = cohen_d(inc_vals, ref_vals)

        end_idx = inc_end_pos
        w = hours_var_window
        a_start = max(0, end_idx - w + 1)
        b_start = max(0, a_start - 2*w)
        a = s[a_start:end_idx+1]
        b = s[b_start:a_start]
        var_ratio = safe_div(np.nanvar(a), np.nanvar(b)) if b.size > 3 else 0.0

        left = max(0, end_idx - 24)
        right = min(len(s), end_idx + 24)
        cus = cusum_intensity(s[left:right])

        cov_inc = safe_div(np.count_nonzero(~np.isnan(inc_vals)), inc_hours if inc_hours>0 else 1)
        cov_pre = safe_div(np.count_nonzero(~np.isnan(base_vals)), np.count_nonzero(pre_mask_vals))
        stability = 1.0 / (1.0 + safe_div(np.nanstd(base_vals), (np.nanmean(base_vals) + 1e-6)))

        results.append([ev, float(0 if np.isnan(z_pos) else z_pos),
                           float(0 if np.isnan(z_neg_abs) else z_neg_abs),
                           float(glr), int(runlen), float(pmi), float(d),
                           float(0 if np.isnan(var_ratio) else var_ratio),
                           float(cus),
                           float(cov_inc), float(cov_pre), float(stability),
                           float(np.nanmean(base_vals)), float(np.nanmean(inc_vals)), float(np.nansum(inc_vals))])
    return results

# ---------------- Scoring ----------------

def rank_events_parallel(piv: pd.DataFrame,
                         incident_start: pd.Timestamp,
                         incident_end: pd.Timestamp,
                         workers: int = 0,
                         parallel: str = "auto",
                         hours_pre_baseline: int = 48,
                         hours_var_window: int = 24,
                         blackout_frac: float = 0.8) -> pd.DataFrame:
    # Blackouts -> NaN
    blackout_hours = detect_global_blackouts(piv, frac_zero_thresh=blackout_frac)
    if len(blackout_hours) > 0:
        piv = piv.copy()
        piv.loc[blackout_hours] = np.nan

    bl = build_baselines(piv)
    mu, sig = baseline_for(piv, bl)

    inc_mask = (piv.index >= incident_start) & (piv.index <= incident_end)
    total_hours = piv.shape[0]
    inc_hours = int(inc_mask.sum())
    pre_start = max(piv.index.min(), incident_start - pd.Timedelta(hours=hours_pre_baseline))
    pre_mask = (piv.index >= pre_start) & (piv.index < incident_start)

    hod = piv.index.hour
    inc_hods = set(hod[inc_mask].tolist())
    same_hod_ref_mask = hod.isin(list(inc_hods)) & (~inc_mask)

    # Prepare arrays (float32 to reduce memory)
    S = piv.values.astype(np.float32, copy=False)
    M = mu.values.astype(np.float32, copy=False)
    SG = sig.values.astype(np.float32, copy=False)
    col_names = list(piv.columns)

    # position of incident_end within index
    inc_end_pos = np.searchsorted(piv.index.values, incident_end.to_datetime64(), side="right") - 1
    inc_end_pos = max(min(inc_end_pos, len(piv.index)-1), 0)

    inc_mask_vals = inc_mask.values
    pre_mask_vals = pre_mask.values
    same_hod_ref_vals = same_hod_ref_mask.values

    # Decide engine
    if parallel == "auto":
        engine = "process" if workers and workers > 1 else "off"
    else:
        engine = parallel

    # Chunk columns
    n_cols = len(col_names)
    if workers is None or workers < 1:
        workers = 1
    chunk_size = max(1, math.ceil(n_cols / (workers * 4)))  # 4 chunks per worker

    tasks = []
    for start in range(0, n_cols, chunk_size):
        end = min(start + chunk_size, n_cols)
        sub_cols = col_names[start:end]
        args = (sub_cols, S[:, start:end], M[:, start:end], SG[:, start:end],
                piv.index.values, inc_mask_vals, pre_mask_vals, same_hod_ref_vals,
                inc_end_pos, hours_var_window, inc_hours, total_hours)
        tasks.append(args)

    results = []
    if engine == "off" or workers == 1:
        for t in tasks:
            results.extend(compute_features_for_events(t))
    elif engine == "thread":
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(compute_features_for_events, t) for t in tasks]
            for f in as_completed(futures):
                results.extend(f.result())
    else:  # process
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(compute_features_for_events, t) for t in tasks]
            for f in as_completed(futures):
                results.extend(f.result())

    feat_df = pd.DataFrame(results, columns=[
        "event_id","z_max","z_drop_max","glr","runlen_2sigma","pmi_overrep","cohen_d",
        "var_ratio","cusum","cov_incident","cov_pre","stability","base_mean","inc_mean","inc_sum"
    ]).set_index("event_id")

    # Robust standardize & weight
    def rstd(col: pd.Series) -> pd.Series:
        med = np.nanmedian(col.values)
        sc = mad(col.values); sc = 1.0 if (np.isnan(sc) or sc <= 0) else sc
        return (col - med) / sc

    to_scale = ["z_max","z_drop_max","glr","runlen_2sigma","pmi_overrep","cohen_d","var_ratio","cusum"]
    scaled_df = pd.DataFrame({k: rstd(feat_df[k]) for k in to_scale}, index=feat_df.index).fillna(0.0).clip(-10,10)

    weights = {
        "z_max": 0.35, "glr": 0.25, "pmi_overrep": 0.15, "cohen_d": 0.10,
        "runlen_2sigma": 0.07, "cusum": 0.05, "var_ratio": 0.02, "z_drop_max": 0.01
    }
    weighted = sum(scaled_df[k]*w for k,w in weights.items())

    conf = (feat_df["cov_incident"].clip(0,1) * feat_df["cov_pre"].clip(0,1) * feat_df["stability"].clip(0,1))
    feat_df["final_score"] = (weighted * conf).fillna(0.0)

    # Reasons
    rc = list(weights.keys())
    tmp = scaled_df[rc].copy()
    tmp["final_score"] = feat_df["final_score"]
    reasons = []
    for ev, row in tmp.iterrows():
        contribs = {k: float(row[k] * weights[k]) for k in rc}
        top2 = sorted(contribs.items(), key=lambda kv: kv[1], reverse=True)[:2]
        reasons.append((ev, [{"feature": k, "contribution": v} for k, v in top2]))
    reason_df = pd.DataFrame.from_records([(ev, r) for ev, r in reasons], columns=["event_id","top_reasons"]).set_index("event_id")

    out = pd.concat([feat_df, scaled_df.add_prefix("std_")], axis=1).join(reason_df, how="left")
    return out.sort_values("final_score", ascending=False)

# ---------------- IO ----------------

def load_or_aggregate(path: str, raw: bool, ts_col: str, event_col: str, count_col: str|int) -> pd.DataFrame:
    """
    Returns an hourly aggregated tall table with columns [ts, event_id, count].
    If raw=True and DuckDB is available, use DuckDB SQL for fast aggregation.
    """
    if not raw:
        # counts already aggregated
        if path.lower().endswith(".parquet") or path.lower().endswith(".pq"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        rename_map = {}
        for c in df.columns:
            lc = c.strip().lower()
            if lc in ("timestamp", "time", "date"): rename_map[c] = "ts"
            elif lc in ("event", "eventid", "id", "pattern"): rename_map[c] = "event_id"
            elif lc in ("value", "count", "n"): rename_map[c] = "count"
        df = df.rename(columns=rename_map)
        assert {"ts","event_id","count"}.issubset(df.columns), "Expected columns: ts, event_id, count"
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        if df["ts"].isna().any(): raise ValueError("Some timestamps could not be parsed.")
        df["ts"] = df["ts"].dt.floor("H")
        return df[["ts","event_id","count"]].copy()

    # raw logs, aggregate
    if HAVE_DUCKDB:
        con = duckdb.connect()
        # Let duckdb scan csv/parquet directly (supports globbing too)
        con.install_extension('httpfs', force=False)
        con.load_extension('httpfs')
        con.execute("SET threads TO {}".format(max(1, len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 4)))
        q = f"""
        SELECT date_trunc('hour', CAST({ts_col} AS TIMESTAMP)) AS ts,
               CAST({event_col} AS VARCHAR) AS event_id,
               SUM(CASE WHEN {count_col} IS NULL THEN 1 ELSE CAST({count_col} AS DOUBLE) END) AS count
        FROM read_auto('{path}')
        GROUP BY 1,2
        """
        df = con.execute(q).df()
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        df["count"] = df["count"].astype(np.float64)
        return df
    else:
        # Pandas fallback (slower): read in chunks and aggregate
        reader = (pd.read_parquet if path.lower().endswith((".parquet",".pq")) else pd.read_csv)
        chunks = []
        it = reader(path, chunksize=1_000_000) if reader is pd.read_csv else [reader(path)]
        for ch in it:
            ch["ts"] = pd.to_datetime(ch[ts_col], errors="coerce").dt.floor("H")
            ch["event_id"] = ch[event_col].astype(str)
            if isinstance(count_col, int):
                # count_col==1 means no provided count -> treat each row as 1
                ch["count"] = 1.0
            else:
                ch["count"] = pd.to_numeric(ch[count_col], errors="coerce").fillna(1.0)
            chunks.append(ch[["ts","event_id","count"]])
        big = pd.concat(chunks, ignore_index=True)
        agg = big.groupby(["ts","event_id"], sort=False, observed=True)["count"].sum().reset_index()
        return agg

def pivot_hourly(df: pd.DataFrame) -> pd.DataFrame:
    ts_min, ts_max = df["ts"].min(), df["ts"].max()
    piv = df.pivot_table(index="ts", columns="event_id", values="count", aggfunc="sum")
    full_idx = as_hourly_index(ts_min, ts_max)
    piv = piv.reindex(full_idx)
    # memory-friendly
    return piv.astype(np.float32)

# ---------------- CLI ----------------

import os

def main():
    ap = argparse.ArgumentParser(description="Anomaly Ranker Foundation - Statistical ensemble anomaly detection")
    ap.add_argument("--input", required=True, help="CSV/Parquet file path (raw or aggregated)")
    ap.add_argument("--raw", action="store_true", help="If set, aggregate raw logs into hourly counts")
    ap.add_argument("--ts-col", default="ts", help="Timestamp column if --raw")
    ap.add_argument("--event-col", default="event_id", help="Event/pattern column if --raw")
    ap.add_argument("--count-col", default="count", help="Count column if --raw (or integer 1 to count rows)")
    ap.add_argument("--incident-start", required=True)
    ap.add_argument("--incident-end", required=True)
    ap.add_argument("--out-csv", default="ranked_events.csv")
    ap.add_argument("--out-json", default="ranked_events.json")
    ap.add_argument("--workers", type=int, default=0, help="0/1 = no parallel; >1 enables parallel")
    ap.add_argument("--parallel", choices=["auto","process","thread","off"], default="auto")
    ap.add_argument("--hours-pre-baseline", type=int, default=48)
    ap.add_argument("--hours-var-window", type=int, default=24)
    ap.add_argument("--blackout-frac", type=float, default=0.8)
    args = ap.parse_args()

    # normalize count_col type
    try:
        args.count_col = int(args.count_col)
    except Exception:
        pass

    df = load_or_aggregate(args.input, args.raw, args.ts_col, args.event_col, args.count_col)
    piv = pivot_hourly(df)

    inc_start = parse_dt(args.incident_start)
    inc_end = parse_dt(args.incident_end)

    ranked = rank_events_parallel(
        piv, inc_start, inc_end,
        workers=args.workers,
        parallel=args.parallel,
        hours_pre_baseline=args.hours_pre_baseline,
        hours_var_window=args.hours_var_window,
        blackout_frac=args.blackout_frac
    )

    ranked.to_csv(args.out_csv, index=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(ranked.reset_index().to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    print("Top 20:")
    print(ranked[["final_score","inc_mean","base_mean","cov_incident","cov_pre"]].head(20).to_string())

if __name__ == "__main__":
    main()
