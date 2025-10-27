#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anomaly Ranker Engine — Complete Incident Analysis Driver
=========================================================

This is the PRIMARY tool for incident analysis. It orchestrates a multi-layer
detection system combining statistical methods, ML detectors, and relational analysis.

Architecture:
  Layer 1: Statistical Ensemble (from anomaly_ranker_foundation)
    - 8 robust statistical features with weighted scoring
  
  Layer 2: ML Pattern Detectors
    - Pattern Break: Hour-of-day anomaly detection
    - Volume Spike: Rate-of-change detection  
    - Isolation Forest: Multi-dimensional outlier detection
  
  Layer 3: Relational Analysis
    - Correlation: Pearson, Spearman, cross-correlation with lags
    - Cascade Detection: Identifies A→B failure propagation
  
  Layer 4: Root Cause Synthesis
    - Hypothesis generation
    - Enhanced scoring with correlation/cascade boosts

Dependencies:
  - anomaly_ranker_foundation (statistical utilities)
  - sklearn (optional, for IsolationForest)
  - scipy (optional, for advanced correlations)

Usage:
  python anomaly_ranker_engine.py \
    --input hourly_counts.csv \
    --incident-start 2025-08-12T20:00:00Z \
    --incident-end   2025-08-12T23:00:00Z \
    --workers 8 \
    --out-csv enhanced_results.csv

Outputs:
  - enhanced_ranked_events.csv (comprehensive scores)
  - correlations.json (event relationships)
  - hypotheses.json (root cause synthesis)
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional deps with fallbacks
try:
    from sklearn.ensemble import IsolationForest
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

try:
    from scipy.signal import correlate
    from scipy.stats import pearsonr, spearmanr, rankdata as sp_rankdata
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# Import the *parallel core* ranker and utilities
from anomaly_ranker_foundation import (
    parse_dt,
    load_or_aggregate,
    pivot_hourly,
    rank_events_parallel,
    build_baselines,
    baseline_for,
    detect_global_blackouts,
    mad,
)

# ---------------- Utilities ----------------

def robust_scale_vec(x: np.ndarray) -> float:
    """IQR->std or MAD fallback; never <= 0."""
    x = x[~np.isnan(x)]
    if x.size == 0:
        return 1.0
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    if iqr > 0:
        return iqr / 1.349
    med = np.median(x)
    m = 1.4826 * np.median(np.abs(x - med))
    if m > 0:
        return m
    s = np.nanstd(x)
    return float(s if s > 0 else 1.0)

def zscore(a: np.ndarray) -> np.ndarray:
    m = np.nanmean(a)
    s = np.nanstd(a)
    s = s if s > 0 else 1.0
    return (a - m) / s

def series_step(index: pd.DatetimeIndex) -> pd.Timedelta:
    if len(index) < 2:
        return pd.Timedelta(hours=1)
    d = pd.Series(index).diff().median()
    return d if pd.notna(d) and d > pd.Timedelta(0) else pd.Timedelta(hours=1)

# ---------------- Data structures ----------------

@dataclass
class EnhancedAnomalyResult:
    event_id: str
    timestamp: pd.Timestamp
    anomaly_score: float
    anomaly_types: List[str]
    confidence: float
    baseline_deviation: float
    proximity_to_incident: float
    context: Dict[str, Any]

# ---------------- Parallel worker funcs ----------------

def process_pattern_break_for_event(args: Tuple) -> List[EnhancedAnomalyResult]:
    event_id, window_data, hourly_baselines, mid_incident, z_thr = args
    anomalies: List[EnhancedAnomalyResult] = []
    series = window_data[event_id].astype(float)

    for ts, val in series.items():
        if pd.isna(val):
            continue
        hod = int(ts.hour)
        base = hourly_baselines.get(event_id, {}).get(hod)
        if not base:
            continue
        mu = base.get('median', np.nan)
        sig = base.get('scale', 1.0)
        if np.isnan(mu) or sig <= 0:
            continue
        z = (val - mu) / sig
        if abs(z) >= z_thr:
            prox = abs((ts - mid_incident).total_seconds()) / 3600.0
            score = float(abs(z) + max(0.0, 2.0 - prox))
            anomalies.append(EnhancedAnomalyResult(
                event_id=str(event_id),
                timestamp=ts,
                anomaly_score=score,
                anomaly_types=['pattern_break'],
                confidence=float(min(0.95, abs(z) / 4.0)),
                baseline_deviation=float(abs(z)),
                proximity_to_incident=float(prox),
                context={'robust_z': float(z)}
            ))
    return anomalies

def process_volume_spike_for_event(args: Tuple) -> List[EnhancedAnomalyResult]:
    event_id, event_series, baseline_median, mid_incident = args
    anomalies: List[EnhancedAnomalyResult] = []
    prev = np.nan
    for ts, cur in event_series.sort_index().items():
        cur = float(cur) if cur is not None else np.nan
        if np.isnan(cur) or np.isnan(baseline_median):
            prev = cur
            continue
        denom = max(1.0, baseline_median)
        if not np.isnan(prev):
            denom = max(denom, prev)
        rate_change = (cur - (0.0 if np.isnan(prev) else prev)) / denom
        spike_mag = cur / max(1.0, baseline_median)
        spike_score = abs(rate_change) * 2.0 + max(0.0, spike_mag - 2.0)
        if spike_score <= 0:
            prev = cur
            continue
        spike_score = float(min(10.0, spike_score))
        prox = abs((ts - mid_incident).total_seconds()) / 3600.0
        score = float(spike_score + max(0.0, 2.0 - prox))
        conf = float(min(0.9, 0.5 + 0.5 * min(1.0, abs(rate_change))))
        anomalies.append(EnhancedAnomalyResult(
            event_id=str(event_id),
            timestamp=ts,
            anomaly_score=score,
            anomaly_types=['volume_spike'],
            confidence=conf,
            baseline_deviation=float(abs(spike_mag - 1.0)),
            proximity_to_incident=float(prox),
            context={'rate_change': float(rate_change), 'spike_magnitude': float(spike_mag)}
        ))
        prev = cur
    return anomalies

def process_isolation_forest_for_event(args: Tuple) -> List[EnhancedAnomalyResult]:
    if not HAVE_SKLEARN:
        return []
    event_id, window_data, mid_incident = args
    s = window_data[event_id].astype(float)
    if s.dropna().shape[0] < 32:
        return []
    X = pd.DataFrame({
        'count': s.values,
        'hour': window_data.index.hour,
        'dow': window_data.index.dayofweek,
        'is_weekend': (window_data.index.dayofweek >= 5).astype(int),
        'is_biz': ((window_data.index.hour >= 9) & (window_data.index.hour <= 18)).astype(int),
    }, index=window_data.index).astype(float)
    # Standardize features
    X = (X - X.mean()) / (X.std() + 1e-6)
    contam = float(np.clip(10.0 / max(len(X), 1), 0.01, 0.1))
    clf = IsolationForest(n_estimators=200, contamination=contam, random_state=42, n_jobs=1)
    mask_valid = ~X.isna().any(axis=1)
    if mask_valid.sum() < 32:
        return []
    pred = clf.fit_predict(X[mask_valid])
    scores = -clf.score_samples(X[mask_valid])  # higher => more anomalous
    # Map back
    out: List[EnhancedAnomalyResult] = []
    for ts, label, sscore in zip(X[mask_valid].index, pred, scores):
        if label == -1:
            prox = abs((ts - mid_incident).total_seconds()) / 3600.0
            conf = float(min(0.9, sscore / (np.nanmedian(scores) + 1e-6)))
            out.append(EnhancedAnomalyResult(
                event_id=str(event_id),
                timestamp=ts,
                anomaly_score=float(min(10.0, sscore + max(0.0, 2.0 - prox))),
                anomaly_types=['iforest_outlier'],
                confidence=conf,
                baseline_deviation=float(sscore),
                proximity_to_incident=float(prox),
                context={'iforest_score': float(sscore)}
            ))
    return out

def compute_event_pair_correlation(args: Tuple) -> Optional[Dict]:
    event1, event2, series1, series2, max_lag = args
    # Mask NaNs, z-score both
    mask = ~(pd.isna(series1) | pd.isna(series2))
    x = series1[mask].values.astype(float)
    y = series2[mask].values.astype(float)
    if x.size < 6:
        return None
    # z-score
    x = (x - x.mean()) / (x.std() + 1e-9)
    y = (y - y.mean()) / (y.std() + 1e-9)

    # Pearson / Spearman
    if HAVE_SCIPY:
        pr, _ = pearsonr(x, y)
        sr, _ = spearmanr(x, y)
    else:
        pr = float(np.corrcoef(x, y)[0, 1])
        # Spearman via ranking
        rx = pd.Series(x).rank().values
        ry = pd.Series(y).rank().values
        sr = float(np.corrcoef(rx, ry)[0, 1])

    # Cross-correlation lags ±max_lag
    if HAVE_SCIPY:
        cc = correlate(x, y, mode='full')
        center = len(cc) // 2
        start = center - max_lag
        end = center + max_lag + 1
        seg = cc[max(0, start):min(len(cc), end)]
        if seg.size == 0:
            xc = 0.0; lag = 0
        else:
            idx = int(np.argmax(np.abs(seg)))
            xc = float(seg[idx] / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))
            lag = int(idx - (max_lag if start >= 0 else center))
    else:
        # Simple fallback: choose lag by max Pearson on rolled variants
        lags = range(-max_lag, max_lag+1)
        vals = []
        for l in lags:
            if l < 0:
                xx, yy = x[:l], y[-l:]
            elif l > 0:
                xx, yy = x[l:], y[:-l]
            else:
                xx, yy = x, y
            if len(xx) >= 6:
                vals.append(float(np.corrcoef(xx, yy)[0, 1]))
            else:
                vals.append(0.0)
        idx = int(np.argmax(np.abs(vals)))
        xc = float(vals[idx])
        lag = list(lags)[idx]

    return {
        'event1': str(event1),
        'event2': str(event2),
        'pearson': float(pr),
        'spearman': float(sr),
        'cross_correlation': {'max_correlation': float(xc), 'optimal_lag_hours': int(lag)}
    }

def process_cascade_batch(args: Tuple) -> List[Tuple[str, str]]:
    batch, time_threshold = args
    pairs: List[Tuple[str, str]] = []
    # For each adjacent anomaly in batch, add A->B if within threshold
    for i in range(len(batch) - 1):
        a = batch[i]; b = batch[i+1]
        if b.timestamp - a.timestamp <= time_threshold and a.event_id != b.event_id:
            pairs.append((a.event_id, b.event_id))
    return pairs

# ---------------- Main analyzer class ----------------

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

class FullyParallelEnhancedAnalyzer:
    def __init__(self, workers: int = 0, parallel: str = "auto", detection_window_hours: int = 3, z_threshold: float = 2.5):
        self.workers = int(max(0, workers))
        self.parallel = parallel
        self.detection_window_hours = detection_window_hours
        self.z_threshold = z_threshold

    # --------- Hourly baselines (exclude incident) ---------

    def build_hourly_baselines(self, piv: pd.DataFrame, incident_start: Optional[pd.Timestamp] = None, incident_end: Optional[pd.Timestamp] = None) -> Dict[str, Dict[int, Dict[str, float]]]:
        """Per-event hour-of-day medians and robust scales; exclude incident span to avoid contamination."""
        if incident_start is not None and incident_end is not None:
            piv = piv[(piv.index < incident_start) | (piv.index > incident_end)]
        hourly_baselines: Dict[str, Dict[int, Dict[str, float]]] = {}
        for eid in piv.columns:
            ser = piv[eid].astype(float)
            if ser.dropna().shape[0] < 24:
                continue
            pat: Dict[int, Dict[str, float]] = {}
            for h in range(24):
                vals = ser[ser.index.hour == h].values
                vals = vals[~np.isnan(vals)]
                if vals.size >= 3:
                    med = float(np.median(vals))
                    sc = float(robust_scale_vec(vals))
                    pat[h] = {'median': med, 'scale': max(1e-6, sc)}
            if pat:
                hourly_baselines[eid] = pat
        return hourly_baselines

    # --------- Pattern breaks (parallel) ---------

    def detect_pattern_breaks_parallel(self, piv: pd.DataFrame, hourly_baselines: Dict[str, Dict[int, Dict[str, float]]], incident_start: pd.Timestamp, incident_end: pd.Timestamp) -> List[EnhancedAnomalyResult]:
        mid = incident_start + (incident_end - incident_start) / 2
        start = mid - timedelta(hours=self.detection_window_hours)
        end = mid + timedelta(hours=self.detection_window_hours)
        mask = (piv.index >= start) & (piv.index <= end)
        if not mask.any():
            return []
        window = piv.loc[mask]
        tasks = [(eid, window, hourly_baselines, mid, self.z_threshold) for eid in window.columns]
        return self._execute_parallel(process_pattern_break_for_event, tasks)

    # --------- Volume spikes (parallel) ---------

    def detect_volume_spikes_parallel(self, piv: pd.DataFrame, hourly_baselines: Dict[str, Dict[int, Dict[str, float]]], incident_start: pd.Timestamp, incident_end: pd.Timestamp) -> List[EnhancedAnomalyResult]:
        mid = incident_start + (incident_end - incident_start) / 2
        start = mid - timedelta(hours=self.detection_window_hours)
        end = mid + timedelta(hours=self.detection_window_hours)
        mask = (piv.index >= start) & (piv.index <= end)
        if not mask.any():
            return []
        window = piv.loc[mask]
        tasks = []
        for eid in window.columns:
            base = hourly_baselines.get(eid, {})
            # Use median across hours as a fallback
            bmed = np.nanmedian([v.get('median', np.nan) for v in base.values()]) if base else np.nanmedian(window[eid].values)
            tasks.append((eid, window[eid], float(bmed if not np.isnan(bmed) else 1.0), mid))
        return self._execute_parallel(process_volume_spike_for_event, tasks)

    # --------- Isolation Forest (parallel) ---------

    def detect_isolation_forest_parallel(self, piv: pd.DataFrame, incident_start: pd.Timestamp, incident_end: pd.Timestamp) -> List[EnhancedAnomalyResult]:
        if not HAVE_SKLEARN:
            return []
        mid = incident_start + (incident_end - incident_start) / 2
        start = mid - timedelta(hours=self.detection_window_hours)
        end = mid + timedelta(hours=self.detection_window_hours)
        mask = (piv.index >= start) & (piv.index <= end)
        if not mask.any():
            return []
        window = piv.loc[mask]
        tasks = [(eid, window, mid) for eid in window.columns]
        return self._execute_parallel(process_isolation_forest_for_event, tasks)

    # --------- Correlation & cross-correlation (parallel, top-K gating) ---------

    def analyze_pattern_correlations_parallel(self, piv: pd.DataFrame, core_rank: pd.DataFrame, incident_start: pd.Timestamp, incident_end: pd.Timestamp, window_hours: int = 6, top_k: int = 400) -> Dict[str, Any]:
        mid = incident_start + (incident_end - incident_start) / 2
        start = mid - timedelta(hours=window_hours)
        end = mid + timedelta(hours=window_hours)
        window = piv[(piv.index >= start) & (piv.index <= end)]
        if window.shape[1] < 2:
            return {'error': 'Insufficient event types for correlation analysis'}

        # Top-K gating by final_score to control O(E^2)
        events = core_rank.sort_values("final_score", ascending=False).head(top_k).index.tolist()
        events = [e for e in events if e in window.columns]
        window = window[events]

        tasks: List[Tuple] = []
        # Lag cap based on cadence: up to 5 steps (~5h if hourly)
        step = series_step(window.index)
        max_lag = min(5, int(round(timedelta(hours=5) / step))) if step > pd.Timedelta(0) else 5

        for i in range(len(events)):
            for j in range(i+1, len(events)):
                tasks.append((events[i], events[j], window[events[i]], window[events[j]], max_lag))

        pairs = self._execute_parallel(compute_event_pair_correlation, tasks)
        pairs = [p for p in pairs if p is not None]

        strong = sorted(pairs, key=lambda d: max(abs(d['pearson']), abs(d['spearman']), abs(d['cross_correlation']['max_correlation'])), reverse=True)[: max(50, len(pairs)//10 or 0)]
        return {'pairs_analyzed': len(tasks), 'strong_correlations': strong}

    # --------- Cascade detection (parallel, cadence-aware) ---------

    def detect_cascade_patterns_parallel(self, all_anomalies: List[EnhancedAnomalyResult], piv_index: pd.DatetimeIndex, incident_start: pd.Timestamp, incident_end: pd.Timestamp) -> Dict[str, Any]:
        if not all_anomalies:
            return {'error': 'No anomalies to analyze'}
        sorted_anoms = sorted(all_anomalies, key=lambda a: a.timestamp)
        # Cadence-aware time threshold
        step = series_step(piv_index)
        time_threshold = max(pd.Timedelta(minutes=30), 2 * step)

        # Batch the adjacency checks
        batch_size = max(10, len(sorted_anoms) // max(1, self.workers)) if self.workers > 1 else len(sorted_anoms)
        batches = [sorted_anoms[i:i+batch_size] for i in range(0, len(sorted_anoms), batch_size)]
        pairs = self._execute_parallel(process_cascade_batch, [(b, time_threshold) for b in batches])

        # Flatten & count
        flat: List[Tuple[str, str]] = []
        for sub in pairs:
            flat.extend(sub)
        if not flat:
            return {'common_cascades': []}

        counts: Dict[Tuple[str, str], int] = {}
        for a, b in flat:
            counts[(a, b)] = counts.get((a, b), 0) + 1
        common = [{'source_event': a, 'target_event': b, 'frequency': c} for (a, b), c in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)]
        return {'common_cascades': common[:50]}

    # --------- Hypotheses synthesis ---------

    def generate_root_cause_hypotheses(self, anomalies: List[EnhancedAnomalyResult], correlations: Dict[str, Any], cascades: Dict[str, Any], incident_start: pd.Timestamp) -> Dict[str, Any]:
        hyp: Dict[str, Any] = {'primary_triggers': [], 'cascade_failures': [], 'correlated_failures': []}
        if anomalies:
            top = sorted(anomalies, key=lambda a: a.anomaly_score, reverse=True)[:5]
            hyp['primary_triggers'] = [{'event_id': a.event_id, 'timestamp': str(a.timestamp), 'reason': a.anomaly_types, 'score': a.anomaly_score} for a in top]
        for c in cascades.get('common_cascades', [])[:5]:
            hyp['cascade_failures'].append({'source': c['source_event'], 'target': c['target_event'], 'frequency': c['frequency']})
        for p in correlations.get('strong_correlations', [])[:5]:
            hyp['correlated_failures'].append({'event1': p['event1'], 'event2': p['event2'], 'corr': max(abs(p['pearson']), abs(p['spearman']), abs(p['cross_correlation']['max_correlation'])), 'lag_hours': p['cross_correlation']['optimal_lag_hours']})
        return hyp

    # --------- Score merge & boosts ---------

    def apply_boosts_and_merge(self, core_rank: pd.DataFrame, pattern_anoms: List[EnhancedAnomalyResult], volume_anoms: List[EnhancedAnomalyResult], iforest_anoms: List[EnhancedAnomalyResult], correlations: Dict[str, Any], cascades: Dict[str, Any]) -> pd.DataFrame:
        df = core_rank.copy()

        def agg(anoms: List[EnhancedAnomalyResult]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int]]:
            best: Dict[str, float] = {}; conf: Dict[str, float] = {}; ts_count: Dict[str, set] = {}
            for a in anoms:
                eid = a.event_id
                val = a.anomaly_score * max(0.1, a.confidence)
                if (eid not in best) or (val > best[eid]):
                    best[eid] = val; conf[eid] = a.confidence
                ts_count.setdefault(eid, set()).add(a.timestamp)
            return best, conf, {k: len(v) for k, v in ts_count.items()}

        patt_best, patt_conf, patt_ts = agg(pattern_anoms)
        vol_best, vol_conf, vol_ts = agg(volume_anoms)
        if_best, if_conf, if_ts = agg(iforest_anoms)

        df['pattern_break_score'] = df.index.map(lambda e: patt_best.get(e, 0.0)).astype(float)
        df['volume_spike_score'] = df.index.map(lambda e: vol_best.get(e, 0.0)).astype(float)
        df['iforest_score']      = df.index.map(lambda e: if_best.get(e, 0.0)).astype(float)

        # Correlation boost: strong pairs; leaders (negative lag) get a slightly higher bump
        corr_boost = pd.Series(0.0, index=df.index)
        for p in correlations.get('strong_correlations', [])[:50]:
            e1, e2 = p['event1'], p['event2']
            strength = float(max(abs(p['pearson']), abs(p['spearman']), abs(p['cross_correlation']['max_correlation'])))
            lag = int(p['cross_correlation'].get('optimal_lag_hours', 0))
            lead_bonus, lag_bonus = 0.15, 0.05
            if e1 in corr_boost.index:
                corr_boost.loc[e1] += strength * (lead_bonus if lag < 0 else (lag_bonus if lag > 0 else 0.10))
            if e2 in corr_boost.index:
                corr_boost.loc[e2] += strength * (lead_bonus if lag > 0 else (lag_bonus if lag < 0 else 0.10))

        # Cascade boost: sources get more than targets
        cas_boost = pd.Series(0.0, index=df.index)
        for c in cascades.get('common_cascades', [])[:50]:
            src, tgt, freq = c['source_event'], c['target_event'], float(c.get('frequency', 1))
            if src in cas_boost.index: cas_boost.loc[src] += 0.10 * math.log1p(freq)
            if tgt in cas_boost.index: cas_boost.loc[tgt] += 0.06 * math.log1p(freq)

        # Multi-detector presence bonus (capped)
        methods = ((df['pattern_break_score'] > 0).astype(int) + (df['volume_spike_score'] > 0).astype(int) + (df['iforest_score'] > 0).astype(int))
        uniq_ts_bonus = df.index.map(lambda e: math.log1p(patt_ts.get(e, 0) + vol_ts.get(e, 0) + if_ts.get(e, 0)))
        df['multi_bonus'] = 0.10 * methods + 0.02 * uniq_ts_bonus + corr_boost + cas_boost

        # Enhanced score (keep core as primary driver)
        df['enhanced_score'] = (
            0.60 * df['final_score'].fillna(0.0)
            + 0.20 * df['pattern_break_score']
            + 0.12 * df['volume_spike_score']
            + 0.08 * df['iforest_score']
            + df['multi_bonus']
        )

        # Clean up infinities
        for c in ['pattern_break_score', 'volume_spike_score', 'iforest_score', 'multi_bonus', 'enhanced_score']:
            df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return df.sort_values('enhanced_score', ascending=False)

    # --------- Parallel execution wrapper ---------

    def _execute_parallel(self, fn, tasks: List[Tuple]):
        if not tasks:
            return []
        if self.workers and self.workers > 1:
            engine = self.parallel if self.parallel in ('process', 'thread') else 'process'
            Exec = ProcessPoolExecutor if engine == 'process' else ThreadPoolExecutor
            out: List = []
            with Exec(max_workers=self.workers) as ex:
                futs = [ex.submit(fn, t) for t in tasks]
                for f in as_completed(futs):
                    res = f.result()
                    if isinstance(res, list):
                        out.extend(res)
                    elif res is not None:
                        out.append(res)
            return out
        else:
            out: List = []
            for t in tasks:
                res = fn(t)
                if isinstance(res, list):
                    out.extend(res)
                elif res is not None:
                    out.append(res)
            return out

# ---------------- Orchestration ----------------

def run_fully_parallel_enhanced(
    input_path: str,
    incident_start: str,
    incident_end: str,
    raw: bool = False,
    ts_col: str = "ts",
    event_col: str = "event_id",
    count_col: str | int = "count",
    workers: int = 0,
    parallel: str = "auto",
    detection_window_hours: int = 3,
    z_threshold: float = 2.5,
    blackout_frac: float = 0.8,
    corr_top_k: int = 400,
    out_csv: Optional[str] = None,
    out_json: Optional[str] = None,
    out_corr: Optional[str] = None,
    out_hypo: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:

    # Normalize count_col
    try:
        count_col_cast: str | int = int(count_col)
    except Exception:
        count_col_cast = count_col

    # Load / aggregate and pivot
    df = load_or_aggregate(input_path, raw, ts_col, event_col, count_col_cast)
    piv = pivot_hourly(df)

    # Incident times
    inc_start = parse_dt(incident_start)
    inc_end = parse_dt(incident_end)

    # Global blackouts -> NaN (avoid false drops)
    blackout_hours = detect_global_blackouts(piv, frac_zero_thresh=blackout_frac)
    if len(blackout_hours) > 0:
        piv = piv.copy()
        piv.loc[blackout_hours] = np.nan

    # Core ranking
    core = rank_events_parallel(
        piv,
        inc_start,
        inc_end,
        workers=workers,
        parallel=parallel,
        blackout_frac=blackout_frac,
    )

    analyzer = FullyParallelEnhancedAnalyzer(workers=workers, parallel=parallel, detection_window_hours=detection_window_hours, z_threshold=z_threshold)

    # Hourly baselines (exclude incident span)
    hourly_base = analyzer.build_hourly_baselines(piv, inc_start, inc_end)

    # Detectors
    patt = analyzer.detect_pattern_breaks_parallel(piv, hourly_base, inc_start, inc_end)
    vol  = analyzer.detect_volume_spikes_parallel(piv, hourly_base, inc_start, inc_end)
    ifor = analyzer.detect_isolation_forest_parallel(piv, inc_start, inc_end)

    # Correlations (top-K gating) + Cascades
    corr = analyzer.analyze_pattern_correlations_parallel(piv, core, inc_start, inc_end, window_hours=6, top_k=corr_top_k)
    casc = analyzer.detect_cascade_patterns_parallel(patt + vol + ifor, piv.index, inc_start, inc_end)

    # Hypotheses
    hypo = analyzer.generate_root_cause_hypotheses(patt + vol + ifor, corr, casc, inc_start)

    # Merge
    enhanced = analyzer.apply_boosts_and_merge(core, patt, vol, ifor, corr, casc)

    # Write outputs
    if out_csv:
        enhanced.to_csv(out_csv, index=True)
    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(enhanced.reset_index().to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    if out_corr:
        with open(out_corr, "w", encoding="utf-8") as f:
            json.dump(corr, f, ensure_ascii=False, indent=2)
    if out_hypo:
        with open(out_hypo, "w", encoding="utf-8") as f:
            json.dump(hypo, f, ensure_ascii=False, indent=2)

    return enhanced, corr, hypo

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="Anomaly Ranker Engine - Complete incident analysis with ML and correlation")
    ap.add_argument("--input", required=True, help="CSV/Parquet path (raw or hourly counts)")
    ap.add_argument("--raw", action="store_true", help="If set, aggregate raw logs into hourly counts")
    ap.add_argument("--ts-col", default="ts")
    ap.add_argument("--event-col", default="event_id")
    ap.add_argument("--count-col", default="count")
    ap.add_argument("--incident-start", required=True)
    ap.add_argument("--incident-end", required=True)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--parallel", choices=["auto","process","thread","off"], default="auto")
    ap.add_argument("--detection-window-hours", type=int, default=3)
    ap.add_argument("--z-threshold", type=float, default=2.5)
    ap.add_argument("--blackout-frac", type=float, default=0.8)
    ap.add_argument("--corr-top-k", type=int, default=400)
    ap.add_argument("--out-csv", default="enhanced_ranked_events.csv")
    ap.add_argument("--out-json", default="enhanced_ranked_events.json")
    ap.add_argument("--out-corr", default="correlations.json")
    ap.add_argument("--out-hypo", default="hypotheses.json")
    args = ap.parse_args()

    enhanced, corr, hypo = run_fully_parallel_enhanced(
        input_path=args.input,
        incident_start=args.incident_start,
        incident_end=args.incident_end,
        raw=args.raw,
        ts_col=args.ts_col,
        event_col=args.event_col,
        count_col=args.count_col,
        workers=args.workers,
        parallel=args.parallel,
        detection_window_hours=args.detection_window_hours,
        z_threshold=args.z_threshold,
        blackout_frac=args.blackout_frac,
        corr_top_k=args.corr_top_k,
        out_csv=args.out_csv,
        out_json=args.out_json,
        out_corr=args.out_corr,
        out_hypo=args.out_hypo,
    )

    print("Top 15 (enhanced):")
    cols = ["enhanced_score","final_score","pattern_break_score","volume_spike_score","iforest_score","multi_bonus"]
    print(enhanced[cols].head(15).to_string())

if __name__ == "__main__":
    main()
