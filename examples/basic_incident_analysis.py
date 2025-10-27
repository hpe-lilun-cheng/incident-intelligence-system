#!/usr/bin/env python3
"""
Basic Incident Analysis Example
================================

This example demonstrates how to use the Anomaly Ranker Engine to analyze
an incident and identify root causes.
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from anomaly_ranker_foundation import load_or_aggregate, pivot_hourly, parse_dt
from anomaly_ranker_engine import FullyParallelEnhancedAnalyzer

def main():
    print("=" * 60)
    print("Basic Incident Analysis Example")
    print("=" * 60)
    
    # Step 1: Load and prepare data
    print("\n[Step 1] Loading data...")
    input_file = Path(__file__).parent.parent / "data" / "sample_data.csv"
    
    if not input_file.exists():
        print(f"Error: Sample data not found at {input_file}")
        print("Please create sample_data.csv with columns: timestamp, event_id, count")
        return
    
    # Load data (already aggregated to hourly)
    df = load_or_aggregate(str(input_file), raw=False, ts_col="ts", 
                           event_col="event_id", count_col="count")
    print(f"Loaded {len(df)} rows with {df['event_id'].nunique()} unique events")
    
    # Create pivot table
    print("\n[Step 2] Creating pivot table...")
    piv = pivot_hourly(df)
    print(f"Pivot shape: {piv.shape[0]} hours × {piv.shape[1]} events")
    
    # Define incident window
    incident_start = parse_dt("2025-08-12T20:00:00Z")
    incident_end = parse_dt("2025-08-12T23:00:00Z")
    print(f"\n[Step 3] Analyzing incident from {incident_start} to {incident_end}")
    
    # Run analysis
    print("\n[Step 4] Running anomaly detection...")
    analyzer = FullyParallelEnhancedAnalyzer(
        workers=4,
        parallel="auto",
        detection_window_hours=3,
        z_threshold=2.5
    )
    
    results = analyzer.analyze(
        piv=piv,
        incident_start=incident_start,
        incident_end=incident_end
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("TOP 10 ANOMALOUS EVENTS")
    print("=" * 60)
    
    top_events = results['enhanced_ranked_events'].head(10)
    print("\nEvent Rankings:")
    for idx, (event_id, row) in enumerate(top_events.iterrows(), 1):
        print(f"\n{idx}. {event_id}")
        print(f"   Enhanced Score: {row['enhanced_score']:.2f}")
        print(f"   Final Score (statistical): {row['final_score']:.2f}")
        print(f"   Pattern Break: {row['pattern_break_score']:.2f}")
        print(f"   Volume Spike: {row['volume_spike_score']:.2f}")
        print(f"   IForest Score: {row['iforest_score']:.2f}")
    
    # Display correlations
    if results['correlations']:
        print("\n" + "=" * 60)
        print("STRONG CORRELATIONS")
        print("=" * 60)
        
        for corr in results['correlations'][:5]:
            print(f"\n{corr['event1']} ↔ {corr['event2']}")
            print(f"   Pearson: {corr['pearson']:.3f}")
            print(f"   Optimal Lag: {corr['cross_correlation']['optimal_lag_hours']} hours")
    
    # Display cascades
    if results['cascades']:
        print("\n" + "=" * 60)
        print("CASCADE PATTERNS")
        print("=" * 60)
        
        for cascade in results['cascades'][:5]:
            print(f"\n{cascade['source_event']} → {cascade['target_event']}")
            print(f"   Frequency: {cascade['frequency']} occurrences")
    
    # Display hypotheses
    if results['hypotheses']:
        print("\n" + "=" * 60)
        print("ROOT CAUSE HYPOTHESES")
        print("=" * 60)
        
        for hyp in results['hypotheses'][:3]:
            print(f"\n• {hyp['description']}")
            print(f"  Confidence: {hyp['confidence']:.2f}")
            print(f"  Type: {hyp['type']}")
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
