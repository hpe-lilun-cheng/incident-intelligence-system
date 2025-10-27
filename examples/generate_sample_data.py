#!/usr/bin/env python3
"""
Generate Sample Data
===================

Creates synthetic log data for testing the Incident Intelligence System.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data(output_path="data/sample_data.csv"):
    """Generate synthetic incident data with clear anomalies."""
    
    np.random.seed(42)
    
    # Time range: 30 days of hourly data
    start_time = datetime(2025, 8, 1, 0, 0, 0)
    hours = 30 * 24  # 30 days
    timestamps = [start_time + timedelta(hours=i) for i in range(hours)]
    
    # Event types with different baseline patterns
    events = {
        'DB_CONNECTION_TIMEOUT': {
            'baseline': lambda h: 5 + 3 * np.sin(h * np.pi / 12),  # Daily cycle
            'anomaly': 450,
            'hour': 500  # Hour 500 = Day 21, Hour 20
        },
        'API_RATE_LIMIT_EXCEEDED': {
            'baseline': lambda h: 20 + 10 * np.sin(h * np.pi / 12),
            'anomaly': 380,
            'hour': 500
        },
        'MEMORY_THRESHOLD_BREACH': {
            'baseline': lambda h: 3 if 8 <= (h % 24) <= 18 else 1,  # Business hours pattern
            'anomaly': 285,
            'hour': 501
        },
        'AUTH_SERVICE_ERROR': {
            'baseline': lambda h: 10 + 5 * np.random.randn(),
            'anomaly': 198,
            'hour': 501
        },
        'CACHE_MISS_SPIKE': {
            'baseline': lambda h: 50 + 10 * np.random.randn(),
            'anomaly': 145,
            'hour': 502
        },
        'SLOW_QUERY_WARNING': {
            'baseline': lambda h: 15 + 5 * np.sin(h * np.pi / 24),
            'anomaly': 0,  # Actually decreases during incident
            'hour': 501
        }
    }
    
    data = []
    
    for hour_idx, ts in enumerate(timestamps):
        for event_id, config in events.items():
            # Calculate baseline count
            base_count = config['baseline'](hour_idx)
            
            # Add noise
            count = max(0, base_count + np.random.randn() * 0.5)
            
            # Inject anomaly during incident
            if 500 <= hour_idx <= 503:  # 4-hour incident
                if hour_idx == config['hour']:
                    count = config['anomaly']
                elif event_id in ['DB_CONNECTION_TIMEOUT', 'API_RATE_LIMIT_EXCEEDED']:
                    # These stay elevated throughout incident
                    count = config['anomaly'] * 0.8
            
            data.append({
                'timestamp': ts.isoformat() + 'Z',
                'event_id': event_id,
                'count': int(round(count))
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Generated {len(df)} rows of sample data")
    print(f"   Saved to: {output_path}")
    print(f"   Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Events: {df['event_id'].nunique()}")
    print(f"   Incident window: 2025-08-21T20:00:00Z to 2025-08-21T23:00:00Z")
    
    # Show some stats
    print("\nSample Statistics:")
    incident_data = df[(df['timestamp'] >= '2025-08-21T20:00:00Z') & 
                       (df['timestamp'] <= '2025-08-21T23:00:00Z')]
    baseline_data = df[df['timestamp'] < '2025-08-21T20:00:00Z']
    
    for event in events.keys():
        inc_mean = incident_data[incident_data['event_id'] == event]['count'].mean()
        base_mean = baseline_data[baseline_data['event_id'] == event]['count'].mean()
        print(f"  {event}:")
        print(f"    Baseline avg: {base_mean:.1f}")
        print(f"    Incident avg: {inc_mean:.1f}")
        print(f"    Ratio: {inc_mean/base_mean:.1f}x")

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    output_file = data_dir / "sample_data.csv"
    generate_sample_data(str(output_file))
