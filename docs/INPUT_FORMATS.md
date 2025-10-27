# Input Format Specifications

## Table of Contents
1. [Overview](#overview)
2. [Anomaly Ranker Input](#anomaly-ranker-input)
3. [Predictive Pattern Miner Input](#predictive-pattern-miner-input)
4. [Data Preparation](#data-preparation)
5. [Examples](#examples)

---

## Overview

Both engines work with **standardized time series data**, but have slightly different input requirements based on their use cases.

### Universal Format
```
Core Triple: (ID, Timestamp, Value)
- ID: Unique identifier for the metric/event/pattern
- Timestamp: When the observation occurred
- Value: Numeric measurement (count, sum, average, etc.)
```

---

## Anomaly Ranker Input

### Required Format

The Anomaly Ranker accepts two input modes:

#### Mode 1: Pre-Aggregated Counts (Recommended)

**File Format**: CSV or Parquet

**Required Columns**:
```csv
timestamp,event_id,count
2025-08-12T00:00:00Z,DB_CONNECTION_TIMEOUT,5
2025-08-12T00:00:00Z,API_RATE_LIMIT_EXCEEDED,12
2025-08-12T01:00:00Z,DB_CONNECTION_TIMEOUT,3
2025-08-12T01:00:00Z,API_RATE_LIMIT_EXCEEDED,8
```

**Column Specifications**:
- `timestamp`: ISO 8601 format, hourly granularity preferred
- `event_id`: String identifier (log template ID, metric name, error code, etc.)
- `count`: Numeric value (can be count, sum, average, max, etc.)

**Alternative Column Names** (auto-detected):
- Timestamp: `timestamp`, `time`, `ts`, `date`
- Event ID: `event_id`, `eventid`, `event`, `id`, `pattern`
- Count: `count`, `value`, `n`

#### Mode 2: Raw Logs (On-the-Fly Aggregation)

**File Format**: CSV or Parquet

**Required Columns**:
```csv
ts,event_id,additional_fields...
2025-08-12T00:15:23Z,DB_CONNECTION_TIMEOUT,...
2025-08-12T00:18:47Z,API_RATE_LIMIT_EXCEEDED,...
2025-08-12T00:22:11Z,DB_CONNECTION_TIMEOUT,...
```

**Usage**:
```bash
python src/anomaly_ranker_engine.py \
    --input raw_logs.csv \
    --raw \                          # Enable raw mode
    --ts-col ts \                   # Timestamp column name
    --event-col event_id \          # Event ID column name
    --count-col 1 \                 # 1 = count each row as 1 occurrence
    --incident-start "..." \
    --incident-end "..."
```

### Command-Line Parameters

**Required**:
- `--input`: Path to input file (CSV or Parquet)
- `--incident-start`: Incident start time (ISO 8601)
- `--incident-end`: Incident end time (ISO 8601)

**Optional**:
- `--workers`: Number of parallel workers (default: 0 = single-threaded)
- `--parallel`: Parallelization mode (default: "auto")
  - `auto`: Use multiprocessing if workers > 1
  - `process`: Force multiprocessing
  - `thread`: Use threading
  - `off`: Disable parallelization
- `--hours-pre-baseline`: Hours before incident for baseline (default: 48)
- `--hours-var-window`: Window for variance calculation (default: 24)
- `--blackout-frac`: Threshold for global blackout detection (default: 0.8)
- `--out-csv`: Output CSV path (default: "enhanced_ranked_events.csv")
- `--out-json`: Output JSON path (default: "enhanced_ranked_events.json")

### Example Input File

**hourly_counts.csv**:
```csv
timestamp,event_id,count
2025-08-12T00:00:00Z,DB_CONNECTION_TIMEOUT,5
2025-08-12T00:00:00Z,API_RATE_LIMIT_EXCEEDED,12
2025-08-12T00:00:00Z,MEMORY_THRESHOLD_BREACH,3
2025-08-12T00:00:00Z,AUTH_SERVICE_ERROR,8
2025-08-12T00:00:00Z,CACHE_MISS_SPIKE,45
2025-08-12T01:00:00Z,DB_CONNECTION_TIMEOUT,3
2025-08-12T01:00:00Z,API_RATE_LIMIT_EXCEEDED,8
2025-08-12T01:00:00Z,MEMORY_THRESHOLD_BREACH,2
...
2025-08-12T20:00:00Z,DB_CONNECTION_TIMEOUT,450  # Incident starts
2025-08-12T20:00:00Z,API_RATE_LIMIT_EXCEEDED,380
2025-08-12T21:00:00Z,DB_CONNECTION_TIMEOUT,520
...
```

### Output Format

**enhanced_ranked_events.csv**:
```csv
event_id,enhanced_score,final_score,pattern_break_score,volume_spike_score,iforest_score,z_max,glr,pmi_overrep,cohen_d,inc_mean,base_mean,cov_incident
DB_CONNECTION_TIMEOUT,8.45,7.12,3.21,2.85,1.54,4.32,125.3,2.15,1.87,450.2,12.5,1.0
API_RATE_LIMIT_EXCEEDED,7.92,6.85,2.95,3.12,1.12,3.98,98.7,1.92,1.65,380.7,25.3,1.0
MEMORY_THRESHOLD_BREACH,6.54,5.43,2.43,2.01,0.98,3.21,67.4,1.54,1.32,285.3,45.2,0.95
```

**correlations.json**:
```json
{
  "pairs_analyzed": 780,
  "strong_correlations": [
    {
      "event1": "DB_CONNECTION_TIMEOUT",
      "event2": "API_RATE_LIMIT_EXCEEDED",
      "pearson": 0.82,
      "spearman": 0.78,
      "cross_correlation": {
        "max_correlation": 0.85,
        "optimal_lag_hours": -2
      }
    }
  ]
}
```

**hypotheses.json**:
```json
{
  "primary_triggers": [
    {
      "event_id": "DB_CONNECTION_TIMEOUT",
      "timestamp": "2025-08-12T20:15:00",
      "reason": ["pattern_break", "volume_spike"],
      "score": 8.45
    }
  ],
  "cascade_failures": [
    {
      "source": "DB_CONNECTION_TIMEOUT",
      "target": "API_RATE_LIMIT_EXCEEDED",
      "frequency": 12
    }
  ]
}
```

---

## Predictive Pattern Miner Input

### Required Format

**File Format**: CSV or Parquet

**Required Columns**:
```csv
timestamp,log_template_id,sla_breach_start,sla_breach_end,from_time,end_time
2025-08-01T00:15:23Z,T501,2025-08-01T00:30:00Z,2025-08-01T00:45:00Z,2025-08-01T00:00:00Z,2025-08-01T01:00:00Z
2025-08-01T00:18:47Z,T502,2025-08-01T00:30:00Z,2025-08-01T00:45:00Z,2025-08-01T00:00:00Z,2025-08-01T01:00:00Z
```

**Column Specifications**:
- `timestamp`: When the log event occurred (any timestamp format)
- `log_template_id`: Template ID from log parser (e.g., Drain)
- `sla_breach_start`: When SLA breach started
- `sla_breach_end`: When SLA breach ended
- `from_time`: Window start time (for grouping)
- `end_time`: Window end time (for grouping)

**Optional Columns** (for clarity analysis):
- `service_name`: Which service produced the log
- `host`: Which host produced the log
- `severity`: Log severity level
- Any other categorical context

### Configuration Options

**MinerConfig Parameters**:

```python
from src.predictive_pattern_miner import MinerConfig

config = MinerConfig(
    # Column names
    time_col="timestamp",
    template_col="log_template_id",
    sla_start_col="sla_breach_start",
    sla_end_col="sla_breach_end",
    window_start_col="from_time",
    window_end_col="end_time",
    context_cols=["service_name", "host"],  # Optional
    
    # Timing
    freq="1min",                    # Time bucket size
    lead_lag_horizon="10min",       # Max lead time to check
    
    # Scoring
    z_weight=0.35,                  # Statistical significance weight
    beta_lead=0.3,                  # Lead correlation boost
    clarity_power=1.0,              # Clarity exponent
    max_lift=10.0,                  # Lift cap
    
    # Filters
    min_support_incident=5,         # Min occurrences in incidents
    min_template_occurrences=10,    # Min total occurrences
    alpha_fdr=0.05,                 # FDR significance level
    
    # Performance
    n_jobs=8,                       # Parallel workers
    top_k_leadlag=2000,             # Compute lead/lag for top-K only
    chunk_size=1000000,             # Chunk size for large files
    show_progress=True              # Show progress bars
)
```

### Example Input File

**logs_with_sla_windows.csv**:
```csv
timestamp,log_template_id,sla_breach_start,sla_breach_end,from_time,end_time,service_name,host
2025-08-01T00:15:23Z,T501,2025-08-01T00:30:00Z,2025-08-01T00:45:00Z,2025-08-01T00:00:00Z,2025-08-01T01:00:00Z,api-gateway,host-1
2025-08-01T00:18:47Z,T502,2025-08-01T00:30:00Z,2025-08-01T00:45:00Z,2025-08-01T00:00:00Z,2025-08-01T01:00:00Z,database,host-2
2025-08-01T00:22:11Z,T501,2025-08-01T00:30:00Z,2025-08-01T00:45:00Z,2025-08-01T00:00:00Z,2025-08-01T01:00:00Z,api-gateway,host-1
2025-08-01T00:25:33Z,T503,2025-08-01T00:30:00Z,2025-08-01T00:45:00Z,2025-08-01T00:00:00Z,2025-08-01T01:00:00Z,cache,host-3
...
# Baseline window (no SLA breach)
2025-08-02T00:15:23Z,T501,,,2025-08-02T00:00:00Z,2025-08-02T01:00:00Z,api-gateway,host-1
2025-08-02T00:18:47Z,T502,,,2025-08-02T00:00:00Z,2025-08-02T01:00:00Z,database,host-2
```

**Notes**:
- SLA breach columns can be empty for baseline windows
- Each row is a log event
- Windows define analysis boundaries (typically 1 hour)
- Can have multiple SLA breaches per file

### Output Format

**ranked_predictive_patterns.csv**:
```csv
log_template_id,score,z,lift,clarity,lead_corr_pos,lead_lag_seconds,c_i,c_b,N_i,N_b,p_value,q_value,significant_fdr,first_seen,last_seen
T501,15.3,4.2,8.5,0.82,0.65,-180,45,8,50,200,0.001,0.015,True,2025-08-01T00:15:23Z,2025-08-15T23:45:12Z
T502,12.7,3.8,6.2,0.75,0.58,-240,38,10,50,200,0.002,0.018,True,2025-08-01T00:18:47Z,2025-08-15T23:52:33Z
T503,10.1,3.2,4.8,0.68,0.45,-120,32,12,50,200,0.005,0.025,True,2025-08-01T00:25:33Z,2025-08-15T23:58:41Z
```

**Column Descriptions**:
- `score`: Composite predictive score
- `z`: Log-odds z-score (statistical significance)
- `lift`: Incident rate / baseline rate
- `clarity`: Entropy-based pattern quality (0-1)
- `lead_corr_pos`: Positive lead correlation
- `lead_lag_seconds`: Lead time (negative = template appears before breach)
- `c_i`: Count in incident windows
- `c_b`: Count in baseline windows
- `N_i`: Total incident windows
- `N_b`: Total baseline windows
- `p_value`: Fisher exact test p-value
- `q_value`: FDR-corrected q-value
- `significant_fdr`: Boolean, passes FDR threshold
- `first_seen`: First occurrence timestamp
- `last_seen`: Last occurrence timestamp

---

## Data Preparation

### From Raw Logs to Ranker Input

**Step 1: Parse logs with Drain** (or any log parser)
```python
# Use Drain3 or similar to extract templates
from drain3 import TemplateMiner

miner = TemplateMiner()
for log_line in raw_logs:
    result = miner.add_log_message(log_line)
    template_id = result["cluster_id"]
```

**Step 2: Aggregate to hourly counts**
```python
import pandas as pd

logs = pd.read_csv("parsed_logs.csv")
logs['timestamp'] = pd.to_datetime(logs['timestamp'])
logs['hour'] = logs['timestamp'].dt.floor('H')

hourly_counts = logs.groupby(['hour', 'template_id']).size().reset_index(name='count')
hourly_counts.rename(columns={'hour': 'timestamp', 'template_id': 'event_id'}, inplace=True)
hourly_counts.to_csv("hourly_counts.csv", index=False)
```

### From Raw Logs to Miner Input

**Step 1: Identify SLA breach windows**
```python
# From monitoring system or manual annotation
sla_breaches = pd.DataFrame({
    'breach_id': [1, 2, 3],
    'sla_breach_start': ['2025-08-01T00:30:00Z', '2025-08-02T14:15:00Z', ...],
    'sla_breach_end': ['2025-08-01T00:45:00Z', '2025-08-02T14:30:00Z', ...]
})
```

**Step 2: Create analysis windows**
```python
# Create 1-hour windows around each breach
def create_windows(breaches):
    windows = []
    for _, breach in breaches.iterrows():
        breach_start = pd.to_datetime(breach['sla_breach_start'])
        # Create windows from 2 hours before to 1 hour after
        for offset in range(-120, 60, 60):  # minutes
            window_start = breach_start + pd.Timedelta(minutes=offset)
            windows.append({
                'window_id': f"{breach['breach_id']}_{offset}",
                'from_time': window_start,
                'end_time': window_start + pd.Timedelta(hours=1),
                'sla_breach_start': breach['sla_breach_start'],
                'sla_breach_end': breach['sla_breach_end']
            })
    return pd.DataFrame(windows)
```

**Step 3: Join logs with windows**
```python
logs['timestamp'] = pd.to_datetime(logs['timestamp'])
windows = create_windows(sla_breaches)

# Cross join and filter
result = logs.merge(windows, how='cross')
result = result[
    (result['timestamp'] >= result['from_time']) &
    (result['timestamp'] < result['end_time'])
]

result.to_csv("logs_with_sla_windows.csv", index=False)
```

---

## Examples

### Example 1: Simple Incident Analysis

```bash
# Input: Pre-aggregated hourly counts
python src/anomaly_ranker_engine.py \
    --input examples/data/web_server_logs.csv \
    --incident-start "2025-08-12T20:00:00Z" \
    --incident-end "2025-08-12T23:00:00Z" \
    --workers 4 \
    --out-csv results/web_incident.csv
```

**Input file (web_server_logs.csv)**:
```csv
timestamp,event_id,count
2025-08-12T00:00:00Z,500_INTERNAL_ERROR,2
2025-08-12T00:00:00Z,404_NOT_FOUND,45
2025-08-12T00:00:00Z,200_SUCCESS,9523
2025-08-12T01:00:00Z,500_INTERNAL_ERROR,1
2025-08-12T01:00:00Z,404_NOT_FOUND,38
2025-08-12T01:00:00Z,200_SUCCESS,9812
...
2025-08-12T20:00:00Z,500_INTERNAL_ERROR,450  # Incident!
2025-08-12T20:00:00Z,404_NOT_FOUND,122
2025-08-12T20:00:00Z,200_SUCCESS,5234
```

### Example 2: Database Incident with Raw Logs

```bash
# Input: Raw database logs
python src/anomaly_ranker_engine.py \
    --input examples/data/db_raw_logs.csv \
    --raw \
    --ts-col log_time \
    --event-col error_code \
    --count-col 1 \
    --incident-start "2025-08-13T15:30:00Z" \
    --incident-end "2025-08-13T16:00:00Z" \
    --workers 8
```

**Input file (db_raw_logs.csv)**:
```csv
log_time,error_code,severity,message
2025-08-13T15:15:23.123Z,CONN_TIMEOUT,ERROR,Connection timeout after 30s
2025-08-13T15:15:24.456Z,DEADLOCK_DETECTED,ERROR,Deadlock on table users
2025-08-13T15:15:25.789Z,SLOW_QUERY,WARN,Query took 15s
2025-08-13T15:30:01.111Z,CONN_TIMEOUT,ERROR,Connection pool exhausted
2025-08-13T15:30:02.222Z,CONN_TIMEOUT,ERROR,Connection timeout after 30s
2025-08-13T15:30:03.333Z,CONN_TIMEOUT,ERROR,Connection timeout after 30s
...
```

### Example 3: Predictive Analysis

```bash
# Input: Logs with SLA breach annotations
python src/predictive_pattern_miner.py \
    --input examples/data/microservices_logs.csv \
    --output results/predictive_patterns.csv \
    --n-jobs 8 \
    --freq 1min \
    --lead-lag-horizon 10min
```

**Input file (microservices_logs.csv)**:
```csv
timestamp,log_template_id,sla_breach_start,sla_breach_end,from_time,end_time,service_name
2025-07-01T00:05:12Z,T101,2025-07-01T00:15:00Z,2025-07-01T00:20:00Z,2025-07-01T00:00:00Z,2025-07-01T01:00:00Z,api-gateway
2025-07-01T00:07:33Z,T102,2025-07-01T00:15:00Z,2025-07-01T00:20:00Z,2025-07-01T00:00:00Z,2025-07-01T01:00:00Z,auth-service
2025-07-01T00:08:45Z,T101,2025-07-01T00:15:00Z,2025-07-01T00:20:00Z,2025-07-01T00:00:00Z,2025-07-01T01:00:00Z,api-gateway
2025-07-01T00:15:01Z,T103,2025-07-01T00:15:00Z,2025-07-01T00:20:00Z,2025-07-01T00:00:00Z,2025-07-01T01:00:00Z,database
```

---

## Best Practices

### Data Quality

1. **Consistent Timestamps**: Use ISO 8601 format with timezone
2. **Complete Coverage**: Include baseline periods (not just incidents)
3. **Appropriate Granularity**: Hourly for Ranker, minute-level for Miner
4. **Clean IDs**: Use stable identifiers (Drain template IDs, not raw text)

### Performance Optimization

1. **Pre-Aggregate When Possible**: Ranker is faster with pre-aggregated data
2. **Use Parquet**: 3-5x faster than CSV for large files
3. **Enable DuckDB**: Install `duckdb` for 10x faster raw log aggregation
4. **Tune Workers**: Set `--workers` to CPU core count

### Common Pitfalls

❌ **Don't**: Mix time zones in input data
✅ **Do**: Standardize to UTC

❌ **Don't**: Use raw log messages as event IDs
✅ **Do**: Use template IDs from log parser

❌ **Don't**: Include only incident periods
✅ **Do**: Include sufficient baseline data (at least 7 days)

❌ **Don't**: Ignore missing values
✅ **Do**: Ensure timestamps are complete and valid

---

## Summary

**Anomaly Ranker**:
- Input: (timestamp, event_id, count) at hourly granularity
- Time window: Single incident + baseline
- Output: Ranked anomalies with scores and relationships

**Predictive Miner**:
- Input: (timestamp, template_id, sla_window_info) at minute granularity
- Time windows: Multiple incidents + baselines
- Output: Predictive patterns with lead times

Both engines are flexible and can work with any time series data that fits the (ID, Timestamp, Value) schema.

**Next**: See [examples/](../examples/) for complete working examples
