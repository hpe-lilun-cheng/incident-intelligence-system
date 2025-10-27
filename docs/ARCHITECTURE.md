# System Architecture

## Table of Contents
1. [Overview](#overview)
2. [Dual-Engine Design](#dual-engine-design)
3. [Anomaly Ranker Architecture](#anomaly-ranker-architecture)
4. [Predictive Pattern Miner Architecture](#predictive-pattern-miner-architecture)
5. [Data Flow](#data-flow)
6. [Integration Points](#integration-points)

---

## Overview

The Incident Intelligence System uses a **dual-engine architecture** to provide both reactive and proactive incident management:

```
┌─────────────────────────────────────────────────────────────┐
│                   INCIDENT INTELLIGENCE                      │
│                         SYSTEM                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────┐      ┌────────────────────────┐   │
│  │  ANOMALY RANKER     │      │  PREDICTIVE PATTERN   │   │
│  │     ENGINE          │      │       MINER           │   │
│  ├─────────────────────┤      ├────────────────────────┤   │
│  │ Reactive Analysis   │      │ Proactive Prevention  │   │
│  │ Real-time Detection │      │ Historical Learning   │   │
│  │ Root Cause ID       │      │ Early Warning         │   │
│  └─────────────────────┘      └────────────────────────┘   │
│           ↓                              ↓                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Unified Output Schema & Action Layer         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Key Principles

1. **Separation of Concerns**: Each engine focuses on its strength
2. **Complementary Design**: Ranker (reactive) + Miner (proactive) = complete solution
3. **Unified Input Format**: Both engines work with standardized time series data
4. **Modular Architecture**: Components can be used independently or together

---

## Dual-Engine Design

### Engine 1: Anomaly Ranker
**Purpose**: "What's wrong RIGHT NOW?"

**Use Case**: Active incident analysis
- Analyzes log patterns during incident windows
- Identifies anomalous events using multiple detection methods
- Finds correlated failures and cascades
- Generates root cause hypotheses

**Analogy**: Emergency room doctor treating current crisis

### Engine 2: Predictive Pattern Miner
**Purpose**: "What will go wrong SOON?"

**Use Case**: Proactive monitoring
- Learns from historical SLA breach patterns
- Identifies log patterns that precede failures
- Computes lead times for early warnings
- Enables prevention rather than reaction

**Analogy**: Preventive medicine specialist stopping future problems

### Why Two Engines?

| Aspect | Anomaly Ranker | Predictive Miner |
|--------|---------------|------------------|
| **Time Focus** | Present (incident window) | Future (prediction) |
| **Data** | Single incident + baseline | Multiple historical incidents |
| **Output** | Anomaly scores | Predictive scores + lead times |
| **Goal** | Root cause identification | Early warning system |
| **Latency** | Minutes (real-time) | Hours/days (batch learning) |

---

## Anomaly Ranker Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│              ANOMALY RANKER ENGINE                           │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Layer 0: Foundation (anomaly_ranker_foundation.py)│    │
│  │  • Statistical library                              │    │
│  │  • Reusable functions                              │    │
│  │  • Can be imported or run standalone               │    │
│  └────────────────────────────────────────────────────┘    │
│                        ↓ imports                             │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Layer 1: Statistical Ensemble (60% weight)        │    │
│  │  ┌──────────────────────────────────────────┐      │    │
│  │  │ Z-score (35%)  │ GLR (25%)  │ PMI (15%) │      │    │
│  │  │ Cohen's d (10%)│ Run len (7%)│ CUSUM (5%)│      │    │
│  │  │ Var ratio (2%) │ Z-drop (1%) │           │      │    │
│  │  └──────────────────────────────────────────┘      │    │
│  └────────────────────────────────────────────────────┘    │
│                        ↓                                     │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Layer 2: ML Pattern Detectors (40% weight)       │    │
│  │  ┌──────────────────────────────────────────┐      │    │
│  │  │ Pattern Break (20%) │ Volume Spike (12%) │      │    │
│  │  │ Isolation Forest (8%)                    │      │    │
│  │  └──────────────────────────────────────────┘      │    │
│  └────────────────────────────────────────────────────┘    │
│                        ↓                                     │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Layer 3: Relational Analysis (Additive Boosts)   │    │
│  │  ┌──────────────────────────────────────────┐      │    │
│  │  │ Correlation (+0.3 per strong correlation)│      │    │
│  │  │ Cascade (+0.5 per frequent cascade)      │      │    │
│  │  └──────────────────────────────────────────┘      │    │
│  └────────────────────────────────────────────────────┘    │
│                        ↓                                     │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Layer 4: Root Cause Synthesis                     │    │
│  │  • Hypothesis generation                           │    │
│  │  • Enhanced scoring (can exceed 100%)              │    │
│  │  • Actionable recommendations                      │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Layer 1: Statistical Ensemble

**Purpose**: Robust anomaly detection using multiple statistical signals

**Components**:

1. **Robust Z-Score (35%)**
   - Compares incident values to hour-of-day baselines
   - Uses MAD/IQR for outlier resistance
   - Most reliable single signal

2. **Poisson GLR Test (25%)**
   - Log-likelihood ratio for count data
   - Detects rate changes in event frequencies
   - Theoretically sound for Poisson processes

3. **PMI Overrepresentation (15%)**
   - Pointwise Mutual Information
   - Identifies events overrepresented during incidents
   - Formula: log(P(event|incident) / P(event))

4. **Cohen's d (10%)**
   - Effect size measure
   - Quantifies practical significance
   - Uses pooled standard deviation

5. **Run Length (7%)**
   - Consecutive periods above 2σ threshold
   - Detects sustained anomalies
   - Filters transient spikes

6. **CUSUM (5%)**
   - Cumulative sum change detection
   - Identifies process shifts
   - Bidirectional (increases and decreases)

7. **Variance Ratio (2%)**
   - Compares volatility during vs before incident
   - Detects instability patterns

8. **Z-Drop (1%)**
   - Detects significant decreases
   - Important for "missing" events (e.g., heartbeats)

**Scoring Formula**:
```
statistical_score = Σ(weight_i × standardized_feature_i)
confidence = coverage_incident × coverage_pre × stability
final_score = statistical_score × confidence
```

### Layer 2: ML Pattern Detectors

**Purpose**: Catch complex anomalies that statistical tests miss

**Components**:

1. **Pattern Break Detector (20%)**
   - Compares to hour-of-day patterns
   - Uses clean baselines (excludes incident windows)
   - Proximity boost: events near incident midpoint score higher
   
   ```python
   for hour_of_day in 0..23:
       expected = baseline[event][hour_of_day]
       z_score = (actual - expected) / scale
       if |z_score| > threshold:
           score = |z_score| + proximity_boost
   ```

2. **Volume Spike Detector (12%)**
   - Rate-of-change analysis
   - Combines acceleration and magnitude
   - Formula: `score = |rate_change| × 2 + max(0, spike_magnitude - 2)`

3. **Isolation Forest (8%)**
   - Unsupervised outlier detection
   - Multi-dimensional feature space:
     * Event count
     * Hour of day
     * Day of week
     * Is weekend
     * Is business hours
   - Adaptive contamination rate: `clip(10/N, 0.01, 0.1)`

### Layer 3: Relational Analysis

**Purpose**: Identify relationships between anomalies for root cause analysis

**Components**:

1. **Correlation Analysis**
   - Pearson (linear relationships)
   - Spearman (monotonic relationships)
   - Cross-correlation with lags (±5 hours)
   - Top-K gating to avoid O(E²) complexity
   - **Boost**: +0.3 per strong correlation (|r| > 0.7)

2. **Cascade Detection**
   - Identifies A→B temporal sequences
   - Window adapts to data cadence
   - Frequency counting to find common patterns
   - **Boost**: +0.5 per frequently observed cascade

**Key Innovation**: No zero-filling in correlations
```python
# Only use overlapping valid data points
mask = ~(pd.isna(series1) | pd.isna(series2))
x = series1[mask]
y = series2[mask]
correlation = pearsonr(x, y)
```

### Layer 4: Root Cause Synthesis

**Purpose**: Transform multi-signal evidence into actionable insights

**Output**:
```json
{
  "primary_triggers": [
    {
      "event_id": "DB_CONNECTION_TIMEOUT",
      "score": 124.5,
      "reasons": ["pattern_break", "volume_spike", "triggered_cascade"]
    }
  ],
  "cascades": [
    {"source": "DB_TIMEOUT", "target": "API_ERROR", "frequency": 12}
  ],
  "hypotheses": [
    "Database connection pool exhaustion triggered API failures"
  ]
}
```

---

## Predictive Pattern Miner Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│           PREDICTIVE PATTERN MINER ENGINE                    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Stage 1: Window Aggregation                       │    │
│  │  • Count templates per incident/baseline window    │    │
│  │  • Chunked processing for large datasets           │    │
│  └────────────────────────────────────────────────────┘    │
│                        ↓                                     │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Stage 2: Statistical Significance                 │    │
│  │  • Fisher Exact Test: P(overrepresentation)       │    │
│  │  • Log-odds Z-score: Effect size                  │    │
│  │  • Benjamini-Hochberg FDR: Control false positives│    │
│  └────────────────────────────────────────────────────┘    │
│                        ↓                                     │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Stage 3: Pattern Quality                          │    │
│  │  • Lift: Incident rate / baseline rate            │    │
│  │  • Clarity: Entropy-based concentration           │    │
│  └────────────────────────────────────────────────────┘    │
│                        ↓                                     │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Stage 4: Temporal Correlation (Top-K only)       │    │
│  │  • Lead/lag cross-correlation                      │    │
│  │  • Identifies early warning signals               │    │
│  │  • Parallel processing per template                │    │
│  └────────────────────────────────────────────────────┘    │
│                        ↓                                     │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Stage 5: Composite Scoring                        │    │
│  │  score = z_weight×z + lift + clarity + β×lead_corr│    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Stage 1: Window Aggregation

**Purpose**: Count template occurrences in incident vs baseline windows

**Input**:
```
timestamp, template_id, sla_breach_start, sla_breach_end, window_start, window_end
```

**Process**:
- Group by (template_id, window_id)
- Count occurrences
- Classify windows as incident or baseline
- Memory-efficient: chunked processing for 20M+ rows

### Stage 2: Statistical Significance

**Fisher Exact Test**:
```
Contingency Table:
                  In Incident  Not In Incident
Has Template            c_i           c_b
No Template           N_i - c_i     N_b - c_b

P-value = P[X >= c_i | X ~ Hypergeometric(N, K, n)]
```

**Log-Odds Z-Score**:
```python
p_i = c_i / N_i  # Incident probability
p_b = c_b / N_b  # Baseline probability
logit_i = log(p_i / (1 - p_i))
logit_b = log(p_b / (1 - p_b))
z = (logit_i - logit_b) / sqrt(var_i + var_b)
```

**Benjamini-Hochberg FDR**:
```python
# Sort p-values: p(1) <= p(2) <= ... <= p(m)
# Reject H_i if p(i) <= (i/m) × α
q_values = p_values × m / ranks
significant = q_values <= α
```

### Stage 3: Pattern Quality

**Lift**:
```
Lift = (c_i / N_i) / (c_b / N_b)
Interpretation: How many times more frequent in incidents
Capped at 10.0 to prevent extreme values
```

**Clarity (Entropy-Based)**:
```python
# For each context column (e.g., service_name)
for value in unique_values:
    p = count(value) / total
    entropy -= p × log(p)

max_entropy = log(num_unique_values)
clarity = 1.0 - (entropy / max_entropy)

# High clarity = concentrated in few contexts (actionable)
# Low clarity = spread across many contexts (noisy)
```

### Stage 4: Temporal Correlation

**Purpose**: Find templates that precede SLA breaches (early warning)

**Process**:
1. For each window, create time series:
   - x = template activity (1-minute buckets)
   - y = SLA breach indicator (1 during breach, 0 otherwise)

2. Compute normalized cross-correlation at negative lags:
   ```python
   # Negative lag means template leads SLA breach
   corrs, lags = normalized_xcorr(x, y, max_lag=-10min)
   best_r = max(corrs)  # Highest positive correlation
   lead_time = lags[argmax(corrs)]  # When template appears before breach
   ```

3. Weight by template activity:
   ```python
   # Templates with more activity get higher weight
   r_weighted = Σ(w_i × r_i) where w_i = activity_i / Σ(activity)
   ```

**Parallel Processing**: Only computed for top-K templates after preliminary scoring

### Stage 5: Composite Scoring

**Formula**:
```python
score = (
    z_weight × z_component +          # Statistical significance (default 35%)
    lift_component +                   # Frequency ratio
    clarity_component +                # Pattern quality
    beta_lead × lead_correlation       # Predictive boost (default 30%)
)
```

**Component Breakdown**:
- **z_component**: Standardized statistical significance
- **lift_component**: min(lift, max_lift) - capped at 10
- **clarity_component**: clarity^power - configurable exponent
- **lead_correlation**: Only positive correlations (template leads breach)

**Output**:
```csv
template_id,score,z,lift,clarity,lead_corr_pos,lead_lag_seconds,p_value,q_value,significant_fdr
T501,15.3,4.2,8.5,0.82,0.65,-180,0.001,0.015,True
T502,12.7,3.8,6.2,0.75,0.58,-240,0.002,0.018,True
```

---

## Data Flow

### Input → Ranker → Output

```
┌────────────────┐
│  Raw Logs or   │
│ Aggregated Data│
└────────┬───────┘
         │ load_or_aggregate()
         ↓
┌────────────────┐
│  Hourly Pivot  │  (Events × Time)
│    Table       │
└────────┬───────┘
         │ rank_events_parallel()
         ↓
┌────────────────────┐
│ Statistical Scores │  Layer 1 output
└────────┬───────────┘
         │ detect_*_parallel()
         ↓
┌────────────────────┐
│  ML Detector       │  Layer 2 output
│    Scores          │
└────────┬───────────┘
         │ analyze_correlations()
         ↓
┌────────────────────┐
│ Correlation &      │  Layer 3 output
│ Cascade Analysis   │
└────────┬───────────┘
         │ synthesize()
         ↓
┌────────────────────┐
│ Enhanced Scores +  │  Final output
│ Hypotheses         │
└────────────────────┘
```

### Input → Miner → Output

```
┌────────────────┐
│  Logs with SLA │
│    Windows     │
└────────┬───────┘
         │ fit()
         ↓
┌────────────────────┐
│ Count Aggregation  │  Per window
└────────┬───────────┘
         │ compute_scores()
         ↓
┌────────────────────┐
│ Statistical Tests  │  Fisher + Z-score
└────────┬───────────┘
         │ BH-FDR correction
         ↓
┌────────────────────┐
│ Quality Metrics    │  Lift + Clarity
└────────┬───────────┘
         │ compute_leadlag() [top-K only]
         ↓
┌────────────────────┐
│ Temporal Analysis  │  Lead/lag correlation
└────────┬───────────┘
         │ composite_score()
         ↓
┌────────────────────┐
│ Ranked Predictive  │  Final output
│    Patterns        │
└────────────────────┘
```

---

## Integration Points

### Unified Data Schema

Both engines accept standardized time series:
```
Required Columns:
- id: Event/template identifier (string)
- timestamp: When event occurred (datetime)
- value: Metric value (count, sum, avg, etc.)

Optional:
- context: Additional dimensions for analysis
```

### Output Integration

**Common Schema**:
```python
{
    "id": "EVENT_123",
    "score": 8.45,
    "type": "anomaly" | "predictive",
    "timestamp": "2025-08-12T20:15:00Z",
    "confidence": 0.92,
    "metadata": {...}
}
```

### Deployment Patterns

**Pattern 1: Real-Time + Batch**
```
Real-time: Ranker analyzes active incidents
Batch: Miner runs nightly to update predictive models
```

**Pattern 2: Unified Platform**
```
┌──────────────────────────────────┐
│      Alert Management System     │
├──────────────────────────────────┤
│  • Incident detected             │
│  → Run Ranker Engine             │
│  → Check Miner predictions       │
│  → Synthesize recommendations    │
└──────────────────────────────────┘
```

**Pattern 3: Progressive Enhancement**
```
Week 1: Deploy Ranker only (reactive)
Week 2: Add Miner (proactive)
Week 3: Integrate both with alert system
```

---

## Performance Characteristics

### Anomaly Ranker

**Time Complexity**: O(H × E) where H = hours, E = events
- Statistical features: O(H × E)
- ML detectors: O(H × E × log(H)) per event
- Correlations: O(K² × H) where K = top-K events
- **Total**: ~2-5 minutes for 1000 events, 1000 hours, 8 workers

**Space Complexity**: O(H × E) for pivot table
- Typical: 1000 events × 1000 hours × 4 bytes = 4 MB
- Large: 10K events × 10K hours = 400 MB

### Predictive Pattern Miner

**Time Complexity**: O(L × W + K × W × B × M)
- Window counting: O(L × W) where L = log rows, W = windows
- Lead/lag: O(K × W × B × M) where K = top-K, B = buckets, M = max_lag
- **Total**: ~10-30 minutes for 20M rows, 1000 windows, 8 workers

**Space Complexity**: O(L + W × T) where T = unique templates
- Chunked processing keeps memory under 2GB even for 20M rows

---

## Summary

The dual-engine architecture provides:
1. ✅ **Comprehensive Coverage**: Reactive + Proactive
2. ✅ **Modular Design**: Each engine usable independently
3. ✅ **Scalable**: Parallel processing, chunked pipelines
4. ✅ **Statistically Rigorous**: Multiple testing control, effect sizes
5. ✅ **Production-Ready**: Battle-tested on millions of events

**Next**: See [INPUT_FORMATS.md](INPUT_FORMATS.md) for data specifications
