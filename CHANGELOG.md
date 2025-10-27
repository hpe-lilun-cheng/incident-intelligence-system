# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-15

### Added
- Initial release of Incident Intelligence System
- Anomaly Ranker Engine with dual-layer detection (statistical + ML)
- Predictive Pattern Miner for SLA breach forecasting
- Comprehensive documentation (Architecture, Input Formats, Algorithms)
- Example scripts and sample data generation
- Full test suite
- MIT License

### Features
- **Anomaly Ranker**:
  - 8 statistical detection methods (Z-score, GLR, PMI, Cohen's d, etc.)
  - 3 ML pattern detectors (Pattern Break, Volume Spike, Isolation Forest)
  - Correlation analysis with lag detection
  - Cascade detection for root cause identification
  - Parallel processing support
  
- **Predictive Pattern Miner**:
  - Fisher Exact Test for statistical significance
  - Lead/lag correlation analysis (up to 10 minutes advance warning)
  - Clarity scoring with entropy-based filtering
  - FDR control via Benjamini-Hochberg correction
  - Chunked processing for large datasets (20M+ rows)

### Changed
- Renamed files for clarity:
  - `log_anomaly_ranker_parallel.py` → `anomaly_ranker_foundation.py`
  - `fully_parallel_enhanced_anomaly_ranker_fixed.py` → `anomaly_ranker_engine.py`
  - `gray_miner.py` → `predictive_pattern_miner.py`

### Performance
- Handles 1000 events × 1000 hours in ~2-5 minutes
- Processes 20M+ log rows in ~10-30 minutes
- Parallel processing scales linearly with CPU cores

### Documentation
- Complete architecture documentation
- Detailed input format specifications
- Algorithm explanations with formulas
- Working examples and tutorials
- API reference

## [Unreleased]

### Planned
- Real-time streaming support
- Kubernetes operator for automated deployment
- Grafana dashboard integration
- Additional ML detectors (LSTM, Transformer-based)
- Multi-variate anomaly detection
- Automated threshold tuning

---

[1.0.0]: https://github.com/yourusername/incident-intelligence-system/releases/tag/v1.0.0
