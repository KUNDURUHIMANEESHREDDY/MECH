# Implementation Plan - Phase 39.15: Advanced Scientific Statistics & Rigor (Expanded)

Establish a publication-grade statistical framework for mechanistic interpretability research. This phase transitions the platform from basic descriptive statistics to a comprehensive suite covering classical hypothesis testing, Bayesian methods, power analysis, and rigorous distribution diagnostics.

## User Review Required

> [!IMPORTANT]
> **Multiple Hypothesis Correction**: This phase introduces mandatory `q-value` (FDR) reporting for any discovery campaign involving multi-component testing (e.g., neuron sweeps). This prevents the "p-hacking" common in large-scale model probing.

> [!CAUTION]
> **Sequential Analysis & Stopping Rules**: We are implementing evidence-based stopping recommendations. The platform will now suggest when an experiment is "sufficiently powered" to stop, or if more samples are required to reach a target power of 0.90.

## Proposed Changes

### 1. Statistical Foundation Layer

#### [NEW] `backend/science/statistics/`
- Create a dedicated package for modular statistical engines:
    - **hypothesis_testing.py**: Implements BH, BY, Bonferroni, and Holm-Bonferroni corrections.
    - **permutation_engine.py**: Generic framework for label, feature, and circuit permutations (10k+ samples).
    - **bootstrap_engine.py**: Advanced CI calculation using Percentile, BCa, and Studentized methods.
    - **power_analysis.py**: Computes observed power and "Required N" for target effect sizes.
    - **effect_sizes.py**: Supports Cohen's d, Hedges' g, Glass Δ, and Cliff's Delta.
    - **diagnostics.py**: Automated normality (Shapiro-Wilk) and outlier (MAD, IQR) detection.
    - **calibration.py**: Implements ECE, MCE, Adaptive ECE, and Brier Scores.
    - **bayesian_comparison.py**: Computes Bayes Factors and Credible Intervals to complement frequentist p-values.

### 2. Orchestration & Traceability

#### [MODIFY] [scientific_validator.py](file:///C:/Users/himaneeshreddyk/Downloads/MECH/New folder (4)/backend/science/reproducibility/scientific_validator.py)
- Refactor to delegate all heavy math to the `backend.science.statistics` package.
- Implement the **Statistical Decision Trace**: Logs every choice (e.g., "Normality failed -> switching to Permutation test") into `statistical_trace.json`.

#### [NEW] [statistical_trace.py](file:///C:/Users/himaneeshreddyk/Downloads/MECH/New folder (4)/backend/science/statistics/statistical_trace.py)
- Manages the audit log for statistical decisions, ensuring researchers can justify their choice of tests.

### 3. Meta-Analysis & Visualization Data

#### [NEW] [meta_analysis.py](file:///C:/Users/himaneeshreddyk/Downloads/MECH/New folder (4)/backend/science/statistics/meta_analysis.py)
- Implements fixed and random effects models to aggregate results across multiple reproduction runs of the same benchmark.

#### [NEW] [visualization_data.py](file:///C:/Users/himaneeshreddyk/Downloads/MECH/New folder (4)/backend/science/statistics/visualization_data.py)
- Generates structured JSON for:
    - Forest Plots (Benchmark comparisons)
    - Volcano Plots (Effect size vs p-value)
    - Reliability Diagrams (Calibration)
    - QQ Plots & Distribution Histograms.

## Verification Plan

### Automated Tests
- `test_advanced_stats_logic.py`: Verify BCa bootstrap intervals against standard R/SciPy implementations.
- `test_fdr_correction.py`: Ensure Benjamini-Hochberg correctly bounds the false discovery rate on synthetic null sets.
- `test_power_and_n.py`: Verify "Required N" calculation correctly predicts sample needs for small effect sizes (d=0.2).
- `test_bayesian_frequentist_parity.py`: Ensure Bayes Factors and p-values provide consistent directional evidence.

### Manual Verification
- Execute a full reproduction suite run and verify the new `statistical_trace.json` is signed and included in the `Research Bundle`.
- Inspect the `scientific_validation.md` for the "Statistical Rigor Dashboard" including Power, ECE, and Forest Plot data.
