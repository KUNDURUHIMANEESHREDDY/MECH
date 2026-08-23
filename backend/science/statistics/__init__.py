"""Statistical analysis toolkit — imports deferred to first attribute access."""

_LAZY_EXPORTS = {
    "HypothesisTesting": (".hypothesis_testing", "HypothesisTesting"),
    "PermutationEngine": (".permutation_engine", "PermutationEngine"),
    "BootstrapEngine": (".bootstrap_engine", "BootstrapEngine"),
    "PowerAnalysis": (".power_analysis", "PowerAnalysis"),
    "EffectSizes": (".effect_sizes", "EffectSizes"),
    "CalibrationMetrics": (".calibration", "CalibrationMetrics"),
    "DistributionDiagnostics": (".diagnostics", "DistributionDiagnostics"),
    "MetaAnalysis": (".meta_analysis", "MetaAnalysis"),
    "SequentialAnalysis": (".sequential_analysis", "SequentialAnalysis"),
    "VisualizationDataGenerator": (".visualization_data", "VisualizationDataGenerator"),
    "StatisticalTrace": (".statistical_trace", "StatisticalTrace"),
    "StatisticalValidator": (".statistical_validator", "StatisticalValidator"),
    "StatisticalProtocol": (".statistical_protocol", "StatisticalProtocol"),
    "BayesianFrequentistComparison": (".bayesian_frequentist", "BayesianFrequentistComparison"),
    "StatisticalRecommender": (".statistical_recommender", "StatisticalRecommender"),
    "StatisticalQualityScore": (".statistical_quality", "StatisticalQualityScore"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    entry = _LAZY_EXPORTS.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(entry[0], __name__)
    value = getattr(module, entry[1])
    globals()[name] = value
    return value
