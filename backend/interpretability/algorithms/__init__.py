"""Interpretability Algorithms package.

Imports are deferred to first attribute access: logit_lens pulls in torch and
transformers (~8s), which would otherwise stall every app/API startup.
"""

_LAZY_EXPORTS = {
    "ActivationQuery": (".activation_search", "ActivationQuery"),
    "ActivationSearchEngine": (".activation_search", "ActivationSearchEngine"),
    "AttentionHeadRanker": (".attention_head_ranker", "AttentionHeadRanker"),
    "FeatureSearchEngine": (".feature_search", "FeatureSearchEngine"),
    "LogitLens": (".logit_lens", "LogitLens"),
    "AlgorithmRegistry": (".registry", "AlgorithmRegistry"),
    "get_algorithm_registry": (".registry", "get_algorithm_registry"),
    "TunedLens": (".tuned_lens", "TunedLens"),
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
