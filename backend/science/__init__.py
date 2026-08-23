"""Science package — imports deferred to first attribute access."""

_LAZY_EXPORTS = {
    "BenchmarkingOrchestrator": (".benchmarking_orchestrator", "BenchmarkingOrchestrator"),
    "PeerReviewSystem": (".peer_review", "PeerReviewSystem"),
    "StatisticalValidator": (".statistics.statistical_validator", "StatisticalValidator"),
    "StatisticalProtocol": (".statistics.statistical_protocol", "StatisticalProtocol"),
    "PhaseGatekeeper": (".phase_gatekeeper", "PhaseGatekeeper"),
    "ResearchPortal": (".research_portal", "ResearchPortal"),
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
