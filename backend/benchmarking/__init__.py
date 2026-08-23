"""MECH benchmarking package.

Submodules are imported lazily: kg_integrator (and its knowledge-graph /
interpretability dependencies) pulls in torch and transformers, which would
otherwise add ~13s to every app/API startup. Names resolve on first access.
"""

_LAZY_EXPORTS = {
    "ContinuousBenchmarking": (".continuous_benchmarking", "ContinuousBenchmarking"),
    "DashboardGenerator": (".dashboard_generator", "DashboardGenerator"),
    "BenchmarkTask": (".benchmark_tasks", "BenchmarkTask"),
    "ExecutionMode": (".benchmark_tasks", "ExecutionMode"),
    "PublishedReference": (".benchmark_tasks", "PublishedReference"),
    "BenchmarkTaskSpec": (".benchmark_tasks", "BenchmarkTaskSpec"),
    "BenchmarkResult": (".benchmark_tasks", "BenchmarkResult"),
    "BenchmarkTaskExecutor": (".benchmark_tasks", "BenchmarkTaskExecutor"),
    "ModelBenchmarkSuite": (".benchmark_runner", "ModelBenchmarkSuite"),
    "BenchmarkReport": (".benchmark_runner", "BenchmarkReport"),
    "BenchmarkRunner": (".benchmark_runner", "BenchmarkRunner"),
    "ModelFamily": (".model_registry", "ModelFamily"),
    "ModelSpec": (".model_registry", "ModelSpec"),
    "ModelAvailability": (".model_registry", "ModelAvailability"),
    "ModelRegistry": (".model_registry", "ModelRegistry"),
    "BenchmarkKGIntegrator": (".kg_integrator", "BenchmarkKGIntegrator"),
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
