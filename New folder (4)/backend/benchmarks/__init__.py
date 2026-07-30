"""Benchmarks Package — Real Mechanistic Interpretability Benchmark Suite."""

from .model_registry import ModelFamily, ModelSpec, ModelRegistry, ModelAvailability, MODEL_CATALOGUE
from .benchmark_tasks import (
    BenchmarkTask,
    BenchmarkTaskSpec,
    BenchmarkResult,
    BenchmarkTaskExecutor,
    PublishedReference,
    TASK_CATALOGUE,
)
from .benchmark_runner import BenchmarkRunner, BenchmarkReport, ModelBenchmarkSuite
from .kg_integrator import BenchmarkKGIntegrator

__all__ = [
    # Model registry
    "ModelFamily", "ModelSpec", "ModelRegistry", "ModelAvailability", "MODEL_CATALOGUE",
    # Tasks
    "BenchmarkTask", "BenchmarkTaskSpec", "BenchmarkResult",
    "BenchmarkTaskExecutor", "PublishedReference", "TASK_CATALOGUE",
    # Runner
    "BenchmarkRunner", "BenchmarkReport", "ModelBenchmarkSuite",
    # KG
    "BenchmarkKGIntegrator",
]
