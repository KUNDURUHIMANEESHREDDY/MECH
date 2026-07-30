"""
Scientific Validation & Reproduction Framework.

Provides infrastructure for reproducing landmark mechanistic interpretability
papers with automated benchmarking, metric comparison, and reporting.
"""

from .base import (
    ReproductionBenchmark,
    ReproductionResult,
    MetricComparison,
    ReproductionReport,
    PaperMetadata,
)

from .runner import ReproductionRunner

from .papers import (
    IOIBenchmark,
    InductionHeadsBenchmark,
    GreaterThanBenchmark,
    ModularArithmeticBenchmark,
    SAEBenchmark,
    LogitLensBenchmark,
    ActivationPatchingBenchmark,
)

__all__ = [
    "ReproductionBenchmark",
    "ReproductionResult",
    "MetricComparison",
    "ReproductionReport",
    "PaperMetadata",
    "ReproductionRunner",
    "IOIBenchmark",
    "InductionHeadsBenchmark",
    "GreaterThanBenchmark",
    "ModularArithmeticBenchmark",
    "SAEBenchmark",
    "LogitLensBenchmark",
    "ActivationPatchingBenchmark",
]