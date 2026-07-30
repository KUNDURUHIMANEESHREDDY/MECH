"""
Base classes for scientific reproduction benchmarks.

Defines the core interfaces for reproducing landmark mechanistic interpretability papers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json


class MetricType(Enum):
    """Types of metrics used in reproduction benchmarks."""

    ACCURACY = "accuracy"
    F1 = "f1"
    PRECISION = "precision"
    RECALL = "recall"
    IOI_LOGIT_DIFF = "ioi_logit_diff"
    INDUCTION_SCORE = "induction_score"
    CIRCUIT_ACCURACY = "circuit_accuracy"
    SPARSITY = "sparsity"
    RECONSTRUCTION_ERROR = "reconstruction_error"
    LOGIT_LENS_KL = "logit_lens_kl"
    PATCHING_EFFECT = "patching_effect"
    CORRELATION = "correlation"
    COSINE_SIMILARITY = "cosine_similarity"
    CUSTOM = "custom"


@dataclass
class PaperMetadata:
    """Metadata for a landmark paper being reproduced."""

    title: str
    authors: List[str]
    year: int
    venue: str
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    key_claims: List[str] = field(default_factory=list)
    reported_metrics: Dict[str, float] = field(default_factory=dict)
    model_requirements: Dict[str, Any] = field(default_factory=dict)
    dataset_requirements: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class MetricComparison:
    """Comparison between reported and reproduced metric values."""

    metric_name: str
    metric_type: MetricType
    reported_value: float
    reproduced_value: float
    absolute_difference: float
    relative_difference: float
    tolerance: float
    passed: bool
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "metric_type": self.metric_type.value,
            "reported_value": self.reported_value,
            "reproduced_value": self.reproduced_value,
            "absolute_difference": self.absolute_difference,
            "relative_difference": self.relative_difference,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "notes": self.notes,
        }


@dataclass
class ReproductionResult:
    """Result of running a single reproduction benchmark."""

    benchmark_name: str
    paper_metadata: PaperMetadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    model_used: str = ""
    dataset_used: str = ""
    metric_comparisons: List[MetricComparison] = field(default_factory=list)
    execution_time_seconds: float = 0.0
    success: bool = False
    error_message: Optional[str] = None
    raw_results: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)  # path -> description

    def add_comparison(self, comparison: MetricComparison) -> None:
        self.metric_comparisons.append(comparison)

    def overall_passed(self) -> bool:
        return all(c.passed for c in self.metric_comparisons) and self.success

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "paper_metadata": {
                "title": self.paper_metadata.title,
                "authors": self.paper_metadata.authors,
                "year": self.paper_metadata.year,
                "venue": self.paper_metadata.venue,
                "arxiv_id": self.paper_metadata.arxiv_id,
                "key_claims": self.paper_metadata.key_claims,
                "reported_metrics": self.paper_metadata.reported_metrics,
            },
            "timestamp": self.timestamp,
            "model_used": self.model_used,
            "dataset_used": self.dataset_used,
            "metric_comparisons": [c.to_dict() for c in self.metric_comparisons],
            "execution_time_seconds": self.execution_time_seconds,
            "success": self.success,
            "error_message": self.error_message,
            "raw_results": self.raw_results,
            "artifacts": self.artifacts,
        }

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class ReproductionReport:
    """Comprehensive report for a reproduction attempt."""

    benchmark_name: str
    paper_metadata: PaperMetadata
    results: List[ReproductionResult]
    summary: Dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_result(self, result: ReproductionResult) -> None:
        self.results.append(result)

    def generate_summary(self) -> Dict[str, Any]:
        total_metrics = sum(len(r.metric_comparisons) for r in self.results)
        passed_metrics = sum(
            1 for r in self.results for c in r.metric_comparisons if c.passed
        )
        successful_runs = sum(1 for r in self.results if r.success)

        self.summary = {
            "total_runs": len(self.results),
            "successful_runs": successful_runs,
            "total_metrics_compared": total_metrics,
            "metrics_passed": passed_metrics,
            "metrics_failed": total_metrics - passed_metrics,
            "overall_pass_rate": passed_metrics / total_metrics if total_metrics > 0 else 0.0,
            "run_pass_rate": successful_runs / len(self.results) if self.results else 0.0,
        }
        return self.summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "paper_metadata": {
                "title": self.paper_metadata.title,
                "authors": self.paper_metadata.authors,
                "year": self.paper_metadata.year,
                "venue": self.paper_metadata.venue,
                "arxiv_id": self.paper_metadata.arxiv_id,
            },
            "results": [r.to_dict() for r in self.results],
            "summary": self.generate_summary(),
            "generated_at": self.generated_at,
        }

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_markdown(self, path: str) -> None:
        lines = [
            f"# Reproduction Report: {self.benchmark_name}",
            f"**Paper**: {self.paper_metadata.title}",
            f"**Authors**: {', '.join(self.paper_metadata.authors)}",
            f"**Year**: {self.paper_metadata.year}",
            f"**Venue**: {self.paper_metadata.venue}",
            f"**arXiv**: {self.paper_metadata.arxiv_id or 'N/A'}",
            f"**Generated**: {self.generated_at}",
            "",
            "## Summary",
            "",
        ]
        summary = self.generate_summary()
        for k, v in summary.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

        for i, result in enumerate(self.results):
            lines.extend([
                f"## Run {i+1}",
                f"- **Model**: {result.model_used}",
                f"- **Dataset**: {result.dataset_used}",
                f"- **Success**: {result.success}",
                f"- **Time**: {result.execution_time_seconds:.2f}s",
                "",
                "### Metric Comparisons",
                "",
                "| Metric | Reported | Reproduced | Abs Diff | Rel Diff | Tolerance | Passed |",
                "|--------|----------|------------|----------|----------|-----------|--------|",
            ])
            for comp in result.metric_comparisons:
                lines.append(
                    f"| {comp.metric_name} | {comp.reported_value:.6f} | {comp.reproduced_value:.6f} | "
                    f"{comp.absolute_difference:.6f} | {comp.relative_difference:.4f} | "
                    f"{comp.tolerance:.4f} | {'✅' if comp.passed else '❌'} |"
                )
            if result.error_message:
                lines.append(f"\n**Error**: {result.error_message}")
            lines.append("")

        with open(path, "w") as f:
            f.write("\n".join(lines))


class ReproductionBenchmark(ABC):
    """Abstract base class for reproduction benchmarks.

    Each benchmark implements the methodology from a specific paper
    and compares results against reported metrics.
    """

    def __init__(
        self,
        name: str,
        paper_metadata: PaperMetadata,
        tolerance: float = 0.1,
    ):
        self.name = name
        self.paper_metadata = paper_metadata
        self.tolerance = tolerance  # relative tolerance for metric comparison

    @abstractmethod
    def setup(self, model_name: str, **kwargs) -> None:
        """Set up the benchmark: load model, prepare data, initialize hooks."""
        pass

    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """Execute the reproduction experiment and return raw results."""
        pass

    @abstractmethod
    def compute_metrics(self, raw_results: Dict[str, Any]) -> Dict[str, float]:
        """Compute metrics from raw experimental results."""
        pass

    def compare_metrics(
        self,
        computed_metrics: Dict[str, float],
    ) -> List[MetricComparison]:
        """Compare computed metrics against reported values."""
        comparisons = []
        for metric_name, reported_value in self.paper_metadata.reported_metrics.items():
            if metric_name in computed_metrics:
                reproduced_value = computed_metrics[metric_name]
                abs_diff = abs(reproduced_value - reported_value)
                rel_diff = abs_diff / abs(reported_value) if abs(reported_value) > 1e-10 else abs_diff
                passed = rel_diff <= self.tolerance

                comparisons.append(MetricComparison(
                    metric_name=metric_name,
                    metric_type=MetricType.CUSTOM,
                    reported_value=reported_value,
                    reproduced_value=reproduced_value,
                    absolute_difference=abs_diff,
                    relative_difference=rel_diff,
                    tolerance=self.tolerance,
                    passed=passed,
                ))
        return comparisons

    def create_result(
        self,
        model_name: str,
        dataset_name: str,
        raw_results: Dict[str, Any],
        execution_time: float,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> ReproductionResult:
        """Create a ReproductionResult from benchmark execution."""
        computed_metrics = self.compute_metrics(raw_results)
        comparisons = self.compare_metrics(computed_metrics)

        result = ReproductionResult(
            benchmark_name=self.name,
            paper_metadata=self.paper_metadata,
            model_used=model_name,
            dataset_used=dataset_name,
            metric_comparisons=comparisons,
            execution_time_seconds=execution_time,
            success=success,
            error_message=error_message,
            raw_results={**raw_results, "computed_metrics": computed_metrics},
        )
        return result

    def run_benchmark(
        self,
        model_name: str,
        dataset_name: str = "",
        **kwargs,
    ) -> ReproductionResult:
        """Run the full benchmark pipeline."""
        import time
        start_time = time.time()

        try:
            self.setup(model_name, **kwargs)
            raw_results = self.run(**kwargs)
            execution_time = time.time() - start_time
            return self.create_result(
                model_name=model_name,
                dataset_name=dataset_name,
                raw_results=raw_results,
                execution_time=execution_time,
                success=True,
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return self.create_result(
                model_name=model_name,
                dataset_name=dataset_name,
                raw_results={},
                execution_time=execution_time,
                success=False,
                error_message=str(e),
            )