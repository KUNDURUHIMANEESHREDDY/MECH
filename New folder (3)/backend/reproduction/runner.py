"""
Reproduction Runner - orchestrates multiple benchmarks and generates reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import time

from .base import (
    ReproductionBenchmark,
    ReproductionResult,
    ReproductionReport,
    PaperMetadata,
)


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    benchmark: ReproductionBenchmark
    model_name: str
    dataset_name: str = ""
    kwargs: Dict[str, Any] = field(default_factory=dict)
    repetitions: int = 1


class ReproductionRunner:
    """Orchestrates running multiple reproduction benchmarks and generating reports."""

    def __init__(
        self,
        output_dir: str = "reproduction_results",
        default_tolerance: float = 0.1,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.default_tolerance = default_tolerance
        self.benchmarks: Dict[str, ReproductionBenchmark] = {}
        self.results: List[ReproductionResult] = []

    def register_benchmark(self, benchmark: ReproductionBenchmark) -> None:
        """Register a benchmark for execution."""
        self.benchmarks[benchmark.name] = benchmark

    def run_benchmark(
        self,
        benchmark_name: str,
        model_name: str,
        dataset_name: str = "",
        repetitions: int = 1,
        **kwargs,
    ) -> List[ReproductionResult]:
        """Run a single benchmark multiple times."""
        if benchmark_name not in self.benchmarks:
            raise ValueError(f"Benchmark '{benchmark_name}' not registered")

        benchmark = self.benchmarks[benchmark_name]
        results = []

        for i in range(repetitions):
            print(f"Running {benchmark_name} (rep {i+1}/{repetitions}) on {model_name}...")
            result = benchmark.run_benchmark(
                model_name=model_name,
                dataset_name=dataset_name,
                **kwargs,
            )
            results.append(result)
            self.results.append(result)

            status = "✅ PASSED" if result.overall_passed() else "❌ FAILED"
            print(f"  {status} in {result.execution_time_seconds:.2f}s")

        return results

    def run_all(
        self,
        model_name: str,
        dataset_configs: Optional[Dict[str, str]] = None,
        repetitions: int = 1,
    ) -> Dict[str, List[ReproductionResult]]:
        """Run all registered benchmarks."""
        dataset_configs = dataset_configs or {}
        all_results = {}

        for name, benchmark in self.benchmarks.items():
            dataset = dataset_configs.get(name, "")
            results = self.run_benchmark(
                name, model_name, dataset, repetitions
            )
            all_results[name] = results

        return all_results

    def generate_report(
        self,
        benchmark_name: str,
        format: str = "both",
    ) -> ReproductionReport:
        """Generate a comprehensive report for a benchmark."""
        benchmark_results = [r for r in self.results if r.benchmark_name == benchmark_name]

        if not benchmark_results:
            raise ValueError(f"No results for benchmark '{benchmark_name}'")

        benchmark = self.benchmarks[benchmark_name]
        report = ReproductionReport(
            benchmark_name=benchmark_name,
            paper_metadata=benchmark.paper_metadata,
            results=benchmark_results,
        )

        # Save reports
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = self.output_dir / f"{benchmark_name}_{timestamp}"

        if format in ("json", "both"):
            report.to_json(f"{base_path}.json")
        if format in ("markdown", "both"):
            report.to_markdown(f"{base_path}.md")

        print(f"Report saved to {base_path}.json and {base_path}.md")
        return report

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a summary across all benchmarks."""
        summary = {
            "total_benchmarks": len(self.benchmarks),
            "total_runs": len(self.results),
            "benchmarks": {},
        }

        for name, benchmark in self.benchmarks.items():
            benchmark_results = [r for r in self.results if r.benchmark_name == name]
            if not benchmark_results:
                continue

            successful = sum(1 for r in benchmark_results if r.success)
            total_metrics = sum(len(r.metric_comparisons) for r in benchmark_results)
            passed_metrics = sum(
                1 for r in benchmark_results for c in r.metric_comparisons if c.passed
            )

            summary["benchmarks"][name] = {
                "paper": benchmark.paper_metadata.title,
                "runs": len(benchmark_results),
                "successful_runs": successful,
                "total_metrics": total_metrics,
                "passed_metrics": passed_metrics,
                "pass_rate": passed_metrics / total_metrics if total_metrics > 0 else 0,
            }

        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.output_dir / f"summary_{timestamp}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    def print_summary(self) -> None:
        """Print a summary of all results to console."""
        print("\n" + "=" * 70)
        print("REPRODUCTION SUMMARY")
        print("=" * 70)

        for name, benchmark in self.benchmarks.items():
            benchmark_results = [r for r in self.results if r.benchmark_name == name]
            if not benchmark_results:
                print(f"\n{name}: No runs")
                continue

            successful = sum(1 for r in benchmark_results if r.success)
            total_metrics = sum(len(r.metric_comparisons) for r in benchmark_results)
            passed_metrics = sum(
                1 for r in benchmark_results for c in r.metric_comparisons if c.passed
            )

            print(f"\n📊 {name}")
            print(f"   Paper: {benchmark.paper_metadata.title}")
            print(f"   Runs: {successful}/{len(benchmark_results)} successful")
            print(f"   Metrics: {passed_metrics}/{total_metrics} passed")

            for result in benchmark_results:
                status = "✅" if result.overall_passed() else "❌"
                print(f"   {status} {result.model_used} ({result.execution_time_seconds:.1f}s)")

        print("\n" + "=" * 70)


def create_default_runner() -> ReproductionRunner:
    """Create a runner with all standard benchmarks registered."""
    from .papers import (
        IOIBenchmark,
        InductionHeadsBenchmark,
        GreaterThanBenchmark,
        ModularArithmeticBenchmark,
        SAEBenchmark,
        LogitLensBenchmark,
        ActivationPatchingBenchmark,
    )

    runner = ReproductionRunner()

    runner.register_benchmark(IOIBenchmark())
    runner.register_benchmark(InductionHeadsBenchmark())
    runner.register_benchmark(GreaterThanBenchmark())
    runner.register_benchmark(ModularArithmeticBenchmark())
    runner.register_benchmark(SAEBenchmark())
    runner.register_benchmark(LogitLensBenchmark())
    runner.register_benchmark(ActivationPatchingBenchmark())

    return runner