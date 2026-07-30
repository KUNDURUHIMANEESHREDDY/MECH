"""Benchmark Runner — Executes all 8 tasks across all available model families.

Orchestrates: model probing → task execution → comparison → KG population.
"""

from __future__ import annotations

import datetime
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .model_registry import ModelFamily, ModelRegistry, ModelAvailability
from .benchmark_tasks import (
    BenchmarkTask,
    BenchmarkTaskSpec,
    BenchmarkResult,
    BenchmarkTaskExecutor,
    ExecutionMode,
    TASK_CATALOGUE,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelBenchmarkSuite:
    """All task results for a single model."""
    model_id: str
    backend: str
    mode: ExecutionMode
    n_layers: int
    n_params_b: float
    task_results: List[BenchmarkResult] = field(default_factory=list)
    total_runtime_s: float = 0.0
    coverage_pct: float = 0.0       # % of 8 tasks completed
    mean_fidelity_pct: float = 0.0
    run_id: str = ""
    generated_at: str = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z"
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "backend": self.backend,
            "mode": self.mode.value,
            "n_layers": self.n_layers,
            "n_params_b": self.n_params_b,
            "coverage_pct": round(self.coverage_pct, 1),
            "mean_fidelity_pct": round(self.mean_fidelity_pct, 2),
            "total_runtime_s": round(self.total_runtime_s, 3),
            "run_id": self.run_id,
            "task_results": [r.to_dict() for r in self.task_results],
            "generated_at": self.generated_at,
        }


@dataclass
class BenchmarkReport:
    """Master report across all models and tasks."""
    suites: List[ModelBenchmarkSuite] = field(default_factory=list)
    overall_coverage_pct: float = 0.0
    overall_fidelity_pct: float = 0.0
    models_tested: int = 0
    tasks_tested: int = 0
    generated_at: str = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z"
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_coverage_pct": round(self.overall_coverage_pct, 1),
            "overall_fidelity_pct": round(self.overall_fidelity_pct, 2),
            "models_tested": self.models_tested,
            "tasks_tested": self.tasks_tested,
            "generated_at": self.generated_at,
            "suites": [s.to_dict() for s in self.suites],
        }

    # Convenience
    def get_suite(self, model_id: str) -> Optional[ModelBenchmarkSuite]:
        for s in self.suites:
            if s.model_id == model_id:
                return s
        return None

    def coverage_matrix(self) -> Dict[str, Dict[str, Optional[float]]]:
        """Returns {model_id: {task_id: fidelity_pct}} for UI rendering."""
        matrix: Dict[str, Dict[str, Optional[float]]] = {}
        for suite in self.suites:
            row: Dict[str, Optional[float]] = {t.value: None for t in BenchmarkTask}
            for r in suite.task_results:
                row[r.task_id.value] = r.fidelity_pct
            matrix[suite.model_id] = row
        return matrix


class BenchmarkRunner:
    """
    Orchestrates the full benchmark suite across all model families.

    Usage
    -----
    >>> runner = BenchmarkRunner()
    >>> report = runner.run_full_suite()
    """

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        executor: Optional[BenchmarkTaskExecutor] = None,
        tasks: Optional[List[BenchmarkTask]] = None,
        families: Optional[List[ModelFamily]] = None,
    ) -> None:
        self._registry = registry or ModelRegistry()
        self._executor = executor or BenchmarkTaskExecutor()
        self._tasks = tasks or list(BenchmarkTask)
        self._families = families or list(ModelFamily)

    def run_full_suite(
        self,
        mode: ExecutionMode = ExecutionMode.MOCK,
        tier: int = 1,
        n_samples_override: Optional[int] = None,
    ) -> BenchmarkReport:
        """Run all tasks on all available models and return a BenchmarkReport."""
        availability = self._registry.probe_all()
        suites: List[ModelBenchmarkSuite] = []

        # Tiered Logic
        if tier == 1:
            target_families = [ModelFamily.GPT2_SMALL]
        else:
            target_families = self._families

        run_id = f"run_{int(time.time())}"

        for family in target_families:
            avail: ModelAvailability = availability[family]
            spec = avail.spec
            logger.info(
                "Running %d tasks on %s [backend=%s, mode=%s]",
                len(self._tasks), spec.model_id, avail.backend, mode
            )
            suite = self._run_model(spec, avail.backend, mode, n_samples_override, run_id)
            suites.append(suite)

        # Aggregate
        all_fidelities = [
            r.fidelity_pct
            for s in suites
            for r in s.task_results
        ]
        coverage_vals = [s.coverage_pct for s in suites]

        return BenchmarkReport(
            suites=suites,
            overall_coverage_pct=sum(coverage_vals) / max(len(coverage_vals), 1),
            overall_fidelity_pct=sum(all_fidelities) / max(len(all_fidelities), 1),
            models_tested=len(suites),
            tasks_tested=len(self._tasks),
        )

    def generate_artifact_package(self, report: BenchmarkReport, output_dir: str = "benchmark_report") -> str:
        """Generates JSON, CSV, and MD artifacts in a dedicated directory."""
        import os
        import json
        import csv

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1. JSON
        json_path = os.path.join(output_dir, "report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        # 2. CSV
        csv_path = os.path.join(output_dir, "report.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["model_id", "task_id", "fidelity_pct", "primary_score", "reference_score", "runtime_s", "peak_memory_mb", "tokens_sec", "flops", "gpu_util", "latency_ms"])
            for suite in report.suites:
                for r in suite.task_results:
                    writer.writerow([
                        r.model_id, r.task_id.value, r.fidelity_pct, r.primary_score, r.reference_score,
                        r.runtime_s, r.peak_memory_mb, r.tokens_per_sec, r.flops, r.gpu_util_pct, r.mean_latency_ms
                    ])

        # 3. Markdown
        md_path = os.path.join(output_dir, "report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Benchmark Report — {report.generated_at}\n\n")
            f.write(f"- **Overall Coverage**: {report.overall_coverage_pct:.1f}%\n")
            f.write(f"- **Overall Fidelity**: {report.overall_fidelity_pct:.2f}%\n")
            f.write(f"- **Models Tested**: {report.models_tested}\n\n")

            f.write("## Performance Dashboard\n")
            f.write("| Model | Throughput (TPS) | Mean Latency (ms) | Peak VRAM (MB) | GPU Util % | FLOPs (Est) |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for suite in report.suites:
                avg_tps = sum(r.tokens_per_sec for r in suite.task_results) / len(suite.task_results)
                avg_lat = sum(r.mean_latency_ms for r in suite.task_results) / len(suite.task_results)
                max_vram = max(r.peak_vram_mb for r in suite.task_results)
                avg_gpu = sum(r.gpu_util_pct for r in suite.task_results) / len(suite.task_results)
                total_flops = sum(r.flops for r in suite.task_results)
                f.write(f"| {suite.model_id} | {avg_tps:.1f} | {avg_lat:.2f} | {max_vram:.1f} | {avg_gpu:.1f}% | {total_flops:.2e} |\n")
            f.write("\n")

            for suite in report.suites:
                f.write(f"## Model: {suite.model_id} ({suite.backend})\n")
                f.write(f"- **Execution Mode**: {suite.mode}\n")
                f.write("| Task | Fidelity % | Published | Observed | 95% CI | Status |\n")
                f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
                for r in suite.task_results:
                    status = "✅ PASS" if r.fidelity_pct > 90 else "⚠️ WARN"
                    f.write(f"| {r.task_id.value} | {r.fidelity_pct}% | {r.reference_score} | {r.primary_score} | [{r.confidence_interval_low}, {r.confidence_interval_high}] | {status} |\n")
                f.write("\n")

        # 4. Raw Experiment Data
        raw_path = os.path.join(output_dir, "raw_experiment_data.json")
        raw_data = []
        for suite in report.suites:
            for r in suite.task_results:
                # Store task-specific raw traces if they exist
                raw_data.append({
                    "task": r.task_id.value,
                    "model": r.model_id,
                    "patch_success_rate": r.patch_success_rate,
                    "overlap_pct": r.published_overlap_pct,
                    "notes": r.notes
                })

        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, indent=2)

        return output_dir

    def run_single_task(
        self,
        task: BenchmarkTask,
        family: ModelFamily,
        mode: ExecutionMode = ExecutionMode.MOCK,
        n_samples: Optional[int] = None,
    ) -> BenchmarkResult:
        """Run a single task on a single model — useful for targeted re-runs."""
        avail = self._registry.probe(family)
        spec_task = TASK_CATALOGUE[task]
        return self._executor.execute(
            spec_task, avail.spec.model_id, avail.backend, mode, n_samples
        )

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _run_model(
        self,
        spec,
        backend: str,
        mode: ExecutionMode,
        n_samples_override: Optional[int],
        run_id: str,
    ) -> ModelBenchmarkSuite:
        results: List[BenchmarkResult] = []
        t0 = time.perf_counter()

        for task_id in self._tasks:
            task_spec: BenchmarkTaskSpec = TASK_CATALOGUE[task_id]
            # Skip SAE task for models without SAE support
            if task_id == BenchmarkTask.SAE and not spec.supports_sae:
                logger.debug("Skipping SAE task for %s (not supported)", spec.model_id)
                continue
            try:
                result = self._executor.execute(
                    task_spec, spec.model_id, backend, mode, n_samples_override, run_id
                )
                results.append(result)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Task %s failed on %s: %s", task_id.value, spec.model_id, exc
                )

        total_rt = time.perf_counter() - t0
        fidelities = [r.fidelity_pct for r in results]
        coverage = len(results) / len(self._tasks) * 100.0

        return ModelBenchmarkSuite(
            model_id=spec.model_id,
            backend=backend,
            mode=mode,
            n_layers=spec.n_layers,
            n_params_b=spec.n_params_b,
            task_results=results,
            total_runtime_s=round(total_rt, 3),
            coverage_pct=round(coverage, 1),
            mean_fidelity_pct=round(
                sum(fidelities) / max(len(fidelities), 1), 2
            ),
            run_id=run_id,
        )
