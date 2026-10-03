"""Golden Benchmark Validation Scheduler.

Schedules and executes continuous validation suites across golden research benchmarks:
IOI, Induction Heads, Greater-Than, Copy Task, Arithmetic, Factual Recall, Logit Lens, SAE.
"""

from __future__ import annotations

import datetime as _dt
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GoldenBenchmarkTask:
    """Golden benchmark validation task specification.

    The three `published_baseline_*` fields are external reference constants.
    They are reference data, not this system's results, and are labelled as
    such wherever they surface.
    """
    benchmark_id: str
    name: str
    target_model: str
    published_baseline_fidelity: float
    published_baseline_runtime_ms: float
    published_baseline_vram_gb: float
    # Which reproduction pipeline measures this benchmark, if one exists.
    # None means there is no implementation, and the run reports unavailable
    # rather than a number derived from the baseline.
    pipeline: Optional[str] = None


@dataclass
class GoldenBenchmarkResult:
    """Execution result of a golden benchmark validation run.

    `current_*` fields are Optional. They are None when the benchmark was not
    executed -- which is not the same as executing it and getting the baseline
    back.
    """
    benchmark_id: str
    name: str
    target_model: str
    published_baseline_fidelity: float
    current_fidelity: Optional[float]
    published_baseline_runtime_ms: float
    current_runtime_ms: Optional[float]
    published_baseline_vram_gb: float
    current_vram_gb: Optional[float]
    status: str  # PASS, REGRESSION, WARNING, NOT_RUN
    executed_at: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")
    measured: bool = False
    reason: Optional[str] = None
    # Set when the published baseline is not the same measurement, which is the
    # case for IOI: running the published circuit through this repo's harness
    # yields a lower figure than the published constant.
    baseline_is_comparable: Optional[bool] = None
    reference_circuit_fidelity_same_harness: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "name": self.name,
            "target_model": self.target_model,
            "published_baseline_fidelity": self.published_baseline_fidelity,
            "current_fidelity": self.current_fidelity,
            "published_baseline_runtime_ms": self.published_baseline_runtime_ms,
            "current_runtime_ms": self.current_runtime_ms,
            "published_baseline_vram_gb": self.published_baseline_vram_gb,
            "current_vram_gb": self.current_vram_gb,
            "status": self.status,
            "measured": self.measured,
            "reason": self.reason,
            "baseline_is_comparable": self.baseline_is_comparable,
            "reference_circuit_fidelity_same_harness":
                self.reference_circuit_fidelity_same_harness,
            "provenance": "live" if self.measured else "unavailable",
            "validation_eligible": self.measured,
            "publication_eligible": False,
            "executed_at": self.executed_at,
        }

class ValidationBenchmarkScheduler:
    """Schedules and executes golden benchmark validation suites."""

    DEFAULT_BENCHMARKS = [
        GoldenBenchmarkTask("bm_ioi", "IOI Circuit Recovery", "GPT2-S", 0.880, 1200.0, 4.2,
                            pipeline="ioi"),
        GoldenBenchmarkTask("bm_ind", "Induction Head Sequence Repeater", "GPT2-S", 0.940, 2100.0, 6.8,
                            pipeline="induction_heads"),
        # No pipeline exists for these three, so they report NOT_RUN. Each
        # previously produced `baseline * 0.995` and was scored PASS against a
        # `baseline * 0.95` threshold, so the suite reported a 100% pass rate
        # without running anything.
        GoldenBenchmarkTask("bm_gt", "Greater-Than Comparative Circuit", "GPT2-M", 0.860, 1800.0, 5.4),
        GoldenBenchmarkTask("bm_arith", "Multi-Digit Arithmetic Circuit", "Llama3-8B", 0.910, 3400.0, 14.2),
        GoldenBenchmarkTask("bm_sae", "SAE Feature Dictionary Recovery", "GPT2-S", 0.895, 1500.0, 4.8),
    ]

    def execute_validation_suite(self) -> List[GoldenBenchmarkResult]:
        """Execute the golden benchmark suite, measuring where that is possible.

        This ran no benchmark. It computed ``current_fid = baseline * 0.995``,
        ``current_rt = baseline * 1.01``, ``current_vram = baseline * 1.0`` and
        then scored PASS against a ``baseline * 0.95`` threshold. Since 0.995
        always exceeds 0.95, every benchmark passed on every run, and
        `health_dashboard.run_continuous_validation` turned that into a
        `pass_rate: 100.0` node written into the knowledge graph as a real
        experiment.

        Now each benchmark with an implemented pipeline is executed and scored
        on what it measured. The rest report NOT_RUN with the reason. A suite
        that cannot measure three of five benchmarks must not claim 100%.
        """
        results: List[GoldenBenchmarkResult] = []

        for bm in self.DEFAULT_BENCHMARKS:
            results.append(self._run_one(bm))

        return results

    def _run_one(self, bm: GoldenBenchmarkTask) -> GoldenBenchmarkResult:
        if bm.pipeline is None:
            return GoldenBenchmarkResult(
                benchmark_id=bm.benchmark_id,
                name=bm.name,
                target_model=bm.target_model,
                published_baseline_fidelity=bm.published_baseline_fidelity,
                current_fidelity=None,
                published_baseline_runtime_ms=bm.published_baseline_runtime_ms,
                current_runtime_ms=None,
                published_baseline_vram_gb=bm.published_baseline_vram_gb,
                current_vram_gb=None,
                status="NOT_RUN",
                measured=False,
                reason=(
                    "No measurement pipeline is implemented for this "
                    "benchmark. It previously reported baseline * 0.995 and "
                    "scored PASS."
                ),
            )

        try:
            measured = self._measure(bm.pipeline)
        except Exception as exc:  # noqa: BLE001
            return GoldenBenchmarkResult(
                benchmark_id=bm.benchmark_id,
                name=bm.name,
                target_model=bm.target_model,
                published_baseline_fidelity=bm.published_baseline_fidelity,
                current_fidelity=None,
                published_baseline_runtime_ms=bm.published_baseline_runtime_ms,
                current_runtime_ms=None,
                published_baseline_vram_gb=bm.published_baseline_vram_gb,
                current_vram_gb=None,
                status="NOT_RUN",
                measured=False,
                reason=f"Pipeline raised: {type(exc).__name__}: {exc}",
            )

        fidelity = measured.get("fidelity")
        if fidelity is None:
            return GoldenBenchmarkResult(
                benchmark_id=bm.benchmark_id,
                name=bm.name,
                target_model=bm.target_model,
                published_baseline_fidelity=bm.published_baseline_fidelity,
                current_fidelity=None,
                published_baseline_runtime_ms=bm.published_baseline_runtime_ms,
                current_runtime_ms=measured.get("runtime_ms"),
                published_baseline_vram_gb=bm.published_baseline_vram_gb,
                current_vram_gb=None,
                status="NOT_RUN",
                measured=False,
                reason=measured.get("reason", "No fidelity was produced."),
                baseline_is_comparable=measured.get("baseline_is_comparable"),
                reference_circuit_fidelity_same_harness=measured.get(
                    "reference_same_harness"),
            )

        baseline_comparable = measured.get("baseline_is_comparable", True)
        # Scored against the published baseline only where that baseline is the
        # same measurement. For IOI it is not, so a verdict would be spurious
        # and the run is reported as measured-but-incomparable.
        if baseline_comparable:
            threshold = bm.published_baseline_fidelity * 0.95
            status = "PASS" if fidelity >= threshold else "REGRESSION"
        else:
            status = "MEASURED"

        return GoldenBenchmarkResult(
            benchmark_id=bm.benchmark_id,
            name=bm.name,
            target_model=bm.target_model,
            published_baseline_fidelity=bm.published_baseline_fidelity,
            current_fidelity=round(fidelity, 4),
            published_baseline_runtime_ms=bm.published_baseline_runtime_ms,
            current_runtime_ms=measured.get("runtime_ms"),
            published_baseline_vram_gb=bm.published_baseline_vram_gb,
            current_vram_gb=measured.get("vram_gb"),
            status=status,
            measured=True,
            reason=measured.get("reason"),
            baseline_is_comparable=baseline_comparable,
            reference_circuit_fidelity_same_harness=measured.get(
                "reference_same_harness"),
        )

    def _measure(self, pipeline: str) -> Dict[str, Any]:
        """Run a real pipeline and return what it measured."""
        import time as _time

        started = _time.perf_counter()

        if pipeline == "ioi":
            from backend.science.reproducibility.ioi_pipeline import (
                IOIReproductionPipeline,
            )
            result = IOIReproductionPipeline(mock_mode=False).run(n_prompts=10)
            if result.get("status") == "unavailable":
                return {"fidelity": None, "reason": result.get("reason")}
            metrics = result["observed_metrics"]
            reference = metrics.get("reference_circuit_faithfulness_same_harness")
            return {
                "fidelity": metrics.get("circuit_faithfulness"),
                "runtime_ms": round((_time.perf_counter() - started) * 1000, 2),
                # The published 0.88 is not what this harness computes for the
                # published circuit, so no PASS/REGRESSION verdict is issued.
                "baseline_is_comparable": False,
                "reference_same_harness": reference,
                "reason": (
                    "Measured on live weights. The published baseline is not "
                    "the same measurement; the published circuit scores "
                    f"{reference} through this harness."
                ) if reference is not None else None,
            }

        if pipeline == "induction_heads":
            from backend.science.reproducibility.induction_heads_pipeline import (
                InductionHeadsPipeline,
            )
            result = InductionHeadsPipeline(mock_mode=False).run(
                n_sequences=10, seq_len=8, seed=42)
            metrics = result["observed_metrics"]
            return {
                "fidelity": metrics.get("induction_score"),
                "runtime_ms": round((_time.perf_counter() - started) * 1000, 2),
                "baseline_is_comparable": False,
                "reference_same_harness": None,
                "reason": (
                    "Measured on live weights. The registry baseline is a "
                    "different quantity (mean attention to previous-token "
                    "copies under a different procedure)."
                ),
            }

        raise NotImplementedError(f"No pipeline named {pipeline!r}")
