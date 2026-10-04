"""Golden Benchmark Validation Scheduler.

Schedules and executes continuous validation suites across golden research benchmarks:
IOI, Induction Heads, Greater-Than, Copy Task, Arithmetic, Factual Recall, Logit Lens, SAE.
"""

from __future__ import annotations

import datetime as _dt
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# The status vocabulary is imported, not re-spelled. This module used to
# declare its own in a field comment -- `# PASS, REGRESSION, WARNING,
# NOT_RUN` -- which named a status it never produced (WARNING) and omitted
# one it does produce (MEASURED).
from .benchmark_status import MEASURED, NOT_RUN, PASS, REGRESSION


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
    # The real vocabulary is NOT_RUN / MEASURED / PASS / REGRESSION.
    #
    # This comment read `# PASS, REGRESSION, WARNING, NOT_RUN`, which was wrong
    # twice over: WARNING was never produced by any code path, and MEASURED --
    # the state this scheduler actually assigns to a measured-but-incomparable
    # benchmark, at the line below -- was not listed. A reader trusting the
    # contract would not have known MEASURED existed.
    #
    # Defined once in backend/validation/benchmark_status.py rather than restated
    # per field, because this is the fourth place in the repository that had its
    # own idea of what a benchmark status means.
    status: str
    executed_at: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")
    measured: bool = False
    reason: Optional[str] = None
    # Set when the published baseline is not the same measurement, which is the
    # case for IOI: running the published circuit through this repo's harness
    # yields a lower figure than the published constant.
    #
    # `None` means *not established*, and `benchmark_status.is_comparable` treats
    # that as NOT comparable. Fail-closed on purpose: an unestablished comparison
    # is not a comparison.
    baseline_is_comparable: Optional[bool] = None
    # The same question asked separately of wall-clock timing.
    #
    # This needed its own flag rather than reusing `baseline_is_comparable`,
    # because the two comparability questions have different answers. Fidelity
    # baselines here are published *metric* values that this harness may or may
    # not compute the same way. Runtime baselines are hardware-specific figures
    # from papers: 1200 ms for IOI, measured here at 77896 ms on an RTX 3050. No
    # amount of fidelity comparability makes those two numbers comparable, and
    # scoring the difference produced a "+6391% latency regression" against a
    # different machine. Comparing local timings belongs to RegressionSuite,
    # which compares against a golden record captured on this hardware.
    runtime_is_comparable: Optional[bool] = None
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
            "runtime_is_comparable": self.runtime_is_comparable,
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
        # Greater-than now has a real pipeline that measures whether the model
        # performs the comparison at all before attempting to localise a
        # circuit, and raises with the measured evidence if it does not. So this
        # reports NOT_RUN *with a reason* instead of the vaguer "no pipeline
        # exists" the arithmetic and SAE entries still get.
        #
        # target_model was "GPT2-M". The pipeline measures gpt2-small, so the
        # declaration did not describe this system's measurement. Corrected to
        # match; the published baseline is retained as reference data and is
        # marked incomparable rather than silently rescored.
        GoldenBenchmarkTask("bm_gt", "Greater-Than Comparative Circuit", "GPT2-S", 0.860, 1800.0, 5.4,
                            pipeline="greater_than"),
        # No pipeline exists for these two. Each previously produced
        # `baseline * 0.995` and was scored PASS against a `baseline * 0.95`
        # threshold, so the suite reported a 100% pass rate without running
        # anything. Both now have pipeline modules that exist but raise: the
        # arithmetic one returned hardcoded accuracies of 0.45 and 0.85 to every
        # caller, and the SAE one drew its whole feature bank from a seeded RNG
        # while claiming OpenWebText in its dataset manifest.
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

    def _run_one(self, bm: GoldenBenchmarkTask,
                 n_seeds: int = 1) -> GoldenBenchmarkResult:
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
                status=NOT_RUN,
                measured=False,
                reason=(
                    "No measurement pipeline is implemented for this "
                    "benchmark. It previously reported baseline * 0.995 and "
                    "scored PASS."
                ),
            )

        try:
            measured = self._measure(bm.pipeline, n_seeds=n_seeds)
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
                status=NOT_RUN,
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
                status=NOT_RUN,
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
            status = PASS if fidelity >= threshold else REGRESSION
        else:
            status = MEASURED

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
            # Carried through from `_measure` rather than derived here, so the
            # pipeline that knows where the baseline came from is the one that
            # says whether it is comparable. Default is `None`, which
            # `benchmark_status.is_comparable` treats as not comparable.
            runtime_is_comparable=measured.get("runtime_is_comparable"),
            reference_circuit_fidelity_same_harness=measured.get(
                "reference_same_harness"),
        )

    def _measure(self, pipeline: str, n_seeds: int = 1) -> Dict[str, Any]:
        """Run a real pipeline and return what it measured.

        Every return includes `runtime_is_comparable`. These published runtime
        baselines were measured on the authors' hardware -- 1200 ms for IOI,
        measured here at 77896 ms on an RTX 3050 -- so none of them is comparable
        to a local run and each says so explicitly rather than leaving the field
        unset and hoping a consumer defaults sensibly.

        `n_seeds` is how many independent draws to pool. It defaults to 1, and
        every return states the seed count it actually used. That is the point:
        a single 100-prompt draw is still one draw, and the difference between
        one draw and five has to be visible in the data rather than implied by
        `n_samples`. Only IOI supports pooling today; the other pipelines are
        single-draw and say so.
        """

        runtime_not_comparable = {
            "runtime_is_comparable": False,
            "runtime_comparability_reason": (
                "The published runtime baseline was measured on unspecified "
                "hardware. A local wall-clock figure is not comparable to it. "
                "Local timing regression belongs to RegressionSuite, which "
                "compares against a golden record captured on this machine."
            ),
        }
        import time as _time

        started = _time.perf_counter()

        if pipeline == "ioi":
            from backend.science.reproducibility.ioi_pipeline import (
                IOIReproductionPipeline,
            )

            if n_seeds > 1:
                audit = IOIReproductionPipeline(
                    mock_mode=False).run_stability_audit(n_seeds=n_seeds)
                pooled = audit.get("pooled") or {}
                return {
                    "fidelity": pooled.get("mean"),
                    "runtime_ms": round((_time.perf_counter() - started) * 1000, 2),
                    "baseline_is_comparable": False,
                    **runtime_not_comparable,
                    "reference_same_harness": None,
                    "correct": None,
                    "trials": None,
                    "ci_target": "circuit_faithfulness pooled across seeds",
                    "n_seeds": pooled.get("n_seeds_used"),
                    "seeds": pooled.get("seeds"),
                    "per_seed_values": pooled.get("values"),
                    "reason": (
                        f"Pooled across {pooled.get('n_seeds_used')} seeds on "
                        f"live weights. {pooled.get('method')}. "
                        + (pooled.get("reason") or "")
                    ).strip(),
                }

            result = IOIReproductionPipeline(mock_mode=False).run(n_prompts=10)
            if result.get("status") == "unavailable":
                return {"fidelity": None, "n_seeds": 0,
                        "reason": result.get("reason")}
            metrics = result["observed_metrics"]
            reference = metrics.get("reference_circuit_faithfulness_same_harness")
            return {
                "fidelity": metrics.get("circuit_faithfulness"),
                "runtime_ms": round((_time.perf_counter() - started) * 1000, 2),
                # The published 0.88 is not what this harness computes for the
                # published circuit, so no PASS/REGRESSION verdict is issued.
                "baseline_is_comparable": False,
                **runtime_not_comparable,
                "reference_same_harness": reference,
                # One draw. Stated rather than left implicit, because
                # `n_samples=10` on its own reads like a settled figure.
                "n_seeds": 1,
                "seeds": [IOIReproductionPipeline.SEED_BASE],
                "per_seed_values": [metrics.get("circuit_faithfulness")],
                "reason": (
                    "Measured on live weights from a single seed. "
                    "The Wilson interval bounds precision within that draw and "
                    "says nothing about seed-to-seed variation; pass "
                    "n_seeds>1 to pool. The published baseline is not the same "
                    "measurement; the published circuit scores "
                    f"{reference} through this harness."
                ) if reference is not None else (
                    "Measured on live weights from a single seed, so no "
                    "seed-level interval is derivable."
                ),
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
                **runtime_not_comparable,
                "reference_same_harness": None,
                # Single-draw, and fixed at seed 42. Pooling is not implemented
                # for this pipeline, so the seed count is stated rather than
                # left to be inferred from n_sequences.
                "n_seeds": 1,
                "seeds": [42],
                "per_seed_values": [metrics.get("induction_score")],
                "reason": (
                    "Measured on live weights from a single fixed seed. The "
                    "Wilson interval bounds precision within that draw; "
                    "seed-to-seed variation is not measured for this pipeline. "
                    "The registry baseline is a different quantity (mean "
                    "attention to previous-token copies under a different "
                    "procedure)."
                ),
            }

        if pipeline == "greater_than":
            from backend.science.reproducibility.greater_than_pipeline import (
                GreaterThanCircuitPipeline,
            )
            # Raises LiveUnavailable when the model does not perform the
            # comparison, carrying the measured evidence. The caller's except
            # path turns that into NOT_RUN with the reason intact.
            result = GreaterThanCircuitPipeline(mock_mode=False).run()
            metrics = result["observed_metrics"]
            return {
                "fidelity": metrics.get("mlp_importance_score"),
                "runtime_ms": round((_time.perf_counter() - started) * 1000, 2),
                # The published baseline was obtained on a different model and
                # under a setup this harness does not reproduce, so no verdict is
                # issued against it.
                "baseline_is_comparable": False,
                **runtime_not_comparable,
                "reference_same_harness": None,
                "reason": (
                    "Measured on live weights. The dominant MLP layer was "
                    f"derived from the measurement (layer "
                    f"{metrics.get('dominant_layer')}); Hanna et al. report "
                    "mid-layer concentration. The registry baseline comes from a "
                    "different model and setup, so it is not rescored."
                ),
                # Single draw: this pipeline takes no seed argument and has no
                # pooling path, so seed-to-seed variation is not measured.
                "n_seeds": 1,
                "seeds": [],
                "per_seed_values": [metrics.get("mlp_importance_score")],
            }

        raise NotImplementedError(f"No pipeline named {pipeline!r}")
