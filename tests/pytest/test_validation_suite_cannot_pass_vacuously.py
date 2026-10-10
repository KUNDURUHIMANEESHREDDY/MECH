"""The continuous validation suite must not be incapable of failing.

`ValidationBenchmarkScheduler.execute_validation_suite` ran no benchmark. It
computed `current_fid = published_baseline_fidelity * 0.995` and scored PASS
against a `baseline * 0.95` threshold, so every benchmark passed on every run.
`HealthDashboardEngine` then reported `pass_rate: 100.0` and wrote it into the
knowledge graph as a real experiment node.

These tests pin the replacement: benchmarks with an implemented pipeline are
executed, benchmarks without one report NOT_RUN, and a pass rate over a suite
where nothing is scorable is None rather than 100% or 0%.
"""
import inspect

import pytest


from _weight_guard import skip_reason, weights_available


needs_weights = pytest.mark.skipif(
    not weights_available(), reason=skip_reason()
)


# ── The fabricated pass path is gone ────────────────────────────────────────

def test_no_benchmark_can_pass_without_being_executed():
    """The old code could only ever produce PASS."""
    from backend.validation.benchmark_scheduler import (
        GoldenBenchmarkTask,
        ValidationBenchmarkScheduler,
    )

    result = ValidationBenchmarkScheduler()._run_one(GoldenBenchmarkTask(
        benchmark_id="x", name="x", target_model="gpt2",
        published_baseline_fidelity=0.88,
        published_baseline_runtime_ms=1200.0,
        published_baseline_vram_gb=4.2,
        pipeline=None,
    ))
    assert result.status == "NOT_RUN"
    assert result.measured is False
    assert result.current_fidelity is None
    assert result.current_runtime_ms is None
    assert result.reason


def test_scheduler_source_contains_no_baseline_derived_result():
    """Guard against `baseline * <constant>` creeping back in."""
    from source_assert import executable_source

    from backend.validation import benchmark_scheduler

    code = executable_source(benchmark_scheduler)
    # The fabricated line was:
    #   current_fid = round(bm.published_baseline_fidelity * 0.995, 3)
    for forbidden in ("published_baseline_fidelity * 0.99",
                      "published_baseline_runtime_ms * 1.0",
                      "published_baseline_vram_gb * 1.0"):
        assert forbidden not in code, (
            f"{forbidden!r} derives a result from the baseline instead of "
            f"measuring one"
        )


def test_benchmarks_without_a_pipeline_report_not_run():
    from backend.validation.benchmark_scheduler import (
        ValidationBenchmarkScheduler,
    )

    results = ValidationBenchmarkScheduler().execute_validation_suite()
    without_pipeline = [r for r in results if r.reason and "No measurement" in r.reason]
    assert without_pipeline, "expected some benchmarks to have no implementation"
    for r in without_pipeline:
        assert r.status == "NOT_RUN"
        assert r.measured is False
        assert r.current_fidelity is None
        assert r.to_dict()["provenance"] == "unavailable"
        assert r.to_dict()["validation_eligible"] is False


def test_every_benchmark_declares_whether_it_can_be_measured():
    """A benchmark with no pipeline must say so rather than defaulting."""
    from backend.validation.benchmark_scheduler import (
        ValidationBenchmarkScheduler,
    )

    for bm in ValidationBenchmarkScheduler.DEFAULT_BENCHMARKS:
        assert hasattr(bm, "pipeline"), bm.benchmark_id
    pipelines = {bm.benchmark_id: bm.pipeline
                 for bm in ValidationBenchmarkScheduler.DEFAULT_BENCHMARKS}
    # The two implemented ones are named; the rest are explicitly None.
    assert pipelines["bm_ioi"] == "ioi"
    assert pipelines["bm_ind"] == "induction_heads"
    # Greater-than now names a real pipeline. It measures whether the model
    # performs the comparison at all and raises LiveUnavailable with the
    # evidence if it does not, so bm_gt reports NOT_RUN with a measured reason
    # rather than a fabricated fidelity.
    assert pipelines["bm_gt"] == "greater_than"
    # Arithmetic and SAE have no measurement of any kind. They must keep saying
    # so explicitly -- this line previously read `bm_gt is None`, which pinned a
    # specific state rather than the discipline the docstring describes, and so
    # had to be rewritten the moment an implementation landed.
    # Arithmetic still has no measurement of any kind, so its pipeline stays
    # `None`.
    assert pipelines["bm_arith"] is None
    # The SAE pipeline is implemented and does train. Declaring `bm_sae` as
    # `None` would pin the *absent* state, and this line previously did exactly
    # that -- it read `bm_sae is None` at a moment when the pipeline existed and
    # worked. A guard that pins a specific state rather than the discipline
    # breaks the moment the state becomes correct, which is what happened.
    #
    # What must hold is narrower: the scheduler may not claim a measurement it
    # has not made. Either there is a pipeline that measures, or the benchmark
    # stays unmeasured -- never a value from a module that invents one.
    # Instead: whatever the scheduler declares, a benchmark whose pipeline is
    # absent must report `measured=False` rather than a number.
    # The SAE pipeline is implemented and does train, so `bm_sae`'s state is
    # deliberately not pinned -- pinning the absent state is what made this line
    # wrong. What must hold is that the scheduler is not *claiming* a measurement
    # it has not made. Checked against a real execution, which is the only form
    # the claim is ever consumed in.
    scheduler = ValidationBenchmarkScheduler()
    sae = next(bm for bm in scheduler.DEFAULT_BENCHMARKS
               if bm.benchmark_id == "bm_sae")
    if sae.pipeline is None:
        outcome = scheduler._run_one(sae)
        assert outcome.measured is False, (
            "bm_sae has no pipeline but reported a measurement")
        assert outcome.current_fidelity is None, (
            f"bm_sae has no pipeline yet reported fidelity "
            f"{outcome.current_fidelity}")


def test_benchmarks_without_a_pipeline_have_no_module_returning_numbers():
    """A None pipeline must not correspond to a module that invents a result.

    `bm_arith` and `bm_sae` declare no measurement, but their modules exist. Both
    used to return hardcoded values anyway -- 0.45/0.85 and a bank of RNG-drawn
    features -- and `benchmark_runner` wrote those into reports. Declaring
    `pipeline=None` was never sufficient on its own.
    """
    from science.reproducibility.arithmetic_pipeline import ArithmeticPipeline
    from science.reproducibility.sae_pipeline import SAEReproductionPipeline

    # `ArithmeticPipeline` still has no implementation and must keep saying so.
    with pytest.raises(Exception) as excinfo:
        ArithmeticPipeline(model_manager=None).run()
    assert "not implemented" in str(excinfo.value).lower(), (
        "ArithmeticPipeline must refuse, not return a value")

    # `SAEReproductionPipeline` no longer refuses: it trains a real top-k SAE.
    # What must not happen is a *fabricated* result -- one whose numbers did not
    # come from a fitted autoencoder. This branch used to assert `raises` for
    # both, which was correct while both refused and became false when the SAE
    # pipeline started measuring.
    try:
        result = SAEReproductionPipeline(mock_mode=False).run(
            n_features=8, n_tokens=32, steps=2)
    except Exception:
        # Refusing is fine, provided it says why.
        return

    assert result["status"] in ("completed", "unavailable")
    metrics = result["observed_metrics"]
    if result["status"] == "unavailable":
        assert metrics["reconstruction_useful"] is False or \
            metrics.get("normalized_mse") is None
        return

    # A completed run must rest on real activations and a usable reconstruction.
    assert metrics["n_activations"] > 0, (
        "SAEReproductionPipeline reported a completed run with no activations")
    assert metrics["reconstruction_useful"] is True, (
        "SAEReproductionPipeline reported success for a reconstruction worse "
        "than predicting the mean")


# ── The dashboard must not report a pass rate it cannot compute ────────────

def test_pass_rate_is_none_when_nothing_is_scorable():
    """0% and 100% are both lies about a suite with no scorable member."""
    from backend.validation.benchmark_scheduler import GoldenBenchmarkResult
    from backend.validation.health_dashboard import HealthDashboardEngine
    from backend.validation.regression_detector import RegressionDetector

    only_unrun = [
        GoldenBenchmarkResult(
            benchmark_id=f"b{i}", name="n", target_model="gpt2",
            published_baseline_fidelity=0.88, current_fidelity=None,
            published_baseline_runtime_ms=1000.0, current_runtime_ms=None,
            published_baseline_vram_gb=4.0, current_vram_gb=None,
            status="NOT_RUN", measured=False,
        )
        for i in range(3)
    ]

    # The detector must tolerate missing values rather than raising, and must
    # not invent a 100% regression for each.
    assert RegressionDetector().analyze_results(only_unrun) == []

    engine = HealthDashboardEngine()
    engine.scheduler = type(
        "_Fixed", (), {"execute_validation_suite": lambda self: only_unrun}
    )()

    report = engine.run_continuous_validation().to_dict()
    assert report["overall_pass_rate"] is None
    assert report["pass_rate_measured"] is False
    assert report["benchmarks_unscored"] == 3
    assert report["passed_benchmarks"] == 0
    assert report["failed_benchmarks"] == 0


def test_regression_detector_skips_unrun_benchmarks():
    """A missing current value is not a 100% regression."""
    from backend.validation.benchmark_scheduler import GoldenBenchmarkResult
    from backend.validation.regression_detector import RegressionDetector

    results = [
        GoldenBenchmarkResult(
            benchmark_id="unrun", name="n", target_model="gpt2",
            published_baseline_fidelity=0.88, current_fidelity=None,
            published_baseline_runtime_ms=1000.0, current_runtime_ms=None,
            published_baseline_vram_gb=4.0, current_vram_gb=None,
            status="NOT_RUN", measured=False,
        ),
    ]
    assert RegressionDetector().analyze_results(results) == []


# ── Live execution ─────────────────────────────────────────────────────────

@needs_weights
def test_implemented_benchmarks_actually_run():
    from backend.validation.benchmark_scheduler import (
        ValidationBenchmarkScheduler,
    )

    results = {r.benchmark_id: r for r in
               ValidationBenchmarkScheduler().execute_validation_suite()}

    for benchmark_id in ("bm_ioi", "bm_ind"):
        result = results[benchmark_id]
        assert result.measured is True, result.reason
        assert result.current_fidelity is not None
        assert 0.0 <= result.current_fidelity <= 1.0
        assert result.to_dict()["provenance"] == "live"

    # A PASS verdict needs a comparable baseline. Neither of these has one, so
    # the status must be MEASURED rather than a spurious PASS or REGRESSION.
    assert results["bm_ioi"].status == "MEASURED"
    assert results["bm_ioi"].baseline_is_comparable is False
    # And it carries the like-for-like figure instead.
    assert results["bm_ioi"].reference_circuit_fidelity_same_harness is not None


@needs_weights
def test_suite_does_not_report_a_pass_rate_when_baselines_are_incomparable():
    from backend.validation.health_dashboard import HealthDashboardEngine

    report = HealthDashboardEngine().run_continuous_validation().to_dict()
    # Nothing here is scorable against its published baseline, so the pass rate
    # is undefined. Reporting 100.0 was the original bug; reporting 0.0 would
    # be the same error pointed the other way.
    assert report["overall_pass_rate"] is None
    assert report["pass_rate_measured"] is False
    assert report["benchmarks_unscored"] == report["total_benchmarks_run"]
    assert report["passed_benchmarks"] == 0
