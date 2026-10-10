"""The benchmark executor must measure, not reproduce the paper by construction.

Every path in `benchmark_tasks.py` used to return
`reference.metric_value + noise`. The IOI and induction-heads tasks now run
their real reproduction pipelines, and these tests pin that:

* a real run produces a score derived from the pipeline, not from the reference
* the confidence interval comes from observed trials, and names what it covers
* resource figures follow from a real token count
* a task with no implemented pipeline says so instead of inventing a number

Tests needing GPT-2 skip when the weights are absent; the fail-closed
assertions run unconditionally, because those are the properties that matter
most and they are the ones that were broken.
"""
import pytest

from benchmarking.benchmark_tasks import (
    BenchmarkTask,
    BenchmarkTaskExecutor,
    ExecutionMode,
    TASK_CATALOGUE,
)


from _weight_guard import skip_reason, weights_available


needs_weights = pytest.mark.skipif(
    not weights_available(), reason=skip_reason()
)


# ── Unimplemented work refuses, it does not invent ──────────────────────────

def test_task_without_a_pipeline_raises_rather_than_inventing():
    """Six of the eight catalogue tasks have no measurement implementation."""
    spec = TASK_CATALOGUE[BenchmarkTask.GREATER_THAN]
    with pytest.raises(NotImplementedError) as exc:
        BenchmarkTaskExecutor().execute(
            spec, "gpt2", "hf", mode=ExecutionMode.PRODUCTION, n_samples=4,
        )
    message = str(exc.value)
    assert "No measurement pipeline is implemented" in message
    # The message names what *is* implemented, so the caller has a next step.
    assert "IOIReproductionPipeline" in message
    assert "InductionHeadsPipeline" in message
    # And it says the old behaviour, so nobody reinstates it.
    assert "published reference value" in message


def test_mock_mode_is_labelled_a_fixture():
    """A fixture must never look like a measurement."""
    result = BenchmarkTaskExecutor().execute(
        TASK_CATALOGUE[BenchmarkTask.IOI], "gpt2", "hf",
        mode=ExecutionMode.MOCK, n_samples=8,
    )
    assert result.is_fixture is True
    assert result.backend_effective == "mock fixture"
    # No real trials, so no interval.
    assert result.confidence_interval_derived is False
    assert result.confidence_interval_low is None
    assert result.confidence_interval_target is None


def test_caller_backend_is_recorded_but_not_claimed_as_the_execution_path():
    """`backend` was ignored by every code path before.

    A caller passing "transformerlens" had no way to tell that the value was
    never consulted, so the result now records what actually ran.
    """
    result = BenchmarkTaskExecutor().execute(
        TASK_CATALOGUE[BenchmarkTask.IOI], "gpt2", "transformerlens",
        mode=ExecutionMode.MOCK, n_samples=4,
    )
    assert result.backend == "transformerlens"
    assert result.backend_effective != "transformerlens"


def test_unknown_model_refuses_to_estimate_flops():
    """Parameter counts are resolved, not inferred from the id's substrings."""
    with pytest.raises(ValueError, match="Parameter count"):
        BenchmarkTaskExecutor().execute(
            TASK_CATALOGUE[BenchmarkTask.IOI], "mystery-model-7b", "hf",
            mode=ExecutionMode.MOCK, n_samples=4,
        )


# ── Real runs ──────────────────────────────────────────────────────────────

@needs_weights
def test_ioi_score_comes_from_the_pipeline_not_the_reference():
    """The old code returned reference + noise, which always lands near the
    published number. The real pipeline measures, and here it underperforms."""
    spec = TASK_CATALOGUE[BenchmarkTask.IOI]
    result = BenchmarkTaskExecutor().execute(
        spec, "gpt2", "hf", mode=ExecutionMode.PRODUCTION, n_samples=8,
    )

    assert result.is_fixture is False
    assert result.backend_effective == "IOIReproductionPipeline"
    assert result.primary_score != pytest.approx(spec.reference.metric_value * 100)

    from backend.science.reproducibility.ioi_pipeline import IOIReproductionPipeline
    direct = IOIReproductionPipeline(mock_mode=False).run(n_prompts=8)
    expected = direct["observed_metrics"]["circuit_faithfulness"] * 100
    assert result.primary_score == pytest.approx(expected, abs=0.05)


@needs_weights
def test_ioi_interval_comes_from_observed_prompt_outcomes():
    result = BenchmarkTaskExecutor().execute(
        TASK_CATALOGUE[BenchmarkTask.IOI], "gpt2", "hf",
        mode=ExecutionMode.PRODUCTION, n_samples=8,
    )
    assert result.confidence_interval_derived is True
    assert "observed outcomes" in result.confidence_interval_method
    assert result.confidence_interval_low is not None
    assert result.confidence_interval_high is not None
    # Wilson stays inside [0, 100] even at proportions the normal
    # approximation mishandles.
    assert 0.0 <= result.confidence_interval_low <= result.confidence_interval_high <= 100.0


@needs_weights
def test_induction_interval_names_the_quantity_it_covers():
    """The primary score is a mean attention fraction, not a proportion.

    Reporting a narrow behavioural interval beside it without saying so would
    present it as a bound on the headline number.
    """
    result = BenchmarkTaskExecutor().execute(
        TASK_CATALOGUE[BenchmarkTask.INDUCTION_HEADS], "gpt2", "hf",
        mode=ExecutionMode.PRODUCTION, n_samples=6,
    )
    assert result.confidence_interval_derived is True
    assert result.confidence_interval_target == "behavioural_accuracy"
    assert result.backend_effective == "InductionHeadsPipeline"


@needs_weights
def test_real_token_count_makes_throughput_real():
    """Throughput used to be divided by `n * 50`, a count nobody made."""
    result = BenchmarkTaskExecutor().execute(
        TASK_CATALOGUE[BenchmarkTask.IOI], "gpt2", "hf",
        mode=ExecutionMode.PRODUCTION, n_samples=8,
    )
    assert result.tokens_per_sec is not None
    assert result.tokens_per_sec > 0
    assert result.resource_metrics_measured is True
    assert result.flops is not None
    # A real count implies a real denominator; the old estimate would have
    # implied 400 tokens for 8 prompts.
    assert result.mean_latency_ms is not None


@needs_weights
def test_induction_heads_are_recovered_by_measurement_alone():
    """100% overlap with the published set, derived not hardcoded.

    The canonical list is only read for the post-hoc comparison, so this is a
    genuine prediction rather than a lookup.
    """
    from backend.science.reproducibility.induction_heads_pipeline import (
        InductionHeadsPipeline,
    )
    result = InductionHeadsPipeline(mock_mode=False).run(
        n_sequences=12, seq_len=8, seed=42)

    comparison = result["reference_comparison"]
    assert comparison["overlap_pct"] == 100.0, comparison
    # And the mechanism is attributed by measurement, not assumed.
    assert result["mechanism"]["mechanism"] == "previous_token_copying"
    assert (result["mechanism"]["best_previous_token"]
            > result["mechanism"]["best_induction"])
    # Behaviourally the copy is real: high accuracy with the repeat, none
    # without it.
    assert result["behaviour"]["repeated_block_accuracy"] > 0.8
    assert result["behaviour"]["single_block_accuracy"] < 0.2


@needs_weights
def test_published_ioi_baseline_is_not_the_same_measurement():
    """The 0.86 in the registry is not what this harness computes.

    Measured directly: running the *published* IOI circuit through
    `ioi_pipeline`'s own harness yields materially less than 0.86. So
    comparing a discovered circuit against 0.86 compares a quantity with an
    unrelated one, and a "failure to reproduce" reading of it is unsound.
    """
    from backend.science.reproducibility.ioi_pipeline import IOIReproductionPipeline

    result = IOIReproductionPipeline(mock_mode=False).run(n_prompts=10)
    metrics = result["observed_metrics"]

    same_harness = metrics["reference_circuit_faithfulness_same_harness"]
    assert same_harness is not None
    assert 0.0 <= same_harness <= 1.0

    # The published constant is 0.86. This harness's own reading of the
    # published circuit is lower, which is exactly why the flag is set.
    assert same_harness < 0.86, (
        "if this harness now reproduces 0.86 for the published circuit, the "
        "two numbers have converged and the calibration can be simplified"
    )
    assert metrics["published_baseline_is_comparable"] is False
    assert "not the same measurement" in metrics["published_baseline_note"]


@needs_weights
def test_discovered_circuit_is_compared_like_for_like_against_the_reference():
    """The answerable question: is it as good as the published circuit?

    Both numbers come from identical code on identical prompts, so the ratio
    is meaningful. This is the comparison that replaced "0.6981 vs 0.86".
    """
    from backend.agents.critic import Critic

    out = Critic().reproduce("ioi", n_prompts=10)
    calibrated = out["calibrated_vs_published_circuit"]
    assert calibrated is not None, "no like-for-like comparison was produced"

    assert calibrated["basis"] == (
        "both circuits measured by this pipeline, same prompts"
    )
    assert calibrated["discovered"] > 0
    assert calibrated["published_circuit_same_harness"] > 0
    assert calibrated["ratio"] == pytest.approx(
        calibrated["discovered"] / calibrated["published_circuit_same_harness"],
        abs=1e-3,
    )
    # The gate must state that its own basis is the external constants, so a
    # reader cannot mistake `passed: False` for a like-for-like failure.
    assert out["gate"]["reference_basis"] == "published_external_constants"
    assert out["gate"]["reference_basis_is_like_for_like"] is False
    assert out["gate"]["calibrated"] == calibrated


@needs_weights
def test_calibrated_comparison_does_not_decide_the_verdict():
    """Beating the reference circuit must not become the gate's decision.

    Matching a circuit found by the same pipeline is evidence that the
    measurement works, not evidence that a paper is validated. The two
    questions stay separate, so the calibration is reported alongside the gate
    rather than substituted into it.
    """
    from agents.critic import Critic, CONFIDENCE_THRESHOLD

    out = Critic().reproduce("ioi", n_prompts=10)
    gate = out["gate"]
    calibrated = out["calibrated_vs_published_circuit"]

    # The verdict follows the external-constant comparison only, whatever the
    # calibration says.
    assert gate["passed"] == (
        gate["value"] >= CONFIDENCE_THRESHOLD * 100
    )
    assert gate["metric"] == "overall_fidelity_pct"

    # And the calibration is surfaced as its own, clearly-labelled field
    # rather than folded into the verdict.
    assert calibrated is not None
    assert calibrated["at_least_published"] is (
        calibrated["ratio"] >= 1.0
    )
    # It never claims eligibility for itself.
    assert "publication_eligible" not in calibrated
    assert "validation_eligible" not in calibrated



@needs_weights
def test_ioi_discovery_is_not_echoing_the_published_head_list():
    """Overlap must be partial, which is what proves independence.

    If discovery returned the published heads, overlap would be 100% -- and
    `docs/roadmap.md` records a measured 3-of-10. Pinning that it is strictly
    less than the full list is the stable invariant: it cannot drift as
    discovery improves, but it would fail immediately if the published list
    were ever used as the answer.
    """
    from backend.science.reproducibility.ioi_pipeline import (
        REFERENCE_ONLY_DISCOVERED_HEADS,
        IOIReproductionPipeline,
    )

    result = IOIReproductionPipeline(mock_mode=False).run(n_prompts=10)
    discovered = set(result["observed_metrics"]["discovered_nodes"])
    published = set(REFERENCE_ONLY_DISCOVERED_HEADS)

    assert result["observed_metrics"]["discovery_provenance"] == "live"
    assert discovered, "no circuit was discovered"
    assert len(discovered & published) < len(published), (
        "the discovered set contains every published head, which is what "
        "returning the reference list would look like"
    )
    # And the discovered set is not merely a subset of the published one either.
    assert discovered - published, (
        "discovery returned only published heads, so it found nothing new"
    )


@needs_weights
def test_induction_metrics_are_measured_not_drawn():
    """A measured metric is reproducible from its seed and matches its counts.

    The old ICL score was ``uniform(0.55,0.65) - itself + uniform(0.08,0.14)``,
    which algebraically equals the last draw, so it landed in 0.08-0.14 for
    every run and could not be anything else.

    Two properties distinguish a measurement from a draw, and both are checked
    here. Reproducibility: the same seed gives the same value, because the
    metric is a function of the data rather than of a free-running RNG.
    Agreement with the counts: the reported accuracy equals correct/total over
    the predictions actually made. Note that different seeds may still yield the
    same accuracy -- two stimuli can give the same number correct -- so
    variation across seeds is not the property being asserted.
    """
    from backend.science.reproducibility.induction_heads_pipeline import (
        InductionHeadsPipeline,
    )

    def run(seed: int):
        return InductionHeadsPipeline(mock_mode=False).run(
            n_sequences=8, seq_len=8, seed=seed)

    first, repeat = run(7), run(7)
    assert first["observed_metrics"] == repeat["observed_metrics"], (
        "the same seed produced different metrics; they are being drawn, not "
        "measured"
    )

    behaviour = first["behaviour"]
    assert behaviour["measured"] is True
    correct, total = behaviour["repeated_block_correct"], behaviour["repeated_block_n"]
    assert total > 0
    assert behaviour["repeated_block_accuracy"] == pytest.approx(
        correct / total, abs=1e-4
    )
    control_correct = behaviour["single_block_correct"]
    control_total = behaviour["single_block_n"]
    assert behaviour["single_block_accuracy"] == pytest.approx(
        control_correct / control_total, abs=1e-4
    )
    assert first["observed_metrics"]["in_context_learning_score"] == pytest.approx(
        behaviour["in_context_gain"], abs=1e-4
    )
    # And it is not the old constant band.
    assert behaviour["in_context_gain"] != pytest.approx(0.11, abs=0.03)
