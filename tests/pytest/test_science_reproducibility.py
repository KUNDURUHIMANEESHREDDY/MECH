"""Pytest tests for Landmark Paper Reproducibility Pipelines."""

from __future__ import annotations

import re

import pytest

from science.reproducibility.greater_than_pipeline import GreaterThanCircuitPipeline
from science.reproducibility.induction_heads_pipeline import InductionHeadsPipeline
from science.reproducibility.ioi_pipeline import IOIReproductionPipeline
from science.reproducibility.logit_lens_pipeline import LogitLensPipeline
from science.reproducibility.paper_registry import BenchmarkRegistry
from science.reproducibility.reproducibility_report import ReproducibilityReportEngine
from science.reproducibility.sae_pipeline import SAEReproductionPipeline
from science.reproducibility.dataset_versioning import DatasetVersioningEngine


# ── Paper Registry ────────────────────────────────────────────────────────────

def test_paper_registry_has_all_papers():
    registry = BenchmarkRegistry()
    ids = registry.list_paper_ids()
    assert "ioi" in ids
    assert "induction_heads" in ids
    assert "greater_than" in ids
    assert "logit_lens" in ids
    assert "sparse_autoencoders" in ids


def test_paper_registry_metrics_defined():
    registry = BenchmarkRegistry()
    paper = registry.get("ioi")
    assert paper is not None
    assert len(paper.required_metrics) == 3
    metric_names = [m.name for m in paper.required_metrics]
    assert "circuit_faithfulness" in metric_names
    assert "circuit_completeness" in metric_names


def test_paper_registry_custom_registration():
    from science.reproducibility.paper_registry import Paper, ExpectedMetric
    registry = BenchmarkRegistry()
    paper = Paper(
        paper_id="test_paper", title="Test Paper", authors=["A"], year=2024,
        venue="TestConf", arxiv_id="0000.00000",
        primary_model="gpt2-small", dataset_name="Test Dataset",
        dataset_version="1.0.0", tokenizer_id="gpt2", random_seed=42,
        required_metrics=[ExpectedMetric("test_metric", 0.90, "ratio")],
        pipeline_class="test.Pipeline",
    )
    registry.register(paper)
    assert registry.get("test_paper") is not None


# ── Dataset Versioning ────────────────────────────────────────────────────────

def test_dataset_versioning_creates_manifest():
    engine = DatasetVersioningEngine()
    manifest = engine.create_manifest(
        paper_id="ioi", pipeline_name="IOIReproductionPipeline",
        model_id="gpt2-small", hf_repo_id="gpt2",
        dataset_name="IOI Dataset", random_seed=42,
    )
    assert manifest.paper_id == "ioi"
    assert manifest.random_seed == 42
    assert manifest.python_version != ""


# ── Reproducibility Report Engine ────────────────────────────────────────────

def test_report_engine_gold_tier():
    engine = ReproducibilityReportEngine()
    report = engine.generate_report(
        paper_id="ioi", pipeline_name="IOIPipeline", model_id="gpt2-small",
        dataset_manifest_id="manifest_test",
        observed_metrics={"circuit_faithfulness": 0.87, "circuit_completeness": 0.82, "circuit_minimality": 0.93},
    )
    assert report["overall_tier"] in ("Gold", "Silver", "Bronze")
    assert len(report["metric_results"]) == 3
    assert report["overall_fidelity_pct"] > 90.0


def test_report_engine_needs_investigation():
    engine = ReproducibilityReportEngine()
    report = engine.generate_report(
        paper_id="ioi", pipeline_name="IOIPipeline", model_id="gpt2-small",
        dataset_manifest_id="manifest_bad",
        observed_metrics={"circuit_faithfulness": 0.40, "circuit_completeness": 0.35, "circuit_minimality": 0.30},
    )
    assert report["overall_tier"] == "Needs Investigation"


# ── IOI Pipeline ──────────────────────────────────────────────────────────────

def test_ioi_pipeline_runs():
    """The IOI pipeline must either measure or say it could not.

    It fails closed when the model does not return the required comparison
    tokens. The assertions below therefore hold only on a completed run; a
    blocked run must expose no metrics and must state a reason, rather than
    carrying an `observed_metrics` block of placeholder numbers.
    """
    pipeline = IOIReproductionPipeline(mock_mode=True)
    result = pipeline.run(n_prompts=20, seed=42)
    assert result["pipeline"] == "IOIReproductionPipeline-HighFidelity"

    if result.get("status") == "completed":
        assert result["observed_metrics"]["n_samples"] == 20
        assert "circuit_faithfulness" in result["observed_metrics"]
        assert result["reproducibility_report"]["overall_tier"] in (
            "Gold", "Silver", "Bronze", "Needs Investigation")
    else:
        assert result["status"] in ("unavailable", "blocked", "error")
        assert result["provenance"] != "live"
        assert result["validation_eligible"] is False
        assert result["publication_eligible"] is False
        assert result["reason"]
        # No placeholder metrics masquerading as a completed measurement.
        assert not isinstance(result.get("observed_metrics"), dict)


def test_ioi_pipeline_generates_manifest_only_when_completed():
    """A manifest identifies a real run; a blocked pipeline has none."""
    pipeline = IOIReproductionPipeline(mock_mode=True)
    result = pipeline.run(n_prompts=10)

    if result.get("status") == "completed":
        assert result["manifest_id"].startswith("manifest_ioi")
    else:
        assert "manifest_id" not in result
        assert result["reason"]


# ── Induction Heads Pipeline ──────────────────────────────────────────────────

def test_induction_heads_pipeline_reports_shape_without_inventing_heads():
    """Mock mode must report unmeasured, not return the published heads.

    This test asserted `len(result["induction_heads_found"]) >= 3`, and the
    pipeline satisfied it by scoring heads as `random.uniform(0.75, 0.92)` if
    they were in a hardcoded canonical set. The attention matrix it computed to
    do that was discarded.

    In mock mode there is no model, so no attention exists to measure. The
    honest result is unavailable, and that is what this asserts.
    """
    pipeline = InductionHeadsPipeline(mock_mode=True)
    result = pipeline.run(n_sequences=20)

    assert result["pipeline"] == "InductionHeadsPipeline"
    metrics = result["observed_metrics"]
    for name in ("induction_score", "prefix_match_accuracy",
                 "in_context_learning_score"):
        assert name in metrics, name
        assert metrics[name] is None, (
            f"{name} was {metrics[name]!r} with no model loaded; a mock must "
            f"not produce a metric value"
        )

    assert result["provenance"] == "unavailable"
    assert result["validation_eligible"] is False
    assert "induction_heads_found" not in result


def test_induction_heads_are_not_hardcoded():
    """No detection path may consult the canonical list.

    The published canonical set was the detection criterion: score was high iff
    the head was in it, so only those heads could ever be returned and the
    threshold could not exclude anything. This checks the list is only read for
    the post-hoc comparison.
    """
    import inspect

    from backend.science.reproducibility import induction_heads_pipeline as mod

    source = inspect.getsource(mod)
    # The list may appear in the reference comparison and in docs, but never in
    # a membership test that feeds a score.
    assert ".intersection(CANONICAL_HEADS)" in source, (
        "expected the canonical set to be used for the overlap comparison"
    )
    for forbidden in ("in CANONICAL_HEADS", "CANONICAL_HEADS if",
                      "CANONICAL_HEADS and"):
        assert forbidden not in source, (
            f"{forbidden!r} makes the canonical list a detection criterion"
        )


def test_attention_measurement_requires_real_weights():
    """The bug this pipeline had: sdpa returns None for every layer.

    transformers returns a tuple of Nones for `output_attentions=True` under
    the sdpa path. Nothing raises at the model boundary; the failure appears
    much later as 'NoneType' is not subscriptable. The adapter now forces
    eager, and this asserts attentions actually arrive.
    """
    import torch

    from backend.science.models.gpt2_adapter import GPT2Adapter

    adapter = GPT2Adapter(variant="small", mock_mode=False)
    if adapter._model is None:
        import pytest
        pytest.skip("GPT-2 weights are not loaded")

    token_ids = adapter._tokenizer("the cat sat on the")["input_ids"]
    ids = torch.tensor([token_ids], device=adapter._model.device)
    outputs = adapter._forward_with_hooks_ids(ids)

    assert outputs.attentions is not None
    none_layers = [i for i, t in enumerate(outputs.attentions) if t is None]
    assert not none_layers, (
        f"layers {none_layers} returned no attention weights; eager attention "
        f"must be forced for measurements to be possible"
    )
    # [batch, heads, query, key] -- the query/key axes are the sequence length.
    seq_len = len(token_ids)
    assert outputs.attentions[0].shape[-2:] == (seq_len, seq_len)


# ── Greater-Than Pipeline ─────────────────────────────────────────────────────

def test_greater_than_pipeline_refuses_without_weights():
    """A circuit measurement with no weights must fail closed.

    This test previously asserted the opposite: it ran the pipeline with
    mock_mode=True and required a populated result containing
    `patch_effect_magnitude` and `circuit_accuracy`. Those values came from a
    hardcoded `{7: 0.82, 8: 0.91, 9: 0.78}` table that `_compute_patch_effect`
    returned without ever looking at its prompt argument, so the "measurement"
    was structurally incapable of disagreeing with the paper it reproduced.

    `circuit_accuracy` was independently wrong: it tested whether the century
    string appeared in a single next-token prediction, which counts (1942, 1918)
    as incorrect because range(1942, 1919) is empty.
    """
    from science.models.adapter_base import LiveUnavailable

    pipeline = GreaterThanCircuitPipeline(mock_mode=True)
    with pytest.raises(LiveUnavailable) as excinfo:
        pipeline.run()
    assert "weights" in str(excinfo.value).lower()


def test_greater_than_pipeline_has_no_hardcoded_patch_effect_table():
    """Guard against the fabricated measurement being reintroduced.

    Source-level because the fabrication was not reachable by any behavioural
    probe: the function returned its argument-independent lookup table for every
    input, so any test asserting "it runs" passed while nothing was measured.
    """
    from source_assert import executable_source

    import science.reproducibility.greater_than_pipeline as gt

    src = executable_source(gt)

    assert "_compute_patch_effect" not in src, (
        "the reference-keyed patch-effect lookup is back"
    )
    # Any dict literal mapping a layer number to a plausible float in 0..1 is the
    # shape of the original fabrication.
    assert not re.search(r"\{\s*7\s*:\s*0\.\d+\s*,\s*8\s*:\s*0\.\d+", src), (
        "a hardcoded per-layer effect table has reappeared"
    )
    # The dominant layer must be derived from the measurement.
    assert "dominant = max(means, key=means.get)" in src


# ── Logit Lens Pipeline ───────────────────────────────────────────────────────

def test_logit_lens_pipeline_refuses_without_weights():
    """A layer sweep with no weights must fail closed.

    This previously ran with mock_mode=True and passed, because the sweep it
    returned was not a measurement:

    * `top_token` per layer was `expected if progress > 0.60 else " the"` -- the
      correct answer hardcoded in past 60% depth, so the lens converged on the
      right answer by construction. Converging early is the whole thing the logit
      lens is used to test.
    * `_layer_entropy` returned `3.5 * exp(-2.0 * layer / n)`.
    * `_convergence_layer` returned `int(n_layers * 0.65)` -- always 7 for GPT-2,
      for every prompt and every model.
    """
    from science.models.adapter_base import LiveUnavailable

    pipeline = LogitLensPipeline(mock_mode=True)
    with pytest.raises(LiveUnavailable) as excinfo:
        pipeline.run()
    assert "weights" in str(excinfo.value).lower()


def test_logit_lens_has_no_hardcoded_sweep():
    """Guard against the constructed sweep being reintroduced."""
    from source_assert import executable_source

    import science.reproducibility.logit_lens_pipeline as ll

    src = executable_source(ll)

    # Match the definitions, not the bare words: "mean_convergence_layer" is a
    # metric key in observed_metrics and would match a looser pattern.
    assert not re.search(r"def\s+_layer_entropy\b", src), (
        "entropy is coming from a hand-written exponential again"
    )
    assert not re.search(r"def\s+_convergence_layer\b", src), (
        "the convergence layer is a constant fraction of depth again"
    )
    # The answer must never be substituted for a prediction.
    assert not re.search(r'top_token\s*=\s*expected', src), (
        "the expected token is being written into the layer sweep"
    )
    assert not re.search(r"math\.exp\(", src), (
        "entropy is being generated rather than measured"
    )


# ── SAE Pipeline ──────────────────────────────────────────────────────────────

def test_sae_pipeline_measures_or_refuses_but_never_fabricates():
    """The contract, at the level it should have been written.

    These three tests previously asserted the opposite. They could only pass:

    * `test_sae_pipeline_sparsity` required `l0_sparsity > 0.50`, which
      `rng.betavariate(0.5, 5.0)` guaranteed -- the distribution was chosen so
      that most simulated features land below the 0.05 activation threshold.
    * `test_sae_top_features_monosemantic` required all ten top features to
      exceed 0.5, guaranteed twice over: `rng.betavariate(3.0, 1.5)` skews high,
      and the list was *sorted by that same score*, so the top ten are the ten
      highest draws by construction.

    `top_activating_tokens` was `rng.sample` over a 19-word list, and
    `reconstruction_mse` was `0.03 + (1 - mean_monosemanticity) * 0.04` -- no
    autoencoder was trained or evaluated. The simulation also ran regardless of
    mock_mode, and the dataset manifest claimed OpenWebText, never read.

    Then a later fix made the pipeline refuse outright, and the test was
    rewritten to demand `LiveUnavailable` with "not implemented" in the
    message. That was correct while it was true and became false when the
    pipeline started training a real top-k SAE.

    So the test now states the invariant that survives both states: the pipeline
    either measures against real weights or refuses, and never returns a feature
    bank that did not come from a trained autoencoder. Pinning either specific
    outcome would break the moment the other became correct -- which is what
    just happened.
    """
    from science.models.adapter_base import LiveUnavailable
    from science.reproducibility.sae_pipeline import SAEReproductionPipeline

    pipeline = SAEReproductionPipeline()

    try:
        result = pipeline.run(n_features=8, n_tokens=32, steps=2)
    except LiveUnavailable as exc:
        # Refusing is a valid outcome. It must say what it is refusing to do,
        # so a reader can tell missing weights from missing code.
        message = str(exc).lower()
        assert any(phrase in message for phrase in
                   ("requires torch", "needs loaded", "could not be fitted",
                    "no features", "empty")), (
            f"the refusal does not say what it is refusing to do: {message[:200]}")
        return

    # A result is the other valid outcome. Everything in it must be measured.
    assert result["provenance"] in ("live", "unavailable"), (
        f"unexpected provenance {result['provenance']!r}")
    metrics = result["observed_metrics"]

    if result["status"] == "unavailable":
        # An unusable fit must say so rather than reporting the numbers.
        assert metrics["reconstruction_useful"] is False or \
            metrics.get("normalized_mse") is None, (
            "status is unavailable but the payload looks like a usable fit")
        return

    assert result["status"] == "completed"
    assert metrics["reconstruction_mse"] is not None
    assert metrics["n_activations"] > 0, (
        "a result with no activations was produced; nothing was measured")
    # A reconstruction no better than predicting the mean is not a result.
    assert metrics["reconstruction_useful"] is True, (
        "status is completed but the reconstruction is worse than the mean")
    assert metrics["monosemanticity_scored"] is False


def test_sae_pipeline_has_no_feature_simulation():
    """Guard against the RNG feature bank being reintroduced."""
    from source_assert import executable_source

    import science.reproducibility.sae_pipeline as sae

    src = executable_source(sae)

    assert "_simulate_sae_features" not in src
    # Match the *call*, not the bare word. The raise message names what the old
    # code did ("a betavariate draw", "claimed OpenWebText"), and that message is
    # executable -- a string literal in a raise, not a docstring -- so
    # executable_source() legitimately keeps it. Matching words would make this
    # guard fail on its own explanation.
    assert not re.search(r"\.betavariate\(", src), (
        "measured distributions are being drawn from an RNG again"
    )
    assert not re.search(r"\brng\.sample\b|\brandom\.sample\b", src), (
        "top_activating_tokens is being drawn from a static word list again"
    )
    # A manifest naming a corpus is a keyword argument, so require the form
    # rather than the bare string.
    assert not re.search(r"dataset_name\s*=\s*[\"'][^\"']*OpenWebText", src), (
        "a dataset this platform does not read is still being claimed"
    )


def test_arithmetic_pipeline_refuses_rather_than_fabricating():
    """The arithmetic accuracies were unconditional, not mock-gated.

    `ArithmeticPipeline.run` returned `{0.45, 0.85}` for every caller including
    ones with weights loaded, and `benchmark_runner` reached it through its
    `run(model_id=...)` fallback and stored the result in a report.
    """
    from science.models.adapter_base import LiveUnavailable
    from science.reproducibility.arithmetic_pipeline import ArithmeticPipeline

    pipeline = ArithmeticPipeline(model_manager=None)
    with pytest.raises(LiveUnavailable) as excinfo:
        pipeline.run(model_id="gpt2-small")
    assert "not implemented" in str(excinfo.value).lower()


def _has_gpt2() -> bool:
    """Whether real GPT-2 weights are loadable here.

    Same helper as test_benchmark_executor_measurement.py: `needs_weights` is a
    skipif alias rather than a registered marker, so a test that needs a forward
    pass has to gate on this or it fails wherever the weights are absent.
    """
    try:
        from science.models.gpt2_adapter import GPT2Adapter
        return GPT2Adapter(variant="small", mock_mode=False)._model is not None
    except Exception:
        return False


needs_weights = pytest.mark.skipif(
    not _has_gpt2(), reason="GPT-2 weights are not loaded"
)


def test_copy_and_factual_recall_refuse_rather_than_fabricate():
    """Two more unconditional fabrications, both in benchmark_runner's dict.

    `CopyTaskPipeline.run` returned 0.92 / 0.88 and `FactualRecallPipeline.run`
    returned 0.75 / 0.65, each under a comment claiming a mock environment that
    the code did not implement -- there was no flag, so callers with weights
    loaded received the numbers and benchmark_runner wrote them into reports.
    """
    from science.models.adapter_base import LiveUnavailable
    from science.reproducibility.copy_task_pipeline import CopyTaskPipeline
    from science.reproducibility.factual_recall_pipeline import FactualRecallPipeline

    for pipeline in (CopyTaskPipeline(model_manager=None),
                     FactualRecallPipeline(model_manager=None)):
        with pytest.raises(LiveUnavailable) as excinfo:
            pipeline.run(model_id="gpt2-small")
        assert "not implemented" in str(excinfo.value).lower()


def test_benchmark_runner_does_not_fabricate_vram_or_manifest():
    """The runner fabricated provenance around whichever pipeline it called.

    Three separate inventions, all in `run_all`:

    * `peak_vram = 6.7 if "gpt2" in model_id else 12.4` -- a constant chosen by
      model name, stored under the key `peak_vram_gb`, i.e. presented as an
      observed quantity. It was never read from the GPU.
    * `dataset_manifest_id="mock_dataset_manifest_v1"` -- a literal dataset
      identity stamped onto every generated report, so each one claimed
      traceability to a dataset that never existed.
    * `explanation_of_diffs=["Milestone A Validation Run"]` -- a fixed narrative
      asserting a validation that had not happened.

    Source-level because the numbers were plausible and nothing downstream
    compared them against hardware, so a behavioural probe would pass.
    """
    from source_assert import executable_source

    import science.reproducibility.benchmark_runner as runner

    src = executable_source(runner)

    assert not re.search(r"""6\.7\s+if\s+["']gpt2["']""", src), (
        "peak VRAM is being chosen by model name instead of measured"
    )
    assert not re.search(r"""["']mock_dataset_manifest_v1["']""", src), (
        "a fabricated dataset manifest id is being attached to reports"
    )
    assert "Milestone A Validation Run" not in src, (
        "a fixed narrative is being reported as an explanation of differences"
    )
    # Peak VRAM must come from torch, or be absent.
    assert "max_memory_allocated" in src
    # In mock mode nothing reaches the GPU, so peak VRAM is not applicable.
    # torch.cuda.max_memory_allocated() reports 0.0 there, which reads as
    # "measured: the model needs no memory" rather than "not measured".
    assert re.search(r"if\s+self\.mock_mode\s+or\s+not\s+torch\.cuda\.is_available\(\)", src), (
        "mock-mode runs must report peak VRAM as None, not torch's 0.0"
    )


def test_benchmark_runner_does_not_default_to_mock_mode():
    """`run_all` built every pipeline in mock mode, unconditionally.

    `self.mock_mode = True` was hardcoded in `__init__`, so the class documented
    as "the continuous integration suite" could only ever produce fixtures --
    and produced them through the same report path as real measurements.
    """
    import inspect as _inspect

    from science.reproducibility.benchmark_runner import BenchmarkRunner

    default = _inspect.signature(BenchmarkRunner.__init__).parameters["mock_mode"].default
    assert default is False, (
        f"BenchmarkRunner defaults to mock_mode={default!r}; fixtures belong in "
        f"tests, not in the runner that claims to validate"
    )


def test_benchmark_runner_does_not_probe_signatures_with_typeerror():
    """A TypeError raised inside a measurement must not trigger a retry.

    The runner called `pipeline.run(seed=seed)` and fell back to
    `pipeline.run(model_id=...)` on TypeError. That conflated a genuine bug
    inside a measurement with a signature mismatch -- and it is how the
    arithmetic stub's fabricated values were reached, since
    `ArithmeticPipeline.run(model_id)` rejects `seed`.
    """
    from source_assert import executable_source

    import science.reproducibility.benchmark_runner as runner

    src = executable_source(runner)

    assert not re.search(r"except\s+TypeError", src), (
        "signature detection is still done by catching TypeError"
    )
    assert "inspect.signature" in src


def test_benchmark_runner_distinguishes_not_run_from_error():
    """Four of the six pipelines now raise by design; that is not a crash."""
    from source_assert import executable_source

    import science.reproducibility.benchmark_runner as runner

    src = executable_source(runner)

    assert "except LiveUnavailable" in src, (
        "LiveUnavailable is being reported as ERROR, conflating 'not "
        "measured' with 'the measurement broke'"
    )


@needs_weights
@pytest.mark.parametrize("mock_mode", [True, False])
def test_benchmark_runner_never_reports_a_fixture_as_a_pass(mock_mode):
    """A mock-mode run must not say PASS.

    `status` defaulted to "PASS" for any run that did not raise, so a fixture
    run -- which measured nothing -- was indistinguishable from a completed
    measurement to anything reading the result dict.
    """
    from science.reproducibility.benchmark_runner import BenchmarkRunner

    result = BenchmarkRunner(mock_mode=mock_mode).run_all(model_id="gpt2-small")

    statuses = {pid: d["status"] for pid, d in result["reports"].items()}
    ran = [s for s in statuses.values() if s in ("PASS", "FIXTURE")]

    assert ran, f"expected some pipelines to complete, got {statuses}"
    if mock_mode:
        assert "PASS" not in statuses.values(), (
            f"a fixture run reported PASS: {statuses}"
        )
        assert all(s == "FIXTURE" for s in ran), statuses
    else:
        assert "FIXTURE" not in statuses.values(), (
            f"a real run was labelled a fixture: {statuses}"
        )

    # Unrun benchmarks never carry a VRAM number, in either mode.
    for pid, d in result["reports"].items():
        if d["status"] == "NOT_RUN":
            assert d["peak_vram_gb"] is None, (
                f"{pid} did not run but reports peak_vram_gb={d['peak_vram_gb']}"
            )


def test_logit_lens_final_layer_matches_the_model_forward_pass():
    """The lens must reproduce the real forward pass at its last layer.

    This is the check that makes the intermediate rows trustworthy. An earlier
    version pushed `hidden_states[-1]` back through `ln_f`, which transformers
    may have already applied, and the resulting final-layer logit differed from
    the model's own by 84 units while still yielding the same argmax -- so a
    token-only assertion would have passed on a lens that was quietly wrong.
    """
    from science.models.gpt2_adapter import GPT2Adapter

    adapter = GPT2Adapter(variant="small", mock_mode=False)
    lens = adapter.logit_lens("The capital of France is")
    real = adapter.get_logits("The capital of France is")

    final = lens["layers"][-1]
    assert final["layer"] == len(lens["layers"]) - 1
    assert final["top_token"] == real["top_token"]
    assert abs(final["top_logit"] - real["top_tokens"][0]["logit"]) < 0.01
    assert final.get("is_model_output") is True
    # Entropy must be finite; it was NaN before, because softmax underflows
    # ~50,000 of GPT-2's 50,257 vocabulary entries to exact zero at these logit
    # magnitudes and the log was taken on the clamped zero.
    assert all(row["entropy"] == row["entropy"] for row in lens["layers"])
