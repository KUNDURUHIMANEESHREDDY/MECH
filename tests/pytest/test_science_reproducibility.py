"""Pytest tests for Landmark Paper Reproducibility Pipelines."""

from __future__ import annotations

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

def test_greater_than_pipeline_runs():
    pipeline = GreaterThanCircuitPipeline(mock_mode=True)
    result = pipeline.run()
    assert result["pipeline"] == "GreaterThanCircuitPipeline"
    assert "patch_effect_magnitude" in result["observed_metrics"]
    assert "circuit_accuracy" in result["observed_metrics"]


def test_greater_than_layer_effects():
    pipeline = GreaterThanCircuitPipeline(mock_mode=True)
    result = pipeline.run()
    layer_effects = result["layer_patch_effects"]
    assert "8" in layer_effects   # Layer 8 is the primary comparator


# ── Logit Lens Pipeline ───────────────────────────────────────────────────────

def test_logit_lens_pipeline_runs():
    pipeline = LogitLensPipeline(mock_mode=True)
    result = pipeline.run()
    assert result["pipeline"] == "LogitLensPipeline"
    metrics = result["observed_metrics"]
    assert "final_layer_top1_accuracy" in metrics
    assert "convergence_layer_ratio" in metrics


def test_logit_lens_layer_sweep():
    pipeline = LogitLensPipeline(mock_mode=True)
    result = pipeline.run()
    # Verify all prompts have layer sweeps
    for item in result["layer_results"]:
        assert len(item["layer_sweep"]) > 0
        # Final layers should predict correctly
        final = item["layer_sweep"][-1]
        assert "top_token" in final
        assert "entropy" in final


# ── SAE Pipeline ──────────────────────────────────────────────────────────────

def test_sae_pipeline_runs():
    pipeline = SAEReproductionPipeline(mock_mode=True)
    result = pipeline.run(n_features=30)
    assert result["pipeline"] == "SAEReproductionPipeline"
    assert result["total_features_analysed"] == 30
    metrics = result["observed_metrics"]
    assert "l0_sparsity" in metrics
    assert "monosemanticity_score" in metrics


def test_sae_pipeline_sparsity():
    pipeline = SAEReproductionPipeline(mock_mode=True)
    result = pipeline.run(n_features=50)
    assert result["observed_metrics"]["l0_sparsity"] > 0.50


def test_sae_top_features_monosemantic():
    pipeline = SAEReproductionPipeline(mock_mode=True)
    result = pipeline.run(n_features=50)
    top_10 = result["top_10_features"]
    assert len(top_10) == 10
    # Top features should have high monosemanticity
    assert all(f["monosemanticity_score"] > 0.5 for f in top_10)
