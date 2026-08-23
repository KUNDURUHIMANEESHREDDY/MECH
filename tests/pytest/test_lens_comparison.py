"""Integration Test: Lens Comparison Workflow.

Validates: Residual -> Logit Lens -> Tuned Lens -> Compare Projections.
"""
from backend.api.legacy_dispatcher import build_dispatcher


def test_lens_comparison_workflow():
    dispatcher = build_dispatcher()
    prompt = "The capital of France is Paris"

    # 1. Project through Logit Lens
    logit_res = dispatcher["interpretability/projections/logit_lens"]({"prompt": prompt, "layer": 10})
    assert logit_res["method"] == "LogitLens"

    # 2. Project through Tuned Lens
    tuned_res = dispatcher["interpretability/projections/tuned_lens"]({"prompt": prompt, "layer": 10})
    assert tuned_res["method"] == "TunedLens"

    # 3. Verify projection comparison
    assert logit_res["layer"] == tuned_res["layer"]
    assert tuned_res["prediction_confidence"] >= logit_res["top_k_tokens"][0]["probability"]
