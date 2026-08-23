"""Numerical PyTest Suite: Cross-Prompt Causal Robustness & Distributional Statistics."""

import pytest
from backend.science.controlled_causal_engine import ControlledCausalEngine


def test_cross_prompt_causality_suite():
    engine = ControlledCausalEngine(model_id="gpt2", default_seed=42)
    prompts = [
        ("The capital of France is", " Paris"),
        ("The Eiffel Tower is located in", " Paris"),
        ("The Louvre Museum is in", " Paris"),
    ]

    report = engine.evaluate_cross_prompt_causality(
        prompts=prompts,
        layer=8,
        component_type="neuron",
        component_index=412,
        seed=42,
    )

    assert report.target_component == "L8_N412"
    assert report.prompt_count == 3
    assert len(report.prompt_evaluations) == 3

    # Robust Distributional Statistics
    assert isinstance(report.mean_delta_logit, float)
    assert isinstance(report.median_delta_logit, float)
    assert report.std_delta_logit >= 0.0
    assert report.iqr_delta_logit >= 0.0
    assert report.ci_95_lower <= report.ci_95_upper
    assert 0.0 <= report.expected_sign_rate <= 1.0
    assert 0.0 <= report.mediation_fraction <= 1.0
    assert report.mean_specificity_ratio >= 0.0

    # Epistemic Scope Disclosures
    assert isinstance(report.promotion_reasons, list)
    assert len(report.evidence_scope) >= 1
    assert len(report.remaining_limitations) >= 1
    assert report.overall_evidence_tier in ["CAUSALLY_VERIFIED", "SUPPORTED", "CANDIDATE", "FALSIFIED"]
    assert len(report.falsification_summary) > 10

    # Check Deterministic Control Seeds in evaluations
    for idx, eval_res in enumerate(report.prompt_evaluations):
        assert eval_res.control_selection_seed == 42 + idx
        assert len(eval_res.controls) == 4
        for ctrl in eval_res.controls:
            assert len(ctrl.selection_rationale) > 5
