"""Numerical PyTest Suite: Controlled Causal Interventions & 4-Control Battery."""

import pytest
from backend.science.controlled_causal_engine import ControlledCausalEngine


def test_controlled_causal_intervention_suite():
    engine = ControlledCausalEngine(model_id="gpt2")
    meas = engine.evaluate_component_causality(
        prompt="The capital of France is",
        target_token=" Paris",
        layer=8,
        component_type="neuron",
        component_index=412,
    )

    assert meas.target_component == "L8_N412"
    assert meas.layer == 8
    assert meas.index == 412

    # Invariants on probabilities and logits
    assert 0.0 <= meas.clean_probability <= 1.0
    assert 0.0 <= meas.intervened_probability <= 1.0
    assert meas.clean_rank >= 1
    assert meas.intervened_rank >= 1

    # 4-Control Battery Invariants
    assert len(meas.controls) == 4
    for ctrl in meas.controls:
        assert ctrl.control_type in ["matched_norm", "same_layer", "same_mechanism", "random_global"]
        assert isinstance(ctrl.delta_logit, float)

    # Specificity Ratio is non-negative
    assert meas.robust_specificity_ratio >= 0.0
    assert meas.evidence_tier in ["CAUSALLY_VERIFIED", "SUPPORTED", "CANDIDATE", "FALSIFIED"]
    assert len(meas.verdict) > 10
