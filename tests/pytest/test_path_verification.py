"""Numerical PyTest Suite: End-to-End Pathway Verification, Mediation Rescue & Null Model."""

import pytest
from backend.science.path_verification_engine import PathVerificationEngine


def test_path_verification_engine_e2e_and_rescue():
    engine = PathVerificationEngine(model_id="gpt2")
    report = engine.verify_pathway(
        clean_prompt="The capital of France is",
        target_token=" Paris",
        pathway_id="test_pathway_france",
        layer=8,
        seed=42,
    )

    assert report.pathway_id == "test_pathway_france"
    assert len(report.node_chain) >= 3
    assert len(report.edge_chain) >= 2
    assert len(report.step_measurements) == 6

    # 1. Step measurements integrity
    for step in report.step_measurements:
        assert isinstance(step.step_name, str)
        assert isinstance(step.delta_logit, float)
        assert isinstance(step.delta_prob, float)
        assert len(step.description) > 5

    # 2. Formal Mediation Rescue Experiment Invariants
    rescue = report.mediation_rescue
    assert rescue.source_node == report.node_chain[0]
    assert rescue.mediator_node == report.node_chain[2]
    assert isinstance(rescue.clean_source_activation, float)
    assert isinstance(rescue.clean_mediator_activation, float)
    assert isinstance(rescue.rescued_logit, float)
    assert isinstance(rescue.rescue_delta_recovery, float)
    assert 0.0 <= rescue.rescue_fraction <= 1.0
    assert rescue.formal_mediation_status in [
        "CONFIRMED_CAUSAL_MEDIATOR",
        "PARTIAL_RESCUE",
        "BYSTANDER_NON_MEDIATING",
    ]
    assert len(rescue.rescue_verdict) > 10

    # 3. Empirical Null Distribution Statistics Invariants
    null_dist = report.null_distribution
    assert null_dist.control_path_count == 8
    assert len(null_dist.control_path_deltas) == 8
    assert isinstance(null_dist.mean_null_delta, float)
    assert isinstance(null_dist.median_null_delta, float)
    assert isinstance(null_dist.max_null_delta, float)
    assert 0.0 <= null_dist.observed_path_percentile <= 100.0
    assert 0.0 <= null_dist.empirical_p_value <= 1.0

    # 4. Overall Path Status
    assert report.path_causal_status in [
        "END_TO_END_VERIFIED",
        "PARTIALLY_MEDIATED",
        "NON_MEDIATING",
        "FALSIFIED",
    ]
    assert len(report.path_verdict) > 10
    assert len(report.epistemic_scope) >= 1
