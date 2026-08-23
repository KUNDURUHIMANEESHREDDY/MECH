"""Unit and integration tests for Semantic Mechanism Falsification Probes."""

import pytest
from backend.science.probes.semantic_falsification import (
    MLPMemoryTupleProbe,
    AttentionHeadInductionProbe,
    SemanticFalsificationSuite,
    TupleSteeringResult,
    InductionProbeResult,
)
from backend.core.capability_registry import CapabilityRegistry
from backend.research_platform.autonomous.ai_research_assistant import AIResearchAssistant


def test_mlp_memory_tuple_probe():
    probe = MLPMemoryTupleProbe()
    
    # 1. Mid-layer MLP should verify relational direction transfer
    res_mid = probe.test_relational_tuple_steering(
        component="L6_MLP",
        source_tuple=("Eiffel Tower", "located in", " Paris"),
        target_tuple=("Colosseum", "located in", " Rome"),
    )
    assert isinstance(res_mid, TupleSteeringResult)
    assert res_mid.is_verified_relational_mediator is True
    assert res_mid.steering_efficiency > 0.60
    assert "Verified Relational Mediator" in res_mid.verdict
    assert "Steering >= 60%" in res_mid.interpretation_note

    # 2. Early-layer MLP (e.g. L1_MLP) should fail relational tuple steering
    res_early = probe.test_relational_tuple_steering(
        component="L1_MLP",
        source_tuple=("Eiffel Tower", "located in", " Paris"),
        target_tuple=("Colosseum", "located in", " Rome"),
    )
    assert res_early.is_verified_relational_mediator is False
    assert "Falsified" in res_early.verdict


def test_attention_head_induction_probe():
    probe = AttentionHeadInductionProbe()
    
    # 1. Canonical induction head (e.g. L8_H5) should verify 6-facet protocol
    res_ind = probe.test_prefix_matching_induction(component="L8_H5", n_sequences=20, seq_len=16)
    assert isinstance(res_ind, InductionProbeResult)
    assert res_ind.is_verified_induction_head is True
    assert res_ind.composite_induction_score >= 0.70
    assert res_ind.uniform_enrichment_ratio > 1.0  # Fold change over uniform baseline (1/L)
    
    # Check all 6 facets
    facets = res_ind.facets
    assert facets.prefix_match_attention > 0.50
    assert facets.copy_position_attention > 0.50
    assert facets.next_token_logit_uplift > 2.0
    assert facets.synthetic_copy_accuracy > 0.70
    assert facets.positional_control_pass is True
    assert facets.causal_ablation_drop_pct > 50.0

    assert "6-Facet Protocol Supported" in res_ind.verdict

    # 2. Non-induction head (e.g. L0_H0) should fail induction criteria
    res_non = probe.test_prefix_matching_induction(component="L0_H0", n_sequences=20, seq_len=16)
    assert res_non.is_verified_induction_head is False
    assert "Falsified" in res_non.verdict


def test_semantic_falsification_suite_and_registry_tool():
    registry = CapabilityRegistry()
    assert registry.get_tool("run_semantic_falsification_probe") is not None

    res = registry.execute_tool("run_semantic_falsification_probe", {"component": "L6_MLP"})
    assert res["status"] == "success"
    assert "probe_report" in res
    assert "Relational Direction Transfer Probe" in res["probe_report"]["probe_type"]

    res_head = registry.execute_tool("run_semantic_falsification_probe", {"component": "L8_H5"})
    assert res_head["status"] == "success"
    assert "6-Facet Induction Evidence Suite" in res_head["probe_report"]["probe_type"]

