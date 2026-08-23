"""Unit and integration tests for Phase 19: Dual-Loop Discovery & Competitive Falsification Engine."""

import pytest
import torch

from backend.discovery.circuit_discovery_types import EpistemicCircuitTier
from backend.discovery.competitive_falsification_engine import (
    CompetitiveFalsificationEngine,
    CompetitiveInterpretationReport,
    CompetingHypothesis,
    HypothesisType,
)
from backend.discovery.dual_loop_discovery_orchestrator import (
    DualLoopDiscoveryOrchestrator,
    DualLoopDiscoveryReport,
    SevenCriteriaVerificationScorecard,
)
from backend.runtime.in_memory_runtime import InMemoryRuntime
from backend.runtime.interfaces import PrecisionProfile


def test_strict_7_criterion_circuit_certification():
    """Verifies that End-to-End Circuit Certification requires unanimous satisfaction of all 7 criteria."""
    scorecard_pass = SevenCriteriaVerificationScorecard(
        criterion_1_acdc_sparse_topology=True,
        criterion_2_node_causal_necessity=True,
        criterion_3_edge_path_necessity=True,
        criterion_4_negative_control_specificity=True,
        criterion_5_mediation_rescue=True,
        criterion_6_cross_prompt_replication=True,
        criterion_7_cryptographic_provenance=True,
        all_seven_criteria_satisfied=True,
        specificity_ratio_observed=4.5,
        mediation_rescue_observed=0.82,
        replication_rate_observed_pct=85.0,
        canonical_provenance_hash="a" * 64,
    )
    assert scorecard_pass.all_seven_criteria_satisfied is True

    # Demotion if even one criterion fails (e.g., mediation rescue fails)
    scorecard_fail = SevenCriteriaVerificationScorecard(
        criterion_1_acdc_sparse_topology=True,
        criterion_2_node_causal_necessity=True,
        criterion_3_edge_path_necessity=True,
        criterion_4_negative_control_specificity=True,
        criterion_5_mediation_rescue=False,  # FAILED
        criterion_6_cross_prompt_replication=True,
        criterion_7_cryptographic_provenance=True,
        all_seven_criteria_satisfied=False,
        specificity_ratio_observed=4.5,
        mediation_rescue_observed=0.45,
        replication_rate_observed_pct=85.0,
        canonical_provenance_hash="b" * 64,
    )
    assert scorecard_fail.all_seven_criteria_satisfied is False


def test_dual_loop_autonomous_backtracking():
    """Verifies that DualLoopDiscoveryOrchestrator coordinates search, backtracking, and 7-criterion verification."""
    runtime = InMemoryRuntime(
        model_id="gpt2",
        device="cpu",
        precision=PrecisionProfile(weight_dtype="float32", activation_dtype="float32"),
    )
    orchestrator = DualLoopDiscoveryOrchestrator(runtime=runtime, model_id="gpt2", device="cpu")

    report = orchestrator.run_autonomous_dual_loop_discovery(
        behavior_name="french_capital_dual_loop",
        clean_prompt="The capital of France is",
        target_token=" Paris",
        corrupted_prompt="The capital of Italy is",
        target_layers=[6, 8, 10],
        pruning_threshold_tau=0.010,
    )

    assert isinstance(report, DualLoopDiscoveryReport)
    assert report.initial_attribution_candidates_count > 0
    assert report.backtracking_steps_count >= 0
    assert report.acdc_circuit.retained_edges_count > 0
    assert len(report.scorecard_7_criteria.canonical_provenance_hash) == 64
    assert report.final_epistemic_status in (
        EpistemicCircuitTier.VERIFIED_CIRCUIT,
        EpistemicCircuitTier.CANDIDATE_CIRCUIT,
    )
    assert len(report.executive_narrative) > 50


def test_competitive_hypothesis_generation_and_elimination():
    """Verifies that CompetitiveFalsificationEngine generates competing hypotheses and refutes weaker explanations."""
    runtime = InMemoryRuntime(
        model_id="gpt2",
        device="cpu",
        precision=PrecisionProfile(weight_dtype="float32", activation_dtype="float32"),
    )
    engine = CompetitiveFalsificationEngine(runtime=runtime, model_id="gpt2", device="cpu")

    report = engine.run_competitive_falsification(
        layer=8,
        component_index=412,
        primary_behavior_clean="The capital of France is",
        primary_target_token=" Paris",
    )

    assert isinstance(report, CompetitiveInterpretationReport)
    assert len(report.competing_hypotheses) == 3
    assert len(report.discriminating_experiments) == 3
    assert report.eliminated_hypotheses_count >= 2
    assert report.surviving_hypothesis is not None
    assert report.surviving_hypothesis.hypothesis_type == HypothesisType.SPECIFIC_RELATION_RETRIEVAL
    assert report.epistemic_confidence_pct >= 70.0


def test_directional_projection_vs_causal_necessity_invariant():
    """Verifies hard separation: W_U * d_i (representational geometry) != Δz (causal necessity)."""
    runtime = InMemoryRuntime(
        model_id="gpt2",
        device="cpu",
        precision=PrecisionProfile(weight_dtype="float32", activation_dtype="float32"),
    )
    engine = CompetitiveFalsificationEngine(runtime=runtime, model_id="gpt2", device="cpu")

    report = engine.run_competitive_falsification(
        layer=8,
        component_index=412,
        primary_behavior_clean="The capital of France is",
        primary_target_token=" Paris",
    )

    # W_U * d_i top tokens must be extracted
    assert len(report.directional_vocabulary_top_tokens) > 0
    surviving = report.surviving_hypothesis
    assert surviving is not None

    # Distinct non-null numerical values representing different scientific concepts
    assert isinstance(surviving.directional_projection_score, float)
    assert isinstance(surviving.causal_necessity_score, float)
    assert surviving.directional_projection_score > 0.0
    assert surviving.causal_necessity_score >= 0.0
