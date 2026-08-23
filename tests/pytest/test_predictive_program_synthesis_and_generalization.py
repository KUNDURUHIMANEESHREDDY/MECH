"""Unit and integration tests for Phase 34: Primitive-Generalization & Predictive Mechanistic Program Synthesis."""

import pytest

from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief, DependencyType
from backend.discovery.predictive_synthesis_engine import (
    EmpiricalCausalOutcome,
    MechanisticPredictionTrialResult,
    PredictedCausalOutcome,
    PredictiveMechanisticReport,
    PredictiveSynthesisEngine,
)


def test_program_synthesis_search_on_held_out_behavior():
    """Verifies that the engine synthesizes an a priori candidate program on an unseen held-out behavior."""
    engine = PredictiveSynthesisEngine()
    trial = engine.predict_unseen_behavior(
        behavior_name="heldout_multihop_geography",
        required_operations=["EXTRACT", "FACT_LOOKUP", "FACT_LOOKUP", "ROUTE", "PROJECT"],
    )

    assert isinstance(trial, MechanisticPredictionTrialResult)
    assert trial.has_knowledge_gap is False
    assert "PRIM_EXTRACT_ENTITY" in trial.synthesized_program_expr
    assert "PRIM_FACT_LOOKUP" in trial.synthesized_program_expr
    assert trial.predicted_outcome is not None
    assert len(trial.predicted_outcome.predicted_nodes) == 4


def test_prospective_circuit_prediction_and_causal_verification():
    """Verifies that predicted circuit topology and interventional rescue match empirical ground truth (Jaccard >= 0.85)."""
    engine = PredictiveSynthesisEngine()
    trial = engine.predict_unseen_behavior(
        behavior_name="heldout_inverted_ioi",
        required_operations=["EXTRACT", "SUPPRESS", "ROUTE", "PROJECT"],
    )

    assert trial.is_prediction_verified is True
    assert trial.topology_jaccard_similarity >= 0.85
    assert trial.causal_rescue_error <= 0.05
    assert "PREDICTION_VERIFIED" in trial.verdict


def test_novel_primitive_gap_detection():
    """Verifies that requiring an unknown primitive explicitly flags an Epistemic Knowledge Gap."""
    engine = PredictiveSynthesisEngine()
    trial = engine.predict_unseen_behavior(
        behavior_name="heldout_recursive_syntax_parsing",
        required_operations=["EXTRACT", "RECURSIVE_TREE_PARSER", "ROUTE", "PROJECT"],
    )

    assert trial.has_knowledge_gap is True
    assert trial.is_prediction_verified is True  # Guarding against false composition is verified epistemic behavior
    assert "KNOWLEDGE_GAP_FLAGGED" in trial.verdict
    assert "RECURSIVE_TREE_PARSER" in trial.missing_primitive_description


def test_predictive_verification_certificate_and_dag_integration():
    """Verifies that Predictive Verification reports generate SHA-256 seals and register claims in the DAG."""
    claim_graph = ClaimDependencyGraphEngine()
    engine = PredictiveSynthesisEngine(claim_graph=claim_graph)

    report = engine.run_predictive_evaluation_battery()

    assert isinstance(report, PredictiveMechanisticReport)
    assert len(report.sha256_seal) == 64
    assert report.mechanistic_prediction_accuracy >= 0.90
    assert report.mean_topology_jaccard >= 0.85
    assert len(report.knowledge_gaps_identified) == 1

    # Verify DAG Registration for Verified Prediction
    pred_claim_id = "CLAIM_PREDICTIVE_VERIFICATION_HELDOUT_MULTIHOP_GEOGRAPHY"
    assert pred_claim_id in claim_graph.claims
    assert claim_graph.claims[pred_claim_id].belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED

    # Verify DAG Registration for Knowledge Gap
    gap_claim_id = "CLAIM_KNOWLEDGE_GAP_HELDOUT_RECURSIVE_SYNTAX_PARSING"
    assert gap_claim_id in claim_graph.claims
    assert "Knowledge Gap" in claim_graph.claims[gap_claim_id].claim_statement
