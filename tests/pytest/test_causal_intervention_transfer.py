"""Unit and integration tests for Phase 47: Causal Intervention Portability."""

import pytest

from backend.discovery.causal_intervention_transfer_engine import (
    CausalInterventionTransferEngine,
    CausalTransferEvaluationResult,
    CausalTransferPrediction,
    CausalTransferReport,
    TransferPortabilityClass,
)
from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief


def test_full_causal_transfer_prediction_and_verification():
    """Verifies that high-invariant circuits (GPT-2 -> Pythia) achieve full causal transfer with error <= 5%."""
    engine = CausalInterventionTransferEngine()
    pred = engine.derive_and_seal_transfer_prediction(
        source_model="gpt2-small",
        target_model="pythia-2.8b",
        intervention_id="IOI_NAME_MOVER_STEERING_VECTOR",
    )

    assert isinstance(pred, CausalTransferPrediction)
    assert pred.predicted_class == TransferPortabilityClass.FULL_TRANSFER
    assert len(pred.sha256_sealed_prediction) == 64

    eval_res = engine.execute_and_evaluate_transfer(pred)
    assert isinstance(eval_res, CausalTransferEvaluationResult)
    assert eval_res.is_overall_verified is True
    assert eval_res.empirical_rescue_ratio >= 0.80
    assert eval_res.relative_error_pct <= 5.0


def test_partial_causal_transfer_and_rerouting_explanation():
    """Verifies that re-routed circuits (Qwen -> Mistral) achieve partial transfer with explicit remapping attribution."""
    engine = CausalInterventionTransferEngine()
    pred = engine.derive_and_seal_transfer_prediction(
        source_model="qwen-2.5-7b",
        target_model="mistral-7b",
        intervention_id="INVERTED_INHIBITION_PATCH",
    )

    assert pred.predicted_class == TransferPortabilityClass.PARTIAL_TRANSFER
    assert "SUBSTRATE_REROUTING" in pred.mechanistic_explanation

    eval_res = engine.execute_and_evaluate_transfer(pred)
    assert eval_res.is_class_accurate is True
    assert 0.35 <= eval_res.empirical_rescue_ratio < 0.80
    assert eval_res.explanation_fidelity_score >= 0.95


def test_negative_transfer_prediction_and_failure_attribution():
    """Verifies that polysemantic superposition substrates trigger negative transfer with superposition attribution."""
    engine = CausalInterventionTransferEngine()
    pred = engine.derive_and_seal_transfer_prediction(
        source_model="gpt2-small",
        target_model="synthetic-mha-superposition",
        intervention_id="SPARSE_CIRCUIT_CLAMP",
    )

    assert pred.predicted_class == TransferPortabilityClass.NEGATIVE_TRANSFER_FAILURE
    assert "SUPERPOSITION_COLLAPSE_INTERFERENCE" in pred.mechanistic_explanation

    eval_res = engine.execute_and_evaluate_transfer(pred)
    assert eval_res.is_class_accurate is True
    assert eval_res.empirical_rescue_ratio < 0.35
    assert eval_res.relative_error_pct <= 5.0


def test_causal_transfer_dag_certification_and_lineage():
    """Verifies that the entire 3-regime battery is certified and registered into the Claim DAG."""
    claim_graph = ClaimDependencyGraphEngine()
    engine = CausalInterventionTransferEngine(claim_graph=claim_graph)

    report = engine.run_full_transferability_suite()

    assert isinstance(report, CausalTransferReport)
    assert report.is_overall_certified is True
    assert report.mean_ctpa >= 0.95
    assert report.mean_relative_error_pct <= 5.0
    assert report.failure_attribution_fidelity_pct >= 95.0
    assert len(report.evaluations) == 3

    claim_id = f"CLAIM_CAUSAL_INTERVENTION_TRANSFER_{report.report_id}"
    assert claim_id in claim_graph.claims
    assert claim_graph.claims[claim_id].belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED
    assert "Cross-Substrate Causal Intervention Portability Certified" in claim_graph.claims[claim_id].claim_statement
