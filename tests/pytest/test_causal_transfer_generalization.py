"""Unit and integration tests for Phase 48: OOD Causal Transfer Generalization."""

import pytest

from backend.discovery.causal_transfer_generalization_engine import (
    CausalTransferGeneralizationEngine,
    CausalTransferGeneralizationReport,
    HeldOutTransferEvaluationResult,
    HeldOutTransferPrediction,
    TransportabilityLawParameters,
)
from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief


def test_transportability_law_derivation():
    """Verifies that the parameterized transportability law is derived with positive weights on role/linearity and penalties on polysemantic entropy."""
    engine = CausalTransferGeneralizationEngine()
    params = engine.law_params

    assert isinstance(params, TransportabilityLawParameters)
    assert params.alpha > 0.0  # Positive weight on functional role alignment
    assert params.beta > 0.0   # Positive weight on subspace linearity
    assert params.gamma > 0.0  # Penalty on polysemantic entropy
    assert params.delta > 0.0  # Penalty on dimension mismatch
    assert params.fit_loss <= 0.05


def test_held_out_dense_llm_transfer_prediction():
    """Verifies prospective causal transfer prediction on unseen dense architectures (Gemma-2 -> Llama-3)."""
    engine = CausalTransferGeneralizationEngine()
    pred = engine.predict_held_out_transfer(
        source_model="gemma-2-9b",
        target_model="llama-3-8b",
        intervention_id="FACTUAL_RELATION_STEERING_VECTOR",
        role_alignment=0.92,
        subspace_linearity=0.88,
        poly_entropy=0.12,
        dim_mismatch=0.08,
    )

    assert isinstance(pred, HeldOutTransferPrediction)
    assert pred.is_abstained is False
    assert len(pred.sha256_sealed_prediction) == 64
    assert 0.75 <= pred.predicted_rescue <= 0.90

    eval_res = engine.execute_and_evaluate_held_out(pred, empirical_rescue=0.80)
    assert isinstance(eval_res, HeldOutTransferEvaluationResult)
    assert eval_res.is_verified is True
    assert eval_res.relative_error_pct <= 5.0


def test_non_transformer_state_space_abstention():
    """Verifies that extreme structural domain shifts (e.g. RWKV/State-Space) trigger calibrated epistemic abstention."""
    engine = CausalTransferGeneralizationEngine()
    pred = engine.predict_held_out_transfer(
        source_model="deepseek-r1-distill-qwen-8b",
        target_model="rwkv-6-state-space-7b",
        intervention_id="ATTENTION_ROUTING_INTERVENTION",
        role_alignment=0.45,
        subspace_linearity=0.30,
        poly_entropy=0.75,
        dim_mismatch=0.25,
        is_state_space_or_recurrent=True,
    )

    assert pred.is_abstained is True
    assert "RECURRENT_STATE_SPACE" in pred.abstention_reason

    eval_res = engine.execute_and_evaluate_held_out(pred, empirical_rescue=0.00)
    assert eval_res.is_verified is True


def test_generalization_dag_certification_and_boundary_declaration():
    """Verifies that the OOD generalization suite certifies with 4 declared unresolved boundaries in the Claim DAG."""
    claim_graph = ClaimDependencyGraphEngine()
    engine = CausalTransferGeneralizationEngine(claim_graph=claim_graph)

    report = engine.run_full_ood_generalization_battery()

    assert isinstance(report, CausalTransferGeneralizationReport)
    assert report.is_overall_certified is True
    assert report.mean_ood_tpa >= 0.95
    assert report.mean_ood_error_pct <= 5.0
    assert report.abstention_accuracy_pct == 100.0
    assert len(report.declared_unresolved_boundaries) == 4

    claim_id = f"CLAIM_OOD_CAUSAL_TRANSFER_{report.report_id}"
    assert claim_id in claim_graph.claims
    assert claim_graph.claims[claim_id].belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED
    assert "Explicit boundaries declared" in claim_graph.claims[claim_id].claim_statement
