"""Unit and integration tests for Phase 50: Real-Model Blind Transportability Challenge."""

import pytest

from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief
from backend.discovery.real_model_transportability_engine import (
    IndependentRealModelOracle,
    RealModelBlindDescriptor,
    RealModelTransportabilityEngine,
    RealModelTransportabilityReport,
    RealModelTransportEvaluationResult,
    RealModelTransportPrediction,
)


def test_real_model_oracle_encapsulation_and_blindness():
    """Verifies that the Independent Real-Model Oracle exposes structural probes with zero identity or empirical leakage."""
    oracle = IndependentRealModelOracle()
    probes = oracle.get_real_model_blind_probes()

    assert len(probes) == 4
    for p in probes:
        assert "real_model_id" in p
        assert "role_alignment" in p
        assert "empirical_true_r" not in p  # Strict zero-knowledge isolation


def test_dense_and_moe_real_model_transfer_predictions():
    """Verifies that the frozen law accurately predicts causal transfer on real Dense LLM and Sparse MoE models within <= 5% error."""
    oracle = IndependentRealModelOracle()
    engine = RealModelTransportabilityEngine()
    probes = oracle.get_real_model_blind_probes()

    # Dense LLM probe
    dense_probe = [p for p in probes if "ALPHA" in p["real_model_id"]][0]
    pred_dense = engine.predict_real_model(dense_probe)
    res_dense = oracle.unseal_and_evaluate_real_model_prediction(pred_dense)

    assert res_dense.is_overall_verified is True
    assert res_dense.relative_error_pct <= 5.0
    assert res_dense.empirical_r >= 0.80

    # Sparse MoE probe
    moe_probe = [p for p in probes if "BETA" in p["real_model_id"]][0]
    pred_moe = engine.predict_real_model(moe_probe)
    res_moe = oracle.unseal_and_evaluate_real_model_prediction(pred_moe)

    assert res_moe.is_overall_verified is True
    assert res_moe.relative_error_pct <= 5.0
    assert 0.50 <= res_moe.empirical_r < 0.80


def test_real_mamba_ssm_abstention_and_negative_transfer_detection():
    """Verifies that real Mamba-2 SSM triggers calibrated abstention and real polysemantic code triggers negative transfer detection."""
    oracle = IndependentRealModelOracle()
    engine = RealModelTransportabilityEngine()
    probes = oracle.get_real_model_blind_probes()

    # Mamba-2 SSM probe
    ssm_probe = [p for p in probes if "GAMMA" in p["real_model_id"]][0]
    pred_ssm = engine.predict_real_model(ssm_probe)
    res_ssm = oracle.unseal_and_evaluate_real_model_prediction(pred_ssm)

    assert pred_ssm.is_abstained is True
    assert res_ssm.is_abstention_accurate is True
    assert res_ssm.is_overall_verified is True

    # Polysemantic Code probe
    code_probe = [p for p in probes if "DELTA" in p["real_model_id"]][0]
    pred_code = engine.predict_real_model(code_probe)
    res_code = oracle.unseal_and_evaluate_real_model_prediction(pred_code)

    assert pred_code.is_negative_transfer_predicted is True
    assert res_code.is_negative_transfer_detected is True
    assert res_code.relative_error_pct <= 5.0
    assert res_code.is_overall_verified is True


def test_real_model_challenge_dag_certification_and_lineage():
    """Verifies that the entire real-model challenge passes with TES >= 95% and registers into the Claim DAG."""
    claim_graph = ClaimDependencyGraphEngine()
    engine = RealModelTransportabilityEngine(claim_graph=claim_graph)

    report = engine.run_real_model_challenge()

    assert isinstance(report, RealModelTransportabilityReport)
    assert report.is_overall_certified is True
    assert report.tes_score >= 0.95
    assert report.mean_real_error_pct <= 5.0
    assert report.abstention_accuracy_pct == 100.0
    assert len(report.evaluations) == 4

    claim_id = f"CLAIM_REAL_MODEL_TRANSPORTABILITY_{report.report_id}"
    assert claim_id in claim_graph.claims
    assert claim_graph.claims[claim_id].belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED
    assert "Real-Model Blind Transportability Certified" in claim_graph.claims[claim_id].claim_statement
