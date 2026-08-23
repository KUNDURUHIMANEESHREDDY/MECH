"""Unit and integration tests for Phase 51: Independent External Third-Party Evaluation Authority & Multi-Lab Causal Challenge."""

import pytest

from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief
from backend.discovery.external_evaluation_authority_engine import (
    DecoupledExternalAuthority,
    ExternalChallengeEvaluationResult,
    ExternalChallengePrediction,
    ExternalChallengeReport,
    ExternalEvaluationAuthorityEngine,
    ExternalModelTaskChallenge,
)


def test_external_authority_isolation_and_challenge_generation():
    """Verifies that the Decoupled External Authority generates uncontrolled challenges with zero leakage."""
    auth = DecoupledExternalAuthority()
    probes = auth.get_external_blind_probe_descriptors()

    assert len(probes) == 4
    for p in probes:
        assert "challenge_id" in p
        assert "probe_role" in p
        assert "empirical_true_r" not in p
        assert "model_name" not in p
        assert "task_name" not in p


def test_external_challenge_dense_and_moe_predictions():
    """Verifies that MECH predicts transfer on external Qwen-2.5 and DeepSeek MoE challenges within <= 5% error."""
    auth = DecoupledExternalAuthority()
    engine = ExternalEvaluationAuthorityEngine()
    probes = auth.get_external_blind_probe_descriptors()

    # Dense Transformer (Qwen-2.5) probe
    dense_probe = [p for p in probes if "QWEN25" in p["challenge_id"]][0]
    pred_dense = engine.predict_external_challenge(dense_probe)
    res_dense = auth.unseal_and_evaluate_external_prediction(pred_dense)

    assert res_dense.is_overall_verified is True
    assert res_dense.relative_error_pct <= 5.0
    assert res_dense.empirical_r >= 0.80

    # Sparse MoE (DeepSeek-V2) probe
    moe_probe = [p for p in probes if "DEEPSEEK" in p["challenge_id"]][0]
    pred_moe = engine.predict_external_challenge(moe_probe)
    res_moe = auth.unseal_and_evaluate_external_prediction(pred_moe)

    assert res_moe.is_overall_verified is True
    assert res_moe.relative_error_pct <= 5.0
    assert 0.50 <= res_moe.empirical_r < 0.80


def test_external_challenge_ssm_abstention_and_negative_transfer():
    """Verifies that external RWKV-5 SSM triggers calibrated abstention and CodeLlama triggers negative transfer detection."""
    auth = DecoupledExternalAuthority()
    engine = ExternalEvaluationAuthorityEngine()
    probes = auth.get_external_blind_probe_descriptors()

    # RWKV-5 SSM probe
    ssm_probe = [p for p in probes if "RWKV5" in p["challenge_id"]][0]
    pred_ssm = engine.predict_external_challenge(ssm_probe)
    res_ssm = auth.unseal_and_evaluate_external_prediction(pred_ssm)

    assert pred_ssm.is_abstained is True
    assert res_ssm.is_abstention_accurate is True
    assert res_ssm.is_overall_verified is True

    # CodeLlama Polysemantic Code probe
    code_probe = [p for p in probes if "CODELLAMA" in p["challenge_id"]][0]
    pred_code = engine.predict_external_challenge(code_probe)
    res_code = auth.unseal_and_evaluate_external_prediction(pred_code)

    assert pred_code.is_negative_transfer_predicted is True
    assert res_code.is_negative_transfer_detected is True
    assert res_code.relative_error_pct <= 5.0
    assert res_code.is_overall_verified is True


def test_external_challenge_dual_signature_quorum_and_dag():
    """Verifies that the External Challenge generates a dual-signed quorum report and registers into the Claim DAG."""
    claim_graph = ClaimDependencyGraphEngine()
    engine = ExternalEvaluationAuthorityEngine(claim_graph=claim_graph)

    report = engine.run_external_challenge()

    assert isinstance(report, ExternalChallengeReport)
    assert report.is_overall_certified is True
    assert report.eri_score >= 0.95
    assert report.tes_score >= 0.95
    assert report.mean_ext_error_pct <= 5.0
    assert report.abstention_accuracy_pct == 100.0
    assert len(report.auth_sig_report) == 64
    assert len(report.mech_sig_report) == 64

    claim_id = f"CLAIM_EXTERNAL_CHALLENGE_{report.report_id}"
    assert claim_id in claim_graph.claims
    assert claim_graph.claims[claim_id].belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED
    assert "External Third-Party Replication Quorum Certified" in claim_graph.claims[claim_id].claim_statement
