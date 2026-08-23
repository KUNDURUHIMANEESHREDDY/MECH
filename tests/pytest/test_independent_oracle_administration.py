"""Unit and integration tests for Phase 44: Independent Oracle Administration."""

import pytest

from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief
from backend.discovery.independent_oracle_administration_engine import (
    AuthorityCommitment,
    DecoupledBlackBoxClient,
    IndependentEvaluationScorecard,
    IndependentOracleAdministrationEngine,
    IndependentOracleAuthority,
    SignedDiscoverySubmission,
)


def test_independent_authority_initialization_and_signing():
    """Verifies that the Independent Authority creates isolated challenges with HMAC-SHA256 signatures."""
    authority = IndependentOracleAuthority(authority_id="EXTERNAL_AUDITOR_01")
    commitment = authority.create_isolated_challenge(seed="CHALLENGE_SEED_ALPHA", depth=5)

    assert isinstance(commitment, AuthorityCommitment)
    assert len(commitment.sha256_commitment_hash) == 64
    assert len(commitment.authority_signature) == 64
    assert commitment.authority_id == "EXTERNAL_AUDITOR_01"


def test_zero_trust_black_box_exploration():
    """Verifies that the decoupled client isolates internal state and enables MECH exploration."""
    authority = IndependentOracleAuthority(authority_id="EXTERNAL_AUDITOR_01")
    commitment = authority.create_isolated_challenge(seed="CHALLENGE_SEED_BETA", depth=4)
    client = authority.get_isolated_client(commitment.task_token)

    assert isinstance(client, DecoupledBlackBoxClient)
    assert not hasattr(client, "_hidden_circuit_nodes")  # Strict encapsulation

    grads = client.query_gradients(prompt="Test Prompt")
    assert len(grads) > 0


def test_independent_authority_unsealing_and_scoring():
    """Verifies that the decoupled tournament achieves BDS_decoupled >= 95.0% and SE >= 0.80 bits/query."""
    authority = IndependentOracleAuthority(authority_id="EXTERNAL_AUDITOR_02")
    engine = IndependentOracleAdministrationEngine()

    scorecard = engine.run_decoupled_tournament(
        authority=authority,
        seed="DECOUPLED_EVAL_2026",
        depth=5,
    )

    assert isinstance(scorecard, IndependentEvaluationScorecard)
    assert scorecard.is_certified is True
    assert scorecard.bds_decoupled >= 0.95
    assert scorecard.discovery_efficiency_bits_per_query >= 0.80
    assert scorecard.fcr_pct == 0.0
    assert scorecard.prospective_prediction_error_pct <= 5.0
    assert len(scorecard.authority_counter_signature) == 64
    assert scorecard.signature_verification_status == "VERIFIED_AUTHENTIC"


def test_tamper_proofing_and_dag_dual_signature_audit():
    """Verifies tamper rejection and dual-signed Claim DAG registration."""
    claim_graph = ClaimDependencyGraphEngine()
    authority = IndependentOracleAuthority(authority_id="EXTERNAL_AUDITOR_03")
    engine = IndependentOracleAdministrationEngine(claim_graph=claim_graph)

    # 1. Tamper test
    commitment = authority.create_isolated_challenge(seed="TAMPER_TEST_SEED", depth=4)
    invalid_sub = SignedDiscoverySubmission(
        submission_id="TAMPERED_SUBMISSION",
        task_token=commitment.task_token,
        discovered_nodes=["FAKE_NODE"],
        discovered_edges=[],
        synthesized_program_expr="FAKE",
        prospective_predicted_dz=9.99,
        falsified_shortcut_traps=[],
        abstained_subspaces=[],
        mech_signature="INVALID_SIG_TOO_SHORT",
        sha256_submission_seal="fake_seal",
        timestamp_utc="2026-08-16",
    )
    with pytest.raises(ValueError, match="TAMPERED_SUBMISSION_REJECTED"):
        authority.evaluate_and_countersign(invalid_sub)

    # 2. Genuine run & DAG verification
    scorecard = engine.run_decoupled_tournament(authority=authority, seed="GENUINE_SEED", depth=4)
    claim_id = f"CLAIM_INDEPENDENT_ORACLE_ADMINISTERED_{scorecard.task_token}"
    assert claim_id in claim_graph.claims
    assert claim_graph.claims[claim_id].belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED
    assert "Dual Signatures Verified" in claim_graph.claims[claim_id].claim_statement
