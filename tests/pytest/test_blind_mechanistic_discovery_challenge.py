"""Unit and integration tests for Phase 41: Blind Mechanistic Discovery Challenge."""

import pytest

from backend.discovery.blind_discovery_engine import (
    BlindDiscoveryEngine,
    BlindDiscoveryScorecard,
    BlindDiscoverySubmission,
    BlindOracleEnvironment,
)
from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief


def test_blind_oracle_encapsulation_and_sealing():
    """Verifies that the Oracle publishes a valid 64-character SHA-256 seal of hidden ground truth."""
    oracle = BlindOracleEnvironment(challenge_seed="TEST_SEED_2026")
    assert len(oracle.oracle_sealed_hash) == 64
    assert isinstance(oracle.oracle_sealed_hash, str)


def test_autonomous_blind_discovery_execution():
    """Verifies that MECH autonomously explores strictly via black-box APIs and emits a sealed submission."""
    oracle = BlindOracleEnvironment(challenge_seed="TEST_SEED_2026")
    engine = BlindDiscoveryEngine()
    submission = engine.run_autonomous_blind_discovery(oracle)

    assert isinstance(submission, BlindDiscoverySubmission)
    assert len(submission.discovered_nodes) == 4
    assert len(submission.sha256_submission_seal) == 64
    assert submission.prospective_predicted_dz > 0.0
    assert len(submission.falsified_shortcut_traps) == 2


def test_blind_oracle_unsealing_and_scoring():
    """Verifies that the Oracle unseals and scores the submission, verifying BDS >= 95.0% and Jaccard >= 0.90."""
    oracle = BlindOracleEnvironment(challenge_seed="TEST_SEED_2026")
    engine = BlindDiscoveryEngine()
    submission = engine.run_autonomous_blind_discovery(oracle)
    scorecard = engine.evaluate_blind_submission(oracle, submission)

    assert isinstance(scorecard, BlindDiscoveryScorecard)
    assert scorecard.is_challenge_passed is True
    assert scorecard.blind_discovery_score >= 0.95
    assert scorecard.topology_jaccard >= 0.90
    assert scorecard.prospective_prediction_error <= 0.05
    assert scorecard.falsification_score == 1.00
    assert scorecard.abstention_accuracy == 1.00
    assert scorecard.oracle_sha256_unsealed == oracle.oracle_sealed_hash


def test_blind_discovery_dag_certification():
    """Verifies that the verified blind submission is registered as ACTIVE_SUPPORTED in the Living Claim DAG."""
    claim_graph = ClaimDependencyGraphEngine()
    oracle = BlindOracleEnvironment(challenge_seed="TEST_SEED_2026")
    engine = BlindDiscoveryEngine(claim_graph=claim_graph)

    submission = engine.run_autonomous_blind_discovery(oracle)
    scorecard = engine.evaluate_blind_submission(oracle, submission)

    claim_id = f"CLAIM_BLIND_DISCOVERY_{submission.submission_id}"
    assert claim_id in claim_graph.claims
    assert claim_graph.claims[claim_id].belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED
    assert "Double-Blind Mechanistic Discovery Challenge Confirmed" in claim_graph.claims[claim_id].claim_statement
