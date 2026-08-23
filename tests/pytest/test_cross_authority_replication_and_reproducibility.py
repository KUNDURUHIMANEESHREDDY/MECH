"""Unit and integration tests for Phase 45: Cross-Authority Replication & Multi-Lab Reproducibility."""

import pytest

from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief
from backend.discovery.cross_authority_replication_engine import (
    CrossAuthorityReplicationEngine,
    CrossAuthorityReplicationReport,
)
from backend.discovery.independent_oracle_administration_engine import IndependentOracleAuthority


def test_multi_authority_independent_seeding_and_isolation():
    """Verifies that two distinct authorities operate with independent signing keys and isolated state."""
    auth_1 = IndependentOracleAuthority(authority_id="LAB_OXFORD_01")
    auth_2 = IndependentOracleAuthority(authority_id="LAB_STANFORD_02")

    assert auth_1.authority_id != auth_2.authority_id
    assert auth_1._authority_secret_key != auth_2._authority_secret_key
    assert auth_1._generator.rng_seed != auth_2._generator.rng_seed


def test_decoupled_mech_instances_independent_discovery():
    """Verifies that separate MECH instances use distinct secret signing keys."""
    engine = CrossAuthorityReplicationEngine()
    assert engine._mech_1_engine._mech_secret_key != engine._mech_2_engine._mech_secret_key
    assert len(engine._mech_1_engine._mech_secret_key) == 64
    assert len(engine._mech_2_engine._mech_secret_key) == 64


def test_cross_authority_replication_and_eri_scoring():
    """Verifies that cross-authority replication achieves ERI >= 95.0% and discovery agreement >= 0.90."""
    auth_1 = IndependentOracleAuthority(authority_id="LAB_OXFORD_01")
    auth_2 = IndependentOracleAuthority(authority_id="LAB_STANFORD_02")
    engine = CrossAuthorityReplicationEngine()

    report = engine.run_cross_authority_replication(
        authority_a=auth_1,
        authority_b=auth_2,
        seed_a="OXFORD_TASK_SEED_2026",
        seed_b="STANFORD_TASK_SEED_2026",
        depth=5,
    )

    assert isinstance(report, CrossAuthorityReplicationReport)
    assert report.is_replicated is True
    assert report.epistemic_replicability_index >= 0.95
    assert report.discovery_agreement_jaccard >= 0.90
    assert report.predictive_agreement_error_pct <= 5.0
    assert report.falsification_concordance_pct == 100.0
    assert report.abstention_concordance_pct == 100.0


def test_multi_authority_consensus_dag_quorum():
    """Verifies that the multi-authority claim is registered in the Claim DAG with a 4-party quorum."""
    claim_graph = ClaimDependencyGraphEngine()
    auth_1 = IndependentOracleAuthority(authority_id="LAB_OXFORD_01")
    auth_2 = IndependentOracleAuthority(authority_id="LAB_STANFORD_02")
    engine = CrossAuthorityReplicationEngine(claim_graph=claim_graph)

    report = engine.run_cross_authority_replication(
        authority_a=auth_1,
        authority_b=auth_2,
        seed_a="OXFORD_DAG_SEED",
        seed_b="STANFORD_DAG_SEED",
        depth=4,
    )

    claim_id = f"CLAIM_MULTI_AUTHORITY_REPLICATION_{report.report_id}"
    assert claim_id in claim_graph.claims
    assert claim_graph.claims[claim_id].belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED

    # Check 4-party quorum signatures
    sigs = report.quorum_signatures
    assert len(sigs) == 4
    for key, sig in sigs.items():
        assert len(sig) == 64
        assert isinstance(sig, str)

    assert "4-Party Quorum Verified" in claim_graph.claims[claim_id].claim_statement
