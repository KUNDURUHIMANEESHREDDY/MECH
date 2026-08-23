"""Unit and integration tests for Phase 53: Multi-Lab Independent Replication & Cross-Institutional Epistemic Quorum."""

import pytest

from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief
from backend.discovery.multi_lab_replication_engine import (
    LabReplicationReport,
    MultiLabReplicationEngine,
    MultiLabReplicationQuorum,
)
from backend.discovery.portable_external_replication_package import EpistemicValidationTier


def test_multi_lab_protocol_isolation_and_challenge_diversity():
    """Verifies that 3 distinct independent laboratories evaluate diverse, non-overlapping architectural regimes."""
    engine = MultiLabReplicationEngine()
    quorum = engine.execute_multi_lab_federated_challenge()

    assert len(quorum.participating_labs) == 3
    assert len(quorum.lab_reports) == 3

    lab_ids = [r.lab_id for r in quorum.lab_reports]
    assert "LAB_BERKELEY" in lab_ids
    assert "LAB_OXFORD" in lab_ids
    assert "LAB_STANFORD" in lab_ids


def test_multi_lab_raw_intervention_verification_and_predictions():
    """Verifies that all 3 labs return valid raw measurement manifests and achieve <= 5% error on continuous outcomes."""
    engine = MultiLabReplicationEngine()
    quorum = engine.execute_multi_lab_federated_challenge()

    for report in quorum.lab_reports:
        assert len(report.raw_measurement_manifest_hash) == 64
        assert len(report.lab_signature) == 64
        assert report.mean_error_pct <= 5.0
        assert report.abstention_accuracy_pct == 100.0


def test_cross_lab_heterogeneity_and_quorum_scoring():
    """Verifies that cross-laboratory heterogeneity Delta_hetero <= 0.05 and ML-ERI >= 95.0%."""
    engine = MultiLabReplicationEngine()
    quorum = engine.execute_multi_lab_federated_challenge()

    assert isinstance(quorum, MultiLabReplicationQuorum)
    assert quorum.is_quorum_certified is True
    assert quorum.ml_eri_score >= 0.95
    assert quorum.cross_lab_heterogeneity <= 0.05
    assert quorum.mean_multi_lab_error_pct <= 5.0
    assert len(quorum.multi_party_signatures) == 4


def test_dag_tier_promotion_to_multi_lab_consensus():
    """Verifies that unanimous 4-party quorum promotes the claim strictly to MULTI_LAB_CONSENSUS tier in the Claim DAG."""
    claim_graph = ClaimDependencyGraphEngine()
    engine = MultiLabReplicationEngine(claim_graph=claim_graph)

    quorum = engine.execute_multi_lab_federated_challenge()

    assert quorum.promoted_epistemic_tier == EpistemicValidationTier.MULTI_LAB_CONSENSUS

    claim_id = f"CLAIM_MULTI_LAB_CONSENSUS_{quorum.quorum_id}"
    assert claim_id in claim_graph.claims
    assert claim_graph.claims[claim_id].belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED
    assert "MULTI_LAB_CONSENSUS" in claim_graph.claims[claim_id].claim_statement
