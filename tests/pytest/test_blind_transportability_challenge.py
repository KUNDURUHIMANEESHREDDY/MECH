"""Unit and integration tests for Phase 49: Blind Cross-Architecture Transportability Challenge."""

import pytest

from backend.discovery.blind_transportability_engine import (
    BlindArchitectureDescriptor,
    BlindTransportabilityEngine,
    BlindTransportabilityReport,
    BlindTransportEvaluationResult,
    BlindTransportPrediction,
    FrozenTransportabilityLaw,
    IndependentBlindTransportAuthority,
)
from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief


def test_frozen_law_immutability_and_hash_sealing():
    """Verifies that the canonical transportability law is frozen and its SHA-256 seal is immutable."""
    frozen_law = FrozenTransportabilityLaw.create_canonical_frozen_law()
    assert len(frozen_law.sha256_law_hash) == 64
    assert frozen_law.law_params.alpha == 1.40
    assert frozen_law.law_params.beta == 1.20
    assert frozen_law.law_params.gamma == 1.80
    assert frozen_law.law_params.delta == 1.20
    assert frozen_law.law_params.bias == -0.60


def test_zero_knowledge_blind_predictions():
    """Verifies that MECH receives only structural probe descriptors without empirical ground truths."""
    auth = IndependentBlindTransportAuthority()
    probes = auth.get_blind_probe_descriptors()

    assert len(probes) == 4
    for p in probes:
        assert "arch_id" in p
        assert "role_alignment" in p
        assert "empirical_true_r" not in p  # Zero knowledge isolation

    engine = BlindTransportabilityEngine()
    pred = engine.predict_blind_architecture(probes[0])
    assert isinstance(pred, BlindTransportPrediction)
    assert len(pred.sha256_sealed_prediction) == 64


def test_blind_transportability_unsealing_and_bts_scoring():
    """Verifies that unsealing the 4 blind architectures achieves BTS >= 95.0% and relative error <= 5.0%."""
    auth = IndependentBlindTransportAuthority()
    engine = BlindTransportabilityEngine()

    report = engine.run_blind_transportability_challenge(authority=auth)

    assert isinstance(report, BlindTransportabilityReport)
    assert report.is_overall_certified is True
    assert report.bts_score >= 0.95
    assert report.mean_blind_error_pct <= 5.0
    assert report.abstention_accuracy_pct == 100.0
    assert len(report.evaluations) == 4


def test_blind_transportability_dag_certification_and_lineage():
    """Verifies that the Blind Transportability claim is registered in the Claim DAG."""
    claim_graph = ClaimDependencyGraphEngine()
    engine = BlindTransportabilityEngine(claim_graph=claim_graph)

    report = engine.run_blind_transportability_challenge()

    claim_id = f"CLAIM_BLIND_TRANSPORTABILITY_{report.report_id}"
    assert claim_id in claim_graph.claims
    assert claim_graph.claims[claim_id].belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED
    assert "Blind Cross-Architecture Transportability Challenge Certified" in claim_graph.claims[claim_id].claim_statement
