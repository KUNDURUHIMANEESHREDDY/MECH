"""Unit and integration tests for Phase 46: Adversarial Cross-Authority Divergence."""

import pytest

from backend.discovery.adversarial_cross_authority_divergence_engine import (
    AdversarialCrossAuthorityDivergenceEngine,
    AdversarialDivergenceReport,
    DivergenceRegimeType,
    DivergenceSessionResult,
)
from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief


def test_concordant_regime_replicates_accurately():
    """Verifies that when authorities have identical invariant mechanisms, MECH instances agree."""
    engine = AdversarialCrossAuthorityDivergenceEngine()
    session = engine.evaluate_concordant_regime()

    assert isinstance(session, DivergenceSessionResult)
    assert session.regime_type == DivergenceRegimeType.CONCORDANT_HOMOGENEOUS
    assert session.discovery_jaccard == 1.00
    assert session.prospective_delta_sigma < 1.0
    assert session.forced_consensus_rate == 0.0
    assert session.scientific_fidelity_score >= 0.95
    assert session.epistemic_verdict == "ACCURATE_CONCORDANT_REPLICATION_CONFIRMED"


def test_adversarial_divergent_regime_detects_justified_discrepancy():
    """Verifies that when authorities implement disparate mechanisms, MECH correctly detects divergence without forced consensus."""
    engine = AdversarialCrossAuthorityDivergenceEngine()
    session = engine.evaluate_adversarial_divergent_regime()

    assert isinstance(session, DivergenceSessionResult)
    assert session.regime_type == DivergenceRegimeType.ADVERSARIAL_DIVERGENT_PARALLEL
    assert session.is_discrepancy_justified is True
    assert session.prospective_delta_sigma >= 3.0  # >= 3.0 sigma separation
    assert session.forced_consensus_rate == 0.0   # 0% forced false agreement
    assert session.scientific_fidelity_score >= 0.95
    assert session.epistemic_verdict == "JUSTIFIED_ARCHITECTURAL_DISCREPANCY_CONFIRMED"


def test_asymmetric_abstention_regime_prevents_false_generalization():
    """Verifies that an unidentifiable noise subspace triggers calibrated abstention while clean paths are certified."""
    engine = AdversarialCrossAuthorityDivergenceEngine()
    session = engine.evaluate_asymmetric_abstention_regime()

    assert isinstance(session, DivergenceSessionResult)
    assert session.regime_type == DivergenceRegimeType.ASYMMETRIC_POLYSEMANTIC_ABSTAIN
    assert session.is_discrepancy_justified is True
    assert session.forced_consensus_rate == 0.0
    assert session.scientific_fidelity_score >= 0.95
    assert session.epistemic_verdict == "ASYMMETRIC_EPISTEMIC_ABSTENTION_ENFORCED"


def test_divergence_dag_registration_and_provenance():
    """Verifies that the multi-lab divergence report is certified and registered in the Claim DAG."""
    claim_graph = ClaimDependencyGraphEngine()
    engine = AdversarialCrossAuthorityDivergenceEngine(claim_graph=claim_graph)

    report = engine.run_full_divergence_suite()

    assert isinstance(report, AdversarialDivergenceReport)
    assert report.is_overall_passed is True
    assert report.mean_scientific_fidelity_score >= 0.95
    assert report.forced_false_consensus_rate == 0.0
    assert len(report.sessions) == 3

    claim_id = f"CLAIM_JUSTIFIED_DIVERGENCE_{report.report_id}"
    assert claim_id in claim_graph.claims
    assert claim_graph.claims[claim_id].belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED
    assert "Justified Scientific Disagreement Certified" in claim_graph.claims[claim_id].claim_statement
