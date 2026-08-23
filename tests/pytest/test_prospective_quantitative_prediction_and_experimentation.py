"""Unit and integration tests for Phase 38: Autonomous Prospective Experimentation & Novel Quantitative Prediction."""

import pytest

from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief
from backend.discovery.prospective_prediction_engine import (
    ProspectiveExperimentProtocol,
    ProspectivePredictionCertificate,
    ProspectivePredictionEngine,
    QuantitativePredictionTarget,
)


def test_prospective_quantitative_derivation():
    """Verifies that the engine derives formal a priori quantitative predictions with standard errors."""
    engine = ProspectivePredictionEngine()
    cert = engine.derive_and_execute_prospective_experiment(
        theory_id="THEORY_MULTIHOP_RELATIONAL_REASONING_ALGORITHMIC",
        stimulus_prompt="The birthplace of Marie Curie is located in the European nation of",
        scaling_alpha=1.50,
    )

    assert isinstance(cert.protocol, ProspectiveExperimentProtocol)
    assert len(cert.protocol.targets) == 3
    for t in cert.protocol.targets:
        assert isinstance(t, QuantitativePredictionTarget)
        assert t.predicted_mean > 0.0
        assert t.predicted_std > 0.0


def test_prospective_experimental_execution_and_accuracy():
    """Verifies that relative prediction error is bounded (<= 5%) and Prospective Prediction Accuracy >= 95%."""
    engine = ProspectivePredictionEngine()
    cert = engine.derive_and_execute_prospective_experiment()

    assert isinstance(cert, ProspectivePredictionCertificate)
    assert cert.is_falsification_survived is True
    assert cert.mean_relative_error <= 0.05
    assert cert.prospective_accuracy_pct >= 95.0
    for t in cert.protocol.targets:
        assert t.relative_error <= 0.05


def test_calibration_z_score_confidence_bounds():
    """Verifies that all prospective continuous targets lie strictly within 95% confidence intervals (|z| <= 1.96)."""
    engine = ProspectivePredictionEngine()
    cert = engine.derive_and_execute_prospective_experiment()

    for t in cert.protocol.targets:
        assert abs(t.z_score) <= 1.96
        assert t.is_within_95_ci is True


def test_prospective_certificate_and_dag_integration():
    """Verifies that prospective certificates publish SHA-256 seals and register ACTIVE_SUPPORTED in the DAG."""
    claim_graph = ClaimDependencyGraphEngine()
    engine = ProspectivePredictionEngine(claim_graph=claim_graph)

    cert = engine.derive_and_execute_prospective_experiment(
        theory_id="THEORY_MULTIHOP_RELATIONAL_REASONING_ALGORITHMIC",
    )

    assert len(cert.sha256_seal) == 64
    claim_id = f"CLAIM_PROSPECTIVE_PREDICTION_{cert.theory_id}"
    assert claim_id in claim_graph.claims
    assert claim_graph.claims[claim_id].belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED
    assert "Prospective Quantitative Prediction Confirmed" in claim_graph.claims[claim_id].claim_statement
