"""Unit and integration tests for Phase 30: Multi-Model Transfer Matrix & Subcircuit Universality Decomposition."""

import pytest

from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief
from backend.discovery.multi_model_transfer_matrix import (
    DecomposedUniversalCircuitCertificate,
    FourLevelEpistemicAssessment,
    MultiModelTransferMatrixEngine,
    MultiModelTransferMatrixScorecard,
    SubcircuitRoleDecomposition,
    UniversalityScopeType,
)


def test_4x4_multi_model_transfer_matrix_computation():
    """Verifies that the N x N transfer matrix correctly calculates pairwise causal transfer and scope classification."""
    engine = MultiModelTransferMatrixEngine()
    models = ["gpt2-small", "pythia-70m", "qwen-0.5b", "mistral-7b"]

    transfers = {
        ("gpt2-small", "pythia-70m"): 0.84,
        ("gpt2-small", "qwen-0.5b"): 0.71,
        ("gpt2-small", "mistral-7b"): 0.22,
        ("pythia-70m", "gpt2-small"): 0.84,
        ("pythia-70m", "qwen-0.5b"): 0.76,
        ("pythia-70m", "mistral-7b"): 0.19,
        ("qwen-0.5b", "gpt2-small"): 0.71,
        ("qwen-0.5b", "pythia-70m"): 0.76,
        ("qwen-0.5b", "mistral-7b"): 0.28,
        ("mistral-7b", "gpt2-small"): 0.22,
        ("mistral-7b", "pythia-70m"): 0.19,
        ("mistral-7b", "qwen-0.5b"): 0.28,
    }

    scorecard = engine.compute_transfer_matrix_and_decomposition(
        behavior_name="country_capital",
        model_ids=models,
        pairwise_causal_transfers=transfers,
    )

    assert isinstance(scorecard, MultiModelTransferMatrixScorecard)
    assert len(scorecard.transfer_matrix) == 4
    assert scorecard.transfer_matrix["gpt2-small"]["pythia-70m"] == 0.84
    assert scorecard.transfer_matrix["gpt2-small"]["mistral-7b"] == 0.22
    assert scorecard.overall_universality_scope == UniversalityScopeType.ARCHITECTURAL_FAMILY_SPECIFIC


def test_fine_grained_subcircuit_universality_decomposition():
    """Verifies that subcircuit stages exhibit distinct, independent universality scopes."""
    engine = MultiModelTransferMatrixEngine()
    models = ["gpt2-small", "pythia-70m", "qwen-0.5b"]

    transfers = {
        ("gpt2-small", "pythia-70m"): 0.84,
        ("gpt2-small", "qwen-0.5b"): 0.76,
        ("pythia-70m", "gpt2-small"): 0.84,
        ("pythia-70m", "qwen-0.5b"): 0.78,
        ("qwen-0.5b", "gpt2-small"): 0.76,
        ("qwen-0.5b", "pythia-70m"): 0.78,
    }

    scorecard = engine.compute_transfer_matrix_and_decomposition(
        behavior_name="country_capital",
        model_ids=models,
        pairwise_causal_transfers=transfers,
    )

    stages = {s.subcircuit_stage_name: s for s in scorecard.subcircuit_decompositions}
    assert len(stages) == 4

    # Stage 2 (Relational Retrieval) is broadly universal
    assert stages["STAGE_2_RELATIONAL_RETRIEVAL"].universality_scope == UniversalityScopeType.UNIVERSAL_BROAD
    assert stages["STAGE_2_RELATIONAL_RETRIEVAL"].is_modularly_transferable is True

    # Stage 3 (Attention Routing) is family-specific
    assert stages["STAGE_3_ATTENTION_ROUTING_COPY_SUPPRESSION"].universality_scope == UniversalityScopeType.ARCHITECTURAL_FAMILY_SPECIFIC
    assert stages["STAGE_3_ATTENTION_ROUTING_COPY_SUPPRESSION"].is_modularly_transferable is False


def test_4_level_epistemic_hierarchy_preservation():
    """Verifies that Role Alignment, Topology Alignment, Causal Transfer, and Behavioral Transfer are tracked independently."""
    engine = MultiModelTransferMatrixEngine()
    models = ["gpt2-small", "pythia-70m"]
    transfers = {("gpt2-small", "pythia-70m"): 0.84, ("pythia-70m", "gpt2-small"): 0.84}

    scorecard = engine.compute_transfer_matrix_and_decomposition(
        behavior_name="country_capital",
        model_ids=models,
        pairwise_causal_transfers=transfers,
    )

    assert len(scorecard.pairwise_assessments) == 2
    for assessment in scorecard.pairwise_assessments:
        assert isinstance(assessment, FourLevelEpistemicAssessment)
        assert assessment.functional_role_alignment_score >= 0.90
        assert assessment.topology_alignment_score >= 0.80
        assert assessment.causal_transfer_score == 0.84
        assert assessment.behavioral_transfer_score >= 0.90


def test_decomposed_universal_certificate_and_dag_integration():
    """Verifies that Decomposed Universal Certificates generate SHA-256 seals and register subcircuit claims in the DAG."""
    claim_graph = ClaimDependencyGraphEngine()
    engine = MultiModelTransferMatrixEngine(claim_graph=claim_graph)
    models = ["gpt2-small", "pythia-70m", "qwen-0.5b"]

    transfers = {
        ("gpt2-small", "pythia-70m"): 0.84,
        ("gpt2-small", "qwen-0.5b"): 0.76,
        ("pythia-70m", "gpt2-small"): 0.84,
        ("pythia-70m", "qwen-0.5b"): 0.78,
        ("qwen-0.5b", "gpt2-small"): 0.76,
        ("qwen-0.5b", "pythia-70m"): 0.78,
    }

    scorecard = engine.compute_transfer_matrix_and_decomposition(
        behavior_name="country_capital",
        model_ids=models,
        pairwise_causal_transfers=transfers,
    )

    cert = engine.certify_decomposed_universal_circuit(scorecard)

    assert isinstance(cert, DecomposedUniversalCircuitCertificate)
    assert len(cert.sha256_seal) == 64
    assert len(cert.transfer_matrix_scorecard.subcircuit_decompositions) == 4

    # Verify that all 4 modular stage claims were registered into the Living Claim DAG
    stage_claim_id = "CLAIM_SUBCIRCUIT_COUNTRY_CAPITAL_STAGE_2_RELATIONAL_RETRIEVAL"
    assert stage_claim_id in claim_graph.claims
    assert claim_graph.claims[stage_claim_id].belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED
