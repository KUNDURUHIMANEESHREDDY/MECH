"""Unit and integration tests for Phase 28: Cross-Model Circuit Topology Alignment."""

import pytest

from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief
from backend.discovery.cross_model_alignment_engine import (
    CrossModelAlignmentEngine,
    CrossModelAlignmentScorecard,
    FunctionalRoleType,
    SubstrateIndependenceClass,
    UniversalCircuitCertificate,
)


def test_cross_model_functional_role_mapping():
    """Verifies that causal functional roles are correctly mapped between distinct model architectures."""
    engine = CrossModelAlignmentEngine()

    src_comps = {
        "gpt2_L0_N12": FunctionalRoleType.SUBJECT_EXTRACTOR,
        "gpt2_L8_N412": FunctionalRoleType.RELATIONAL_RETRIEVER,
        "gpt2_L10_H7": FunctionalRoleType.NAME_MOVER,
        "gpt2_L11_N900": FunctionalRoleType.OUTPUT_PROJECTOR,
    }
    src_edges = [
        ("gpt2_L0_N12", "gpt2_L8_N412"),
        ("gpt2_L8_N412", "gpt2_L10_H7"),
        ("gpt2_L10_H7", "gpt2_L11_N900"),
    ]

    tgt_comps = {
        "pythia_L1_N30": FunctionalRoleType.SUBJECT_EXTRACTOR,
        "pythia_L7_N384": FunctionalRoleType.RELATIONAL_RETRIEVER,
        "pythia_L9_H4": FunctionalRoleType.NAME_MOVER,
        "pythia_L11_N512": FunctionalRoleType.OUTPUT_PROJECTOR,
    }
    tgt_edges = [
        ("pythia_L1_N30", "pythia_L7_N384"),
        ("pythia_L7_N384", "pythia_L9_H4"),
        ("pythia_L9_H4", "pythia_L11_N512"),
    ]

    scorecard = engine.align_model_circuits(
        source_model_id="gpt2-small",
        target_model_id="pythia-70m",
        behavior_name="country_capital",
        source_circuit_components=src_comps,
        source_edges=src_edges,
        target_circuit_components=tgt_comps,
        target_edges=tgt_edges,
    )

    assert isinstance(scorecard, CrossModelAlignmentScorecard)
    assert len(scorecard.role_mappings) == 4
    assert scorecard.topology_similarity_score == 1.0  # Perfect topological isomorphism
    assert scorecard.substrate_independence_class == SubstrateIndependenceClass.SUBSTRATE_INDEPENDENT_CANONICAL
    assert scorecard.is_universal_certified is True


def test_topology_similarity_and_substrate_classification():
    """Verifies that divergent circuit topologies are classified as ARCHITECTURE_SPECIFIC_LOCAL."""
    engine = CrossModelAlignmentEngine()

    src_comps = {
        "gpt2_L0_N12": FunctionalRoleType.SUBJECT_EXTRACTOR,
        "gpt2_L8_N412": FunctionalRoleType.RELATIONAL_RETRIEVER,
    }
    src_edges = [("gpt2_L0_N12", "gpt2_L8_N412")]

    # Target model has non-overlapping wiring
    tgt_comps = {
        "m_L0_N1": FunctionalRoleType.SUBJECT_EXTRACTOR,
        "m_L5_N2": FunctionalRoleType.COPY_SUPPRESSOR,
    }
    tgt_edges = [("m_L0_N1", "m_L5_N2")]

    scorecard = engine.align_model_circuits(
        source_model_id="gpt2-small",
        target_model_id="custom-dense-arch",
        behavior_name="country_capital",
        source_circuit_components=src_comps,
        source_edges=src_edges,
        target_circuit_components=tgt_comps,
        target_edges=tgt_edges,
    )

    assert scorecard.topology_similarity_score == 0.0
    assert scorecard.substrate_independence_class in (
        SubstrateIndependenceClass.ARCHITECTURE_SPECIFIC_LOCAL,
        SubstrateIndependenceClass.POLYSEMANTIC_DIVERGENT,
    )
    assert scorecard.is_universal_certified is False


def test_universal_circuit_certificate_and_dag_registration():
    """Verifies that universal certification generates a SHA-256 seal and updates the living Claim DAG."""
    claim_graph = ClaimDependencyGraphEngine()
    engine = CrossModelAlignmentEngine(claim_graph=claim_graph)

    # Align GPT-2 to Pythia
    src_comps = {
        "gpt2_L0_N12": FunctionalRoleType.SUBJECT_EXTRACTOR,
        "gpt2_L8_N412": FunctionalRoleType.RELATIONAL_RETRIEVER,
        "gpt2_L10_H7": FunctionalRoleType.NAME_MOVER,
        "gpt2_L11_N900": FunctionalRoleType.OUTPUT_PROJECTOR,
    }
    src_edges = [
        ("gpt2_L0_N12", "gpt2_L8_N412"),
        ("gpt2_L8_N412", "gpt2_L10_H7"),
        ("gpt2_L10_H7", "gpt2_L11_N900"),
    ]

    tgt_comps = {
        "pythia_L1_N30": FunctionalRoleType.SUBJECT_EXTRACTOR,
        "pythia_L7_N384": FunctionalRoleType.RELATIONAL_RETRIEVER,
        "pythia_L9_H4": FunctionalRoleType.NAME_MOVER,
        "pythia_L11_N512": FunctionalRoleType.OUTPUT_PROJECTOR,
    }
    tgt_edges = [
        ("pythia_L1_N30", "pythia_L7_N384"),
        ("pythia_L7_N384", "pythia_L9_H4"),
        ("pythia_L9_H4", "pythia_L11_N512"),
    ]

    scorecard1 = engine.align_model_circuits(
        source_model_id="gpt2-small",
        target_model_id="pythia-70m",
        behavior_name="country_capital",
        source_circuit_components=src_comps,
        source_edges=src_edges,
        target_circuit_components=tgt_comps,
        target_edges=tgt_edges,
    )

    cert = engine.certify_universal_circuit("country_capital", [scorecard1])

    assert isinstance(cert, UniversalCircuitCertificate)
    assert len(cert.sha256_seal) == 64
    assert cert.substrate_class == SubstrateIndependenceClass.SUBSTRATE_INDEPENDENT_CANONICAL

    # Verify registration in Claim DAG
    claim_id = "CLAIM_UNIVERSAL_COUNTRY_CAPITAL"
    assert claim_id in claim_graph.claims
    assert claim_graph.claims[claim_id].belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED


def test_cross_model_alignment_suite_end_to_end():
    """Verifies end-to-end alignment across GPT-2, Pythia, and Qwen."""
    engine = CrossModelAlignmentEngine()

    c_gpt2 = {
        "g_sub": FunctionalRoleType.SUBJECT_EXTRACTOR,
        "g_rel": FunctionalRoleType.RELATIONAL_RETRIEVER,
        "g_mov": FunctionalRoleType.NAME_MOVER,
        "g_out": FunctionalRoleType.OUTPUT_PROJECTOR,
    }
    e_gpt2 = [("g_sub", "g_rel"), ("g_rel", "g_mov"), ("g_mov", "g_out")]

    c_pythia = {
        "p_sub": FunctionalRoleType.SUBJECT_EXTRACTOR,
        "p_rel": FunctionalRoleType.RELATIONAL_RETRIEVER,
        "p_mov": FunctionalRoleType.NAME_MOVER,
        "p_out": FunctionalRoleType.OUTPUT_PROJECTOR,
    }
    e_pythia = [("p_sub", "p_rel"), ("p_rel", "p_mov"), ("p_mov", "p_out")]

    c_qwen = {
        "q_sub": FunctionalRoleType.SUBJECT_EXTRACTOR,
        "q_rel": FunctionalRoleType.RELATIONAL_RETRIEVER,
        "q_mov": FunctionalRoleType.NAME_MOVER,
        "q_out": FunctionalRoleType.OUTPUT_PROJECTOR,
    }
    e_qwen = [("q_sub", "q_rel"), ("q_rel", "q_mov"), ("q_mov", "q_out")]

    s1 = engine.align_model_circuits("gpt2", "pythia", "country_capital", c_gpt2, e_gpt2, c_pythia, e_pythia)
    s2 = engine.align_model_circuits("gpt2", "qwen", "country_capital", c_gpt2, e_gpt2, c_qwen, e_qwen)

    cert = engine.certify_universal_circuit("country_capital", [s1, s2])

    assert len(cert.evaluated_models) == 3
    assert cert.mean_topology_similarity == 1.0
    assert cert.substrate_class == SubstrateIndependenceClass.SUBSTRATE_INDEPENDENT_CANONICAL
    assert len(cert.to_dict()) > 0
