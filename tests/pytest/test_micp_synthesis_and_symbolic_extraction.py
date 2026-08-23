"""Unit and integration tests for Phase 31: Minimal Invariant Computational Program (MICP) Synthesis."""

import pytest

from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief
from backend.discovery.micp_synthesis_engine import (
    CanonicalMICPCertificate,
    FamilyRoutingMotif,
    MICPSynthesisEngine,
    MinimalInvariantProgram,
    SubstrateBindingMapping,
    SymbolicOpType,
    SymbolicProgramNode,
)


def test_micp_symbolic_core_synthesis():
    """Verifies that the MICP strictly decouples Universal Core, Family Motifs, and Substrate Bindings."""
    engine = MICPSynthesisEngine()
    program = engine.synthesize_minimal_invariant_program("country_capital")

    assert isinstance(program, MinimalInvariantProgram)
    assert len(program.core_symbolic_nodes) == 4

    # Verify Universal Core Nodes
    core_ops = [n.op_type for n in program.core_symbolic_nodes if n.is_universal_core]
    assert SymbolicOpType.EXTRACT_SUBJECT in core_ops
    assert SymbolicOpType.RETRIEVE_RELATION in core_ops
    assert SymbolicOpType.PROJECT_VOCAB_LOGITS in core_ops

    # Verify Family Motifs
    assert "GPT_PYTHIA_STANDARD" in program.family_motifs
    assert "QWEN_ROPE" in program.family_motifs
    assert "MISTRAL_GQA" in program.family_motifs

    # Verify Substrate Bindings
    assert "gpt2-small" in program.substrate_bindings
    assert "pythia-70m" in program.substrate_bindings
    assert "qwen-0.5b" in program.substrate_bindings
    assert "mistral-7b" in program.substrate_bindings


def test_micp_program_minimality_and_fidelity():
    """Verifies that the synthesized program satisfies minimality (>=0.90) and fidelity (>=0.95)."""
    engine = MICPSynthesisEngine()
    program = engine.synthesize_minimal_invariant_program("country_capital")

    assert program.program_minimality_score >= 0.90
    assert program.execution_fidelity_score >= 0.95


def test_micp_symbolic_interpreter_execution():
    """Verifies that the symbolic program can be interpreted and produces exact factual inferences."""
    engine = MICPSynthesisEngine()
    program = engine.synthesize_minimal_invariant_program("country_capital")

    # Interpret query: France -> capital
    trace = engine.execute_symbolic_program(program, subject_entity="France", relation_name="capital")

    assert trace["op_0_extract"] == "EntityToken(France)"
    assert trace["op_1_retrieve"] == "Paris"
    assert trace["PREDICTED_OUTPUT"] == "Paris"


def test_canonical_micp_certificate_and_dag_registration():
    """Verifies that Canonical MICP Certification generates a SHA-256 seal and registers the program claim in the DAG."""
    claim_graph = ClaimDependencyGraphEngine()
    engine = MICPSynthesisEngine(claim_graph=claim_graph)
    program = engine.synthesize_minimal_invariant_program("country_capital")

    cert = engine.certify_micp_program(program)

    assert isinstance(cert, CanonicalMICPCertificate)
    assert len(cert.sha256_seal) == 64
    assert len(cert.evaluated_models) == 4

    # Verify registration in Claim DAG
    claim_id = "CLAIM_MICP_COUNTRY_CAPITAL"
    assert claim_id in claim_graph.claims
    assert claim_graph.claims[claim_id].belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED
    assert len(claim_graph.claims[claim_id].dependencies) == 4
