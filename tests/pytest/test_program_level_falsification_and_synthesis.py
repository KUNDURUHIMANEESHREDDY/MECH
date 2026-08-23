"""Unit and integration tests for Phase 32: Program-Level Causal Falsification & Synthesis."""

import pytest

from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief
from backend.discovery.program_falsification_engine import (
    CompetingProgram,
    FalsifiedProgramCertificate,
    ProgramDiscriminatingTest,
    ProgramFalsificationEngine,
    ProgramFalsificationScorecard,
    ProgramHypothesisType,
    ProgramStatus,
)


def test_competing_program_generation():
    """Verifies that 4 distinct competing symbolic program hypotheses are synthesized."""
    engine = ProgramFalsificationEngine()
    programs = engine.generate_competing_programs("country_capital")

    assert len(programs) == 4
    prog_types = [p.program_type for p in programs]
    assert ProgramHypothesisType.RELATIONAL_INVARIANT_CORE in prog_types
    assert ProgramHypothesisType.LEXICAL_NGRAM_RETRIEVAL in prog_types
    assert ProgramHypothesisType.POSITIONAL_TEMPLATE_MEMORY in prog_types
    assert ProgramHypothesisType.DISTRIBUTED_HOLOGRAPHIC in prog_types


def test_program_discriminating_perturbation_battery():
    """Verifies that discriminating perturbation batteries target lexical, positional, and relational dimensions."""
    engine = ProgramFalsificationEngine()
    tests = engine.generate_discriminating_tests("country_capital")

    assert len(tests) == 3
    test_types = [t.perturbation_type for t in tests]
    assert "LEXICAL_SURFACE_MUTATION" in test_types
    assert "POSITIONAL_SLOT_INVERSION" in test_types
    assert "MODULAR_RESIDUAL_ABLATION" in test_types


def test_cross_model_program_falsification_execution():
    """Verifies that competitive execution refutes spurious programs and selects the invariant core (margin >= 0.90)."""
    engine = ProgramFalsificationEngine()
    scorecard = engine.run_program_falsification_tournament(
        behavior_name="country_capital",
        evaluated_models=["gpt2-small", "pythia-70m", "qwen-0.5b", "mistral-7b"],
    )

    assert isinstance(scorecard, ProgramFalsificationScorecard)
    assert scorecard.is_falsification_unambiguous is True
    assert scorecard.falsification_margin >= 0.90

    # Verify that spurious programs were refuted
    refuted = [p for p in scorecard.competing_programs if p.status == ProgramStatus.FALSIFIED_REFUTED]
    assert len(refuted) == 3

    # Verify that candidate MICP survived
    surviving = next(p for p in scorecard.competing_programs if p.status == ProgramStatus.ACTIVE_SURVIVING)
    assert surviving.program_type == ProgramHypothesisType.RELATIONAL_INVARIANT_CORE
    assert surviving.posterior_probability >= 0.95


def test_falsified_program_certificate_and_dag_integration():
    """Verifies that Falsified Program Certification generates a SHA-256 seal and registers claims in the DAG."""
    claim_graph = ClaimDependencyGraphEngine()
    engine = ProgramFalsificationEngine(claim_graph=claim_graph)

    scorecard = engine.run_program_falsification_tournament(
        behavior_name="country_capital",
        evaluated_models=["gpt2-small", "pythia-70m", "qwen-0.5b", "mistral-7b"],
    )

    cert = engine.certify_falsified_program_tournament(scorecard)

    assert isinstance(cert, FalsifiedProgramCertificate)
    assert len(cert.sha256_seal) == 64
    assert len(cert.refuted_programs) == 3

    # Verify registration in Claim DAG
    claim_id = "CLAIM_CERTIFIED_PROGRAM_COUNTRY_CAPITAL"
    assert claim_id in claim_graph.claims
    assert claim_graph.claims[claim_id].belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED
