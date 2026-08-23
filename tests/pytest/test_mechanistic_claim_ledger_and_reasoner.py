"""Unit and integration tests for Phase 20: Autonomous Scientific Reasoner & Claim Ledger."""

import pytest
import torch

from backend.discovery.mechanistic_claim_ledger import (
    AutonomousScientificReasoner,
    ClaimEvidenceItem,
    CompetingHypothesisRecord,
    HypothesisEpistemicState,
    MechanisticClaimCertificate,
)
from backend.discovery.autonomous_science_orchestrator import (
    AutonomousScienceOrchestrator,
    AutonomousScienceRunResult,
)
from backend.runtime.in_memory_runtime import InMemoryRuntime
from backend.runtime.interfaces import PrecisionProfile


def test_mechanistic_claim_certificate_structure():
    """Verifies that MechanisticClaimCertificate formats all evidence, hypotheses, and invariants cleanly."""
    reasoner = AutonomousScientificReasoner(model_id="gpt2", device="cpu")
    cert = reasoner.generate_mechanistic_claim_certificate(
        circuit_or_component_id="L8_N412",
        behavior_name="french_capital_retrieval",
        clean_prompt="The capital of France is",
        target_token=" Paris",
        causal_delta_z=0.312,
        edge_divergence=0.024,
        control_specificity_ratio=4.85,
        mediation_rescue_fraction=0.78,
        cross_prompt_replication_pct=85.0,
        directional_projection_score=0.85,
    )

    assert isinstance(cert, MechanisticClaimCertificate)
    assert cert.circuit_or_component_id == "L8_N412"
    assert len(cert.evidence_support_checklist) == 5
    assert len(cert.competing_hypotheses_matrix) == 4
    assert cert.surviving_hypothesis_id == "H1_Relational"
    assert len(cert.canonical_certificate_hash) == 64
    assert "W_U * d_i" in cert.invariant_statement

    # Render markdown summary
    md = cert.render_markdown_summary()
    assert "MECHANISTIC CLAIM CERTIFICATE" in md
    assert "✓ Node Causal Necessity" in md
    assert "✓ SURVIVED [H1_Relational]" in md
    assert "✗ FALSIFIED [H2_BroadTopic]" in md
    assert "? UNRESOLVED [H4_PositionalArtifact]" in md


def test_bayesian_epistemic_belief_updates():
    """Verifies that Bayesian belief updating shifts posterior probabilities rationally based on test evidence."""
    reasoner = AutonomousScientificReasoner(model_id="gpt2", device="cpu")
    priors = {"H1": 0.25, "H2": 0.25, "H3": 0.25, "H4": 0.25}

    # Discriminating outcomes favoring H1 and disfavoring H2/H3
    outcomes = [
        ("test_1", {"H1": True, "H2": False, "H3": True, "H4": True}),
        ("test_2", {"H1": True, "H2": True, "H3": False, "H4": True}),
    ]

    posteriors = reasoner.update_bayesian_beliefs(priors, outcomes)

    # Total mass must sum to 1.0 (within float precision)
    assert abs(sum(posteriors.values()) - 1.0) < 1e-3

    # H1 should have significantly higher posterior than refuted hypotheses H2 and H3
    assert posteriors["H1"] > posteriors["H2"]
    assert posteriors["H1"] > posteriors["H3"]


def test_unresolved_boundary_condition_declaration():
    """Verifies that untested hypotheses (e.g. positional artifacts) are strictly labeled UNRESOLVED_OPEN."""
    reasoner = AutonomousScientificReasoner(model_id="gpt2", device="cpu")
    cert = reasoner.generate_mechanistic_claim_certificate(
        circuit_or_component_id="L8_N412",
        behavior_name="french_capital_retrieval",
    )

    h4 = next(h for h in cert.competing_hypotheses_matrix if h.hypothesis_id == "H4_PositionalArtifact")
    assert h4.epistemic_state == HypothesisEpistemicState.UNRESOLVED_OPEN
    assert "Open Question" in (h4.refutation_rationale or "")

    # Unresolved boundary conditions must be non-empty
    assert len(cert.unresolved_boundary_conditions) >= 2


def test_end_to_end_autonomous_scientist_loop():
    """Verifies full execution from search to ACDC pruning to causal testing to claim certification."""
    runtime = InMemoryRuntime(
        model_id="gpt2",
        device="cpu",
        precision=PrecisionProfile(weight_dtype="float32", activation_dtype="float32"),
    )
    orchestrator = AutonomousScienceOrchestrator(runtime=runtime, model_id="gpt2", device="cpu")

    result = orchestrator.execute_autonomous_investigation(
        behavior_name="country_capital_scientific_investigation",
        clean_prompt="The capital of France is",
        target_token=" Paris",
        corrupted_prompt="The capital of Italy is",
        target_layers=[6, 8, 10],
        pruning_threshold_tau=0.010,
    )

    assert isinstance(result, AutonomousScienceRunResult)
    assert result.model_id == "gpt2"
    assert result.dual_loop_report is not None
    assert result.mechanistic_claim_certificate is not None
    assert len(result.formatted_certificate_markdown) > 100
