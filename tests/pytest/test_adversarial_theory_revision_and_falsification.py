"""Unit and integration tests for Phase 36: Adversarial Theory Revision & False Primitive Rejection."""

import pytest

from backend.discovery.adversarial_theory_revision_engine import (
    AdversarialTheoryRevisionEngine,
    AdversarialTheoryScorecard,
    CandidatePrimitiveType,
    PrimitiveCandidateEvaluation,
)
from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief


def test_adversarial_counterexample_rejects_spurious_primitive():
    """Verifies that candidate primitives relying on surface punctuation shortcuts are rejected (Rescue < 0.80)."""
    engine = AdversarialTheoryRevisionEngine()
    eval_res = engine.evaluate_candidate_primitive_adversarially(
        candidate_id="PRIM_CANDIDATE_PUNCTUATION_HEURISTIC",
        candidate_type=CandidatePrimitiveType.SPURIOUS_SURFACE_SHORTCUT,
        description="Punctuation shortcut",
    )

    assert isinstance(eval_res, PrimitiveCandidateEvaluation)
    assert eval_res.is_accepted is False
    assert eval_res.counterexample_rescue < 0.20
    assert "REJECTED ON COUNTEREXAMPLE" in eval_res.rejection_reason


def test_ood_stress_rejects_entangled_primitive():
    """Verifies that polysemantic superposition candidates fail OOD stress testing (OOD Error > 0.10)."""
    engine = AdversarialTheoryRevisionEngine()
    eval_res = engine.evaluate_candidate_primitive_adversarially(
        candidate_id="PRIM_CANDIDATE_POLYSEMANTIC_OVERLAP",
        candidate_type=CandidatePrimitiveType.POLYSEMANTIC_ENTANGLED,
        description="Superposition overlap",
    )

    assert eval_res.is_accepted is False
    assert eval_res.ood_generalization_error > 0.10
    assert "REJECTED ON OOD STRESS" in eval_res.rejection_reason


def test_genuine_primitive_survives_adversarial_tournament():
    """Verifies that genuine invariant primitives survive all 3 regimes (Rescue >= 0.85, OOD Err <= 0.10, Margin >= 0.90)."""
    engine = AdversarialTheoryRevisionEngine()
    eval_res = engine.evaluate_candidate_primitive_adversarially(
        candidate_id="PRIM_CANDIDATE_RECURSIVE_TREE_PARSER",
        candidate_type=CandidatePrimitiveType.GENUINE_INVARIANT_PRIMITIVE,
        description="Genuine recursive parser",
    )

    assert eval_res.is_accepted is True
    assert eval_res.counterexample_rescue >= 0.85
    assert eval_res.ood_generalization_error <= 0.10
    assert eval_res.rival_falsification_margin >= 0.90
    assert eval_res.rejection_reason is None


def test_false_primitive_acceptance_rate_zero_and_dag_integration():
    """Verifies False Primitive Acceptance Rate = 0.0% and verifies Claim DAG lifecycle states."""
    claim_graph = ClaimDependencyGraphEngine()
    engine = AdversarialTheoryRevisionEngine(claim_graph=claim_graph)

    scorecard = engine.run_adversarial_theory_revision_battery()

    assert isinstance(scorecard, AdversarialTheoryScorecard)
    assert scorecard.is_epistemic_safety_verified is True
    assert scorecard.false_primitive_acceptance_rate_pct == 0.0
    assert scorecard.accepted_candidates_count == 1
    assert scorecard.rejected_candidates_count == 2
    assert len(scorecard.sha256_seal) == 64

    # Verify that rejected claims were registered as FALSIFIED_REVERTED in the Claim DAG
    rej_claim_id = "CLAIM_REJECTED_PRIM_CANDIDATE_PUNCTUATION_HEURISTIC"
    assert rej_claim_id in claim_graph.claims
    assert claim_graph.claims[rej_claim_id].belief_status == ClaimEpistemicBelief.FALSIFIED_REVERTED

    # Verify that accepted claim was registered as ACTIVE_SUPPORTED
    acc_claim_id = "CLAIM_ACCEPTED_PRIM_CANDIDATE_RECURSIVE_TREE_PARSER"
    assert acc_claim_id in claim_graph.claims
    assert claim_graph.claims[acc_claim_id].belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED
