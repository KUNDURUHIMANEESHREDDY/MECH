"""Unit and integration tests for Phase 60: Autonomous Scientific Revision & Closed-Loop Theory Evolution Engine."""

import pytest

from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief
from backend.discovery.revision_evidence_engine import HypothesisEvidenceScore, RevisionEvidenceEngine, RevisionPosteriorScorecard
from backend.discovery.revision_experiment_planner import RevisionCandidateExperiment, RevisionExperimentPlanner
from backend.discovery.revision_validation_engine import AutonomousRevisionCertificate, RevisionValidationEngine
from backend.discovery.theory_diff_engine import TheoryDiffEngine, TheoryDiffRecord
from backend.discovery.theory_revision_engine import RevisionHypothesis, TheoryRevisionEngine
from backend.discovery.theory_version_registry import TheoryVersion, TheoryVersionRegistry, TransportabilityLawEquation


def test_theory_version_registry_and_diff_lineage():
    """Verifies that theory versions maintain immutable SHA-256 lineage and formal structural diffs."""
    registry = TheoryVersionRegistry()
    t0 = registry.get_version("T_0_BASE")

    assert t0.parent_version_id is None
    assert len(t0.theory_sha256) == 64

    # Create revised theory T_1
    eq_t1 = TransportabilityLawEquation(
        w_role=1.40,
        w_linear=1.20,
        w_poly=-1.80,
        w_dim=-1.20,
        w_routing=-1.45,  # Revised term
        bias=-0.60,
    )
    t1 = TheoryVersion(
        version_id="T_1_MOE_ROUTING_EXPANDED",
        parent_version_id="T_0_BASE",
        equation=eq_t1,
        assumptions=["Dense Transformer Self-Attention Homogeneity", "Linear Inter-Substrate Projection", "MoE Routing Dispersion Penalty"],
        validity_domain=["Standard Dense Transformers", "Sparse MoE Transformers (e.g. Mixtral)"],
        known_boundaries=["SSM Recurrent Architectures"],
        evidence_experiment_ids=["EXP_CALIBRATION_PHASE48", "EXP_DISCRIMINATING_ROUTING_PERTURBATION"],
        timestamp_utc="2026-08-16T12:00:00Z",
    )
    registry.register_version(t1)

    lineage = registry.get_lineage("T_1_MOE_ROUTING_EXPANDED")
    assert lineage == ["T_0_BASE", "T_1_MOE_ROUTING_EXPANDED"]

    # Compute structural diff
    diff_engine = TheoryDiffEngine()
    diff = diff_engine.compute_diff(
        t_source=t0,
        t_target=t1,
        causal_reason="Observed residual on Mixtral MoE routing dispersion.",
    )

    assert isinstance(diff, TheoryDiffRecord)
    assert diff.parameter_shifts["delta_w_routing"] == -1.45
    assert "MoE Routing Dispersion Penalty" in diff.added_assumptions
    assert "Negligible Routing Perturbation" in diff.removed_assumptions


def test_competing_revision_generation_and_evidence():
    """Verifies generation of >= 3 competing revisions and Bayesian posterior scoring."""
    registry = TheoryVersionRegistry()
    t0 = registry.get_version("T_0_BASE")

    rev_engine = TheoryRevisionEngine()
    hypotheses = rev_engine.generate_candidate_revisions(
        base_theory=t0,
        observed_residual=-0.30,
        target_model="mistralai/Mixtral-8x7B",
        task_name="dynamic_routed_token_comparison",
    )

    assert len(hypotheses) >= 3
    h_ids = [h.hypothesis_id for h in hypotheses]
    assert "REV_H1_MOE_ROUTING_PENALTY" in h_ids
    assert "REV_H2_POLYSEMANTIC_SCALING" in h_ids
    assert "REV_H3_DIMENSION_CAPACITY_CORRECTION" in h_ids

    # Bayesian evidence scoring with empirical observation R_obs = 0.510
    evidence_engine = RevisionEvidenceEngine()
    scorecard = evidence_engine.evaluate_hypotheses(
        hypotheses=hypotheses,
        empirical_r_observed=0.510,
        empirical_delta_z_observed=1.45,
        r_routing_feature=0.65,
    )

    assert isinstance(scorecard, RevisionPosteriorScorecard)
    assert scorecard.winning_hypothesis_id == "REV_H1_MOE_ROUTING_PENALTY"
    assert scorecard.is_decisive is True
    assert scorecard.selection_margin >= 0.40


def test_eig_revision_experiment_discrimination():
    """Verifies that the Max-EIG planner selects experiments that maximize mutual information between revisions."""
    registry = TheoryVersionRegistry()
    t0 = registry.get_version("T_0_BASE")
    rev_engine = TheoryRevisionEngine()
    hypotheses = rev_engine.generate_candidate_revisions(t0, -0.30, "Mixtral", "task")

    planner = RevisionExperimentPlanner()
    ranked_exps = planner.plan_discriminating_experiments(hypotheses)

    assert len(ranked_exps) == 3
    assert ranked_exps[0].experiment_id == "EXP_ROUTING_PERTURBATION"
    assert ranked_exps[0].expected_information_gain_eig >= 0.90
    assert ranked_exps[0].expected_information_gain_eig > ranked_exps[1].expected_information_gain_eig


def test_three_partition_validation_and_certificate():
    """Verifies that the revised theory passes the 3-partition audit (Retrospective, Failure Case, Fresh Holdouts)."""
    registry = TheoryVersionRegistry()
    t0 = registry.get_version("T_0_BASE")

    eq_t1 = TransportabilityLawEquation(
        w_role=1.40,
        w_linear=1.20,
        w_poly=-1.80,
        w_dim=-1.20,
        w_routing=-1.45,
        bias=-0.60,
    )
    t1 = TheoryVersion(
        version_id="T_1_MOE_ROUTING_EXPANDED",
        parent_version_id="T_0_BASE",
        equation=eq_t1,
        assumptions=["Dense Transformer Self-Attention Homogeneity", "Linear Inter-Substrate Projection", "MoE Routing Dispersion Penalty"],
        validity_domain=["Standard Dense Transformers", "Sparse MoE Transformers (e.g. Mixtral)"],
        known_boundaries=["SSM Recurrent Architectures"],
        evidence_experiment_ids=["EXP_ROUTING_PERTURBATION"],
        timestamp_utc="2026-08-16T12:00:00Z",
    )

    diff_engine = TheoryDiffEngine()
    diff = diff_engine.compute_diff(t0, t1, "MoE routing correction.")

    claim_graph = ClaimDependencyGraphEngine()
    val_engine = RevisionValidationEngine(claim_graph=claim_graph)

    cert = val_engine.audit_three_partition_evolution(
        source_theory=t0,
        revised_theory=t1,
        diff_record=diff,
    )

    assert isinstance(cert, AutonomousRevisionCertificate)
    assert cert.is_fully_certified is True
    assert cert.revision_success_rate == 1.0
    assert cert.retrospective_stability >= 0.95
    assert cert.prospective_improvement >= 0.90
    assert cert.false_revision_rate <= 0.05

    claim_id = f"CLAIM_THEORY_REVISION_{t1.version_id}"
    assert claim_id in claim_graph.claims
    assert claim_graph.claims[claim_id].belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED
