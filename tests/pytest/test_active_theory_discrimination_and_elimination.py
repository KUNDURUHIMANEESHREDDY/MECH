"""Unit and integration tests for Phase 39: Active Theory Discrimination & Optimal Experiment Selection."""

import pytest

from backend.discovery.active_theory_discrimination_engine import (
    ActiveDiscriminationTournamentResult,
    ActiveTheoryDiscriminationEngine,
    DiscriminatingExperimentCandidate,
)
from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief


def test_divergent_prediction_generation():
    """Verifies that the engine generates candidate experiments with divergent quantitative predictions."""
    engine = ActiveTheoryDiscriminationEngine()
    candidates = engine.generate_candidate_discriminating_experiments(
        theory_a_id="THEORY_ALGORITHMIC_RELATIONAL_ROUTING",
        theory_b_id="THEORY_POSITIONAL_ASSOCIATION_HEURISTIC",
    )

    assert len(candidates) == 3
    for c in candidates:
        assert isinstance(c, DiscriminatingExperimentCandidate)
        assert c.divergence_sigma > 1.0
        assert c.eig_bits > 0.0


def test_max_eig_experiment_selection():
    """Verifies that the engine selects the candidate with maximum EIG and >= 3.0 sigma divergence."""
    engine = ActiveTheoryDiscriminationEngine()
    candidates = engine.generate_candidate_discriminating_experiments(
        theory_a_id="THEORY_ALGORITHMIC_RELATIONAL_ROUTING",
        theory_b_id="THEORY_POSITIONAL_ASSOCIATION_HEURISTIC",
    )
    optimal = engine.select_optimal_experiment(candidates)

    assert optimal.experiment_id == "EXP_COUNTERFACTUAL_SYNTACTIC_PERMUTATION"
    assert optimal.eig_bits >= 0.90
    assert optimal.divergence_sigma >= 3.0


def test_prospective_discrimination_execution_and_elimination():
    """Verifies that the active tournament decisively eliminates the failing theory (P_winner >= 0.95, Margin >= 0.90)."""
    engine = ActiveTheoryDiscriminationEngine()
    result = engine.run_active_theory_discrimination_tournament(
        theory_a_id="THEORY_ALGORITHMIC_RELATIONAL_ROUTING",
        theory_b_id="THEORY_POSITIONAL_ASSOCIATION_HEURISTIC",
    )

    assert isinstance(result, ActiveDiscriminationTournamentResult)
    assert result.is_decisive is True
    assert result.winning_theory_id == "THEORY_ALGORITHMIC_RELATIONAL_ROUTING"
    assert result.eliminated_theory_id == "THEORY_POSITIONAL_ASSOCIATION_HEURISTIC"
    assert result.posterior_beliefs[result.winning_theory_id] >= 0.95
    assert result.falsification_margin >= 0.90


def test_active_discrimination_dag_lifecycle_updates():
    """Verifies that the winning theory is ACTIVE_SUPPORTED and the eliminated theory is FALSIFIED_REVERTED."""
    claim_graph = ClaimDependencyGraphEngine()
    engine = ActiveTheoryDiscriminationEngine(claim_graph=claim_graph)

    result = engine.run_active_theory_discrimination_tournament(
        theory_a_id="THEORY_ALGORITHMIC_RELATIONAL_ROUTING",
        theory_b_id="THEORY_POSITIONAL_ASSOCIATION_HEURISTIC",
    )

    assert len(result.sha256_seal) == 64

    # 1. Winning Theory -> ACTIVE_SUPPORTED
    win_claim = f"CLAIM_THEORY_{result.winning_theory_id}"
    assert win_claim in claim_graph.claims
    assert claim_graph.claims[win_claim].belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED

    # 2. Eliminated Theory -> FALSIFIED_REVERTED
    elim_claim = f"CLAIM_THEORY_{result.eliminated_theory_id}"
    assert elim_claim in claim_graph.claims
    assert claim_graph.claims[elim_claim].belief_status == ClaimEpistemicBelief.FALSIFIED_REVERTED
    assert "Theory Eliminated under Max-EIG" in claim_graph.claims[elim_claim].claim_statement
