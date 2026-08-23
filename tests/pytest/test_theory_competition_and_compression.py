"""Unit and integration tests for Phase 37: Theory Competition & Explanatory Compression."""

import pytest

from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief
from backend.discovery.theory_competition_engine import (
    MechanisticTheory,
    TheoryCompetitionEngine,
    TheorySelectionStatus,
    TheoryTournamentScorecard,
)


def test_explanatory_utility_and_compression_scoring():
    """Verifies that Occam's Explanatory Utility correctly applies description and assumption penalties."""
    engine = TheoryCompetitionEngine()
    util = engine.compute_explanatory_utility(
        causal_fidelity=0.96,
        predictive_accuracy=0.98,
        ood_generalization=0.94,
        complexity_nodes=4,
        unsupported_assumptions=0,
    )

    # 0.96 + 0.98 + 0.94 - 0.02*4 - 0.05*0 = 2.88 - 0.08 = 2.80
    assert util == pytest.approx(2.80, abs=0.02)


def test_multi_theory_tournament_execution():
    """Verifies that the multi-theory tournament selects the algorithmic composition over rivals."""
    engine = TheoryCompetitionEngine()
    scorecard = engine.run_theory_competition_tournament("multihop_relational_reasoning")

    assert isinstance(scorecard, TheoryTournamentScorecard)
    assert scorecard.is_occam_optimal is True
    assert len(scorecard.evaluated_theories) == 3

    winning = next(t for t in scorecard.evaluated_theories if t.selection_status == TheorySelectionStatus.SELECTED_MINIMAL_THEORY)
    assert "ALGORITHMIC" in winning.theory_id
    assert winning.explanatory_utility_score >= 2.50


def test_occam_pruning_and_supersession():
    """Verifies that over-parameterized circuits are superseded and compression advantage >= 0.40."""
    engine = TheoryCompetitionEngine()
    scorecard = engine.run_theory_competition_tournament("multihop_relational_reasoning")

    assert scorecard.compression_advantage >= 0.40
    assert len(scorecard.superseded_theory_ids) == 1
    assert len(scorecard.falsified_theory_ids) == 1
    assert any("DENSE_CIRCUIT" in tid for tid in scorecard.superseded_theory_ids)
    assert any("LEXICAL" in tid for tid in scorecard.falsified_theory_ids)


def test_theory_competition_dag_integration():
    """Verifies that DAG claims accurately reflect ACTIVE_SUPPORTED, SUPERSEDED, and FALSIFIED_REVERTED states."""
    claim_graph = ClaimDependencyGraphEngine()
    engine = TheoryCompetitionEngine(claim_graph=claim_graph)

    scorecard = engine.run_theory_competition_tournament("multihop_relational_reasoning")

    assert len(scorecard.sha256_seal) == 64

    # 1. Winning Theory Claim -> ACTIVE_SUPPORTED
    win_claim = f"CLAIM_THEORY_{scorecard.winning_theory_id}"
    assert win_claim in claim_graph.claims
    assert claim_graph.claims[win_claim].belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED

    # 2. Dense Theory Claim -> SUPERSEDED
    dense_claim = f"CLAIM_THEORY_{scorecard.superseded_theory_ids[0]}"
    assert dense_claim in claim_graph.claims
    assert claim_graph.claims[dense_claim].belief_status == ClaimEpistemicBelief.SUPERSEDED

    # 3. Lexical Theory Claim -> FALSIFIED_REVERTED
    lex_claim = f"CLAIM_THEORY_{scorecard.falsified_theory_ids[0]}"
    assert lex_claim in claim_graph.claims
    assert claim_graph.claims[lex_claim].belief_status == ClaimEpistemicBelief.FALSIFIED_REVERTED
