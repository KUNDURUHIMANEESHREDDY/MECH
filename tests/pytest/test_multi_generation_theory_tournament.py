"""Unit and integration tests for Phase 61: Multi-Generation Autonomous Theory Branching & Open-World Tournament Engine."""

import pytest

from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief
from backend.discovery.open_world_tournament_engine import OpenWorldTournamentEngine, TournamentCertificate
from backend.discovery.theory_branching_engine import TheoryBranchingEngine
from backend.discovery.theory_complexity_engine import TheoryComplexityBreakdown, TheoryComplexityEngine
from backend.discovery.theory_extinction_engine import TheoryExtinctionEngine, TheoryExtinctionRecord
from backend.discovery.theory_population_manager import GenerationSnapshot, TheoryPopulationManager
from backend.discovery.theory_prediction_matrix import ExperimentDisagreementProfile, TheoryPredictionMatrixEngine
from backend.discovery.theory_speciation_engine import SpeciationClusterResult, TheorySpeciationEngine
from backend.discovery.theory_version_registry import TheoryVersion, TheoryVersionRegistry


def test_theory_population_management_and_complexity():
    """Verifies multi-generation population tracking and structural complexity scoring."""
    pop_mgr = TheoryPopulationManager()
    g0 = pop_mgr.get_latest_generation()

    assert g0.generation_index == 0
    assert len(g0.theories) == 3

    complexity_engine = TheoryComplexityEngine()
    comp = complexity_engine.calculate_complexity(
        theory_id=g0.theories[0].version_id,
        num_primitives=4,
        num_assumptions=3,
        num_parameters=6,
        num_special_case_exceptions=0,
    )

    assert isinstance(comp, TheoryComplexityBreakdown)
    assert comp.total_complexity > 0.0
    assert comp.c_exceptions == 0.0


def test_continuous_prediction_matrix_and_max_eig():
    """Verifies continuous prediction matrix calculation and inter-theory disagreement ranking."""
    matrix_engine = TheoryPredictionMatrixEngine()

    preds = {
        "Theory_A": {"EXP_1": 4.2, "EXP_2": 3.1, "EXP_3": 0.8},
        "Theory_B": {"EXP_1": 1.3, "EXP_2": 3.0, "EXP_3": 2.4},
        "Theory_C": {"EXP_1": 4.0, "EXP_2": 0.7, "EXP_3": 2.1},
    }

    profiles = matrix_engine.compute_prediction_matrix(preds)

    assert len(profiles) == 3
    # EXP_1 should have highest variance (4.2 vs 1.3 vs 4.0)
    assert profiles[0].experiment_id == "EXP_1"
    assert profiles[0].inter_theory_variance > profiles[1].inter_theory_variance
    assert profiles[0].expected_falsification_power >= 0.80


def test_theory_speciation_and_extinction_lineage():
    """Verifies that the engine identifies substrate-conditioned clusters and non-destructively records extinctions."""
    pop_mgr = TheoryPopulationManager()
    g0 = pop_mgr.get_latest_generation()
    t_base = g0.theories[1]

    branch_engine = TheoryBranchingEngine()
    branches = branch_engine.branch_theory(t_base, generation_index=1)
    assert len(branches) == 4

    speciation_engine = TheorySpeciationEngine()
    clusters = speciation_engine.evaluate_speciation(branches)

    assert len(clusters) == 3
    cluster_names = [c.cluster_name for c in clusters]
    assert "CLUSTER_DENSE_TRANSFORMER" in cluster_names
    assert "CLUSTER_SPARSE_MOE" in cluster_names
    assert "CLUSTER_SSM_RECURRENT" in cluster_names

    # Extinction audit
    claim_graph = ClaimDependencyGraphEngine()
    extinction_engine = TheoryExtinctionEngine(claim_graph=claim_graph)
    ext_rec = extinction_engine.eliminate_theory(
        theory_id="T_G0_SIMPLE_LINEAR",
        experiment_id="EXP_DISCRIMINATING_MOE",
        winning_theory_id="T_G1_BRANCH_A_MOE_ROUTED",
        posterior_before=0.33,
        posterior_after=0.01,
        reason="Falsified by MoE routing dispersion perturbation.",
    )

    assert isinstance(ext_rec, TheoryExtinctionRecord)
    assert ext_rec.posterior_after == 0.01
    assert len(extinction_engine.extinction_audit_log) == 1


def test_two_generation_open_world_tournament_cycle():
    """Verifies that the OpenWorldTournamentEngine executes a full 2-generation tournament."""
    claim_graph = ClaimDependencyGraphEngine()
    tournament_engine = OpenWorldTournamentEngine(claim_graph=claim_graph)

    cert = tournament_engine.run_multi_generation_tournament()

    assert isinstance(cert, TournamentCertificate)
    assert cert.is_tournament_certified is True
    assert cert.generations_evaluated == 2
    assert cert.theory_selection_accuracy >= 0.90
    assert cert.premature_collapse_rate <= 0.05
    assert cert.tournament_efficiency >= 0.85
    assert cert.speciation_precision == 1.00

    claim_id = "CLAIM_TOURNAMENT_EVOLUTION_G2"
    assert claim_id in claim_graph.claims
    assert claim_graph.claims[claim_id].belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED
