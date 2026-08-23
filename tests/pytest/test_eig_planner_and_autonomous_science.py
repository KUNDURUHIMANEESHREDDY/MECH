"""Unit and integration tests for Phase 22: Optimal Experiment Design & EIG Planner."""

import pytest
import torch

from backend.discovery.belief_entropy_engine import BeliefEntropyEngine, HypothesisEntropyProfile
from backend.discovery.closed_loop_science_engine import ClosedLoopInvestigationResult, ClosedLoopScienceEngine
from backend.discovery.eig_utility_optimizer import EIGUtilityOptimizer, OptimalExperimentSelection
from backend.discovery.experiment_design_generator import CandidateExperimentDesign, ExperimentCategory, ExperimentDesignGenerator
from backend.runtime.in_memory_runtime import InMemoryRuntime
from backend.runtime.interfaces import PrecisionProfile


def test_belief_entropy_quantification():
    """Verifies that BeliefEntropyEngine computes Shannon entropy and flags pairwise ambiguity."""
    engine = BeliefEntropyEngine(resolution_threshold_bits=0.45)

    # 1. Uniform distribution across 4 hypotheses -> Max entropy (2.0 bits)

    uniform_probs = {"H1": 0.25, "H2": 0.25, "H3": 0.25, "H4": 0.25}
    prof_uniform = engine.compute_entropy_profile(uniform_probs)
    assert abs(prof_uniform.shannon_entropy_bits - 2.0) < 1e-3
    assert prof_uniform.is_uncertainty_resolved is False
    assert len(prof_uniform.highest_uncertainty_pairs) == 3

    # 2. Resolved distribution -> Low entropy (<0.45 bits)
    resolved_probs = {"H1": 0.95, "H2": 0.02, "H3": 0.02, "H4": 0.01}
    prof_resolved = engine.compute_entropy_profile(resolved_probs)
    assert prof_resolved.shannon_entropy_bits < 0.45
    assert prof_resolved.is_uncertainty_resolved is True
    assert prof_resolved.dominant_hypothesis_id == "H1"



def test_discriminating_experiment_generation():
    """Verifies that ExperimentDesignGenerator generates targeted positional, lexical, and semantic experiment designs."""
    generator = ExperimentDesignGenerator()
    candidates = generator.generate_candidate_battery(
        component_id="L8_N412",
        primary_behavior_clean="The capital of France is",
        primary_target_token=" Paris",
    )

    assert len(candidates) >= 4
    categories = [c.category for c in candidates]
    assert ExperimentCategory.POSITIONAL_PERTURBATION in categories
    assert ExperimentCategory.LEXICAL_SYNTAX_CONTROL in categories
    assert ExperimentCategory.SEMANTIC_TOPIC_PROBE in categories
    assert ExperimentCategory.MEDIATION_KNOCKOUT in categories

    for c in candidates:
        assert len(c.test_prompts) > 0
        assert c.causal_relevance_weight > 0.0
        assert c.validity_weight > 0.0
        assert c.estimated_compute_passes > 0


def test_eig_and_multi_factor_utility_optimization():
    """Verifies that EIGUtilityOptimizer evaluates information gain and ranks by scientific utility U(E)."""
    generator = ExperimentDesignGenerator()
    optimizer = EIGUtilityOptimizer()

    candidates = generator.generate_candidate_battery(component_id="L8_N412")
    current_probs = {"H1_Relational": 0.25, "H2_BroadTopic": 0.25, "H3_LexicalTrigger": 0.25, "H4_PositionalArtifact": 0.25}

    selection = optimizer.evaluate_and_select_best_experiment(current_probs, candidates)
    assert isinstance(selection, OptimalExperimentSelection)
    assert selection.best_experiment is not None
    assert selection.best_experiment.selection_rank == 1
    assert selection.best_experiment.total_scientific_utility_u > 0.0
    assert len(selection.ranked_candidates) == len(candidates)

    # Ranking must be monotonic descending by utility U
    utilities = [c.total_scientific_utility_u for c in selection.ranked_candidates]
    assert utilities == sorted(utilities, reverse=True)


def test_closed_loop_autonomous_science_execution():
    """Verifies full execution of the closed-loop autonomous science engine (Observe -> Plan -> Execute -> Update)."""
    runtime = InMemoryRuntime(
        model_id="gpt2",
        device="cpu",
        precision=PrecisionProfile(weight_dtype="float32", activation_dtype="float32"),
    )
    science_engine = ClosedLoopScienceEngine(runtime=runtime, model_id="gpt2", device="cpu")

    result = science_engine.run_autonomous_investigation(
        component_layer=8,
        component_index=412,
        behavior_name="french_capital_active_science",
        clean_prompt="The capital of France is",
        target_token=" Paris",
        max_iterations=3,
    )

    assert isinstance(result, ClosedLoopInvestigationResult)
    assert result.total_iterations_run > 0
    assert len(result.iteration_history) == result.total_iterations_run
    assert result.final_entropy_bits < result.initial_entropy_bits
    assert result.final_probabilities[result.surviving_dominant_hypothesis] > 0.50
    assert len(result.formatted_certificate_markdown) > 100
