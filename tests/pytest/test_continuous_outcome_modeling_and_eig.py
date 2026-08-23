"""Unit and integration tests for Phase 24: Continuous Causal Outcome Modeling & EIG."""

import pytest
import torch

from backend.discovery.continuous_calibration_tracker import (
    ContinuousCalibrationRecord,
    ContinuousCalibrationSummary,
    ContinuousCalibrationTracker,
)
from backend.discovery.continuous_closed_loop_scientist import (
    ContinuousClosedLoopScientist,
    ContinuousInvestigationResult,
)
from backend.discovery.continuous_eig_planner import (
    ContinuousEIGPlanner,
    ContinuousOptimalSelection,
    ContinuousScoredExperimentCandidate,
)
from backend.discovery.continuous_outcome_likelihood_engine import (
    ContinuousMeasurementVector,
    ContinuousOutcomeLikelihoodEngine,
    GaussianOutcomeParam,
)
from backend.discovery.experiment_design_generator import ExperimentCategory, ExperimentDesignGenerator
from backend.runtime.in_memory_runtime import InMemoryRuntime
from backend.runtime.interfaces import PrecisionProfile


def test_continuous_outcome_likelihood_normal_gamma_updates():
    """Verifies that ContinuousOutcomeLikelihoodEngine computes Gaussian densities and updates parameters online."""
    engine = ContinuousOutcomeLikelihoodEngine()
    cat = ExperimentCategory.POSITIONAL_PERTURBATION.value
    hyp = "H1_Relational"

    vec_strong = ContinuousMeasurementVector(
        delta_z=0.30,
        delta_probability=0.35,
        mediation_rescue_fraction=0.78,
        control_specificity_ratio=5.10,
    )

    density_initial = engine.compute_continuous_density(hyp, cat, vec_strong)
    assert density_initial > 0.0

    # Observe multiple strong continuous measurements
    for _ in range(5):
        engine.update_with_continuous_observation(hyp, cat, vec_strong)

    density_updated = engine.compute_continuous_density(hyp, cat, vec_strong)
    # Density around the observed vector should increase after updates
    assert density_updated >= density_initial


def test_continuous_differential_eig_calculation():
    """Verifies that ContinuousEIGPlanner computes differential entropy and Mutual Information EIG."""
    planner = ContinuousEIGPlanner()
    generator = ExperimentDesignGenerator()

    candidates = generator.generate_candidate_battery(component_id="L8_N412")
    current_probs = {"H1_Relational": 0.25, "H2_BroadTopic": 0.25, "H3_LexicalTrigger": 0.25, "H4_PositionalArtifact": 0.25}

    selection = planner.evaluate_and_select_best_continuous_experiment(current_probs, candidates)
    assert isinstance(selection, ContinuousOptimalSelection)
    assert selection.best_experiment is not None
    assert selection.best_experiment.continuous_eig_bits > 0.0
    assert selection.best_experiment.total_scientific_utility_u > 0.0
    assert len(selection.ranked_candidates) == len(candidates)


def test_continuous_calibration_and_mahalanobis_tracking():
    """Verifies that ContinuousCalibrationTracker computes NLL loss, Mahalanobis distance, and regret."""
    tracker = ContinuousCalibrationTracker()
    vec = ContinuousMeasurementVector(delta_z=0.28, delta_probability=0.32, mediation_rescue_fraction=0.76, control_specificity_ratio=4.8)

    rec = tracker.record_continuous_experiment_outcome(
        experiment_id="EXP_CONT_01",
        experiment_category="POSITIONAL_PERTURBATION",
        true_hypothesis_id="H1_Relational",
        observed_vector=vec,
        predicted_eig_bits=0.65,
        prior_entropy_bits=2.0,
        posterior_entropy_bits=1.38,
        oracle_max_oig=0.72,
    )

    assert isinstance(rec, ContinuousCalibrationRecord)
    assert rec.observed_oig_bits == round(2.0 - 1.38, 4)
    assert isinstance(rec.log_likelihood_loss, float)
    assert rec.mahalanobis_distance_error >= 0.0
    assert rec.selection_regret_bits >= 0.0


    summary = tracker.get_continuous_telemetry_summary()
    assert summary.total_continuous_experiments_evaluated == 1
    assert summary.mean_mahalanobis_error >= 0.0


def test_continuous_closed_loop_scientist_end_to_end():
    """Verifies full execution of ContinuousClosedLoopScientist with continuous multi-metric interventions."""
    runtime = InMemoryRuntime(
        model_id="gpt2",
        device="cpu",
        precision=PrecisionProfile(weight_dtype="float32", activation_dtype="float32"),
    )
    scientist = ContinuousClosedLoopScientist(runtime=runtime, model_id="gpt2", device="cpu")

    result = scientist.run_continuous_investigation(
        component_layer=8,
        component_index=412,
        behavior_name="french_capital_continuous_science",
        clean_prompt="The capital of France is",
        target_token=" Paris",
        max_iterations=3,
    )

    assert isinstance(result, ContinuousInvestigationResult)
    assert result.total_iterations_run > 0
    assert result.final_entropy_bits < result.initial_entropy_bits
    assert result.last_observed_continuous_measurements.delta_z >= 0.0
    assert result.continuous_calibration_telemetry.total_continuous_experiments_evaluated > 0
    assert len(result.formatted_certificate_markdown) > 100
