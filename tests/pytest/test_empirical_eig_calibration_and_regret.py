"""Unit and integration tests for Phase 23: Empirical EIG Calibration & Selection Regret."""

import pytest
import torch

from backend.discovery.calibrated_closed_loop_scientist import (
    CalibratedClosedLoopScientist,
    CalibratedInvestigationResult,
)
from backend.discovery.calibrated_eig_planner import CalibratedEIGPlanner, CalibratedOptimalSelection
from backend.discovery.eig_calibration_tracker import (
    CalibrationTelemetrySummary,
    EIGCalibrationRecord,
    EIGCalibrationTracker,
)
from backend.discovery.empirical_outcome_likelihood_engine import (
    EmpiricalLikelihoodParam,
    EmpiricalOutcomeLikelihoodEngine,
)
from backend.discovery.experiment_design_generator import ExperimentCategory, ExperimentDesignGenerator
from backend.runtime.in_memory_runtime import InMemoryRuntime
from backend.runtime.interfaces import PrecisionProfile


def test_empirical_outcome_likelihood_updates():
    """Verifies that EmpiricalOutcomeLikelihoodEngine updates Beta parameters online upon observations."""
    engine = EmpiricalOutcomeLikelihoodEngine()
    cat = ExperimentCategory.POSITIONAL_PERTURBATION.value
    hyp = "H1_Relational"

    initial_p = engine.get_calibrated_likelihood(hyp, cat)
    assert 0.0 < initial_p < 1.0

    # Observe 5 HIGH outcomes
    for _ in range(5):
        engine.update_with_observation(hyp, cat, "HIGH")

    updated_p = engine.get_calibrated_likelihood(hyp, cat)
    assert updated_p > initial_p

    # Observe 10 LOW outcomes
    for _ in range(10):
        engine.update_with_observation(hyp, cat, "LOW")

    final_p = engine.get_calibrated_likelihood(hyp, cat)
    assert final_p < updated_p


def test_eig_calibration_error_calculation():
    """Verifies that EIGCalibrationTracker calculates OIG, ECE, and selection regret accurately."""
    tracker = EIGCalibrationTracker()

    rec = tracker.record_experiment_outcome(
        experiment_id="EXP_TEST_01",
        experiment_category="POSITIONAL_PERTURBATION",
        predicted_eig=0.75,
        prior_entropy_bits=2.0,
        posterior_entropy_bits=1.30,  # Observed OIG = 2.0 - 1.30 = 0.70
        oracle_max_oig=0.85,
    )

    assert isinstance(rec, EIGCalibrationRecord)
    assert abs(rec.observed_oig_bits - 0.70) < 1e-3
    assert abs(rec.calibration_error_bits - 0.05) < 1e-3  # |0.75 - 0.70| = 0.05
    assert abs(rec.selection_regret_bits - 0.15) < 1e-3   # 0.85 - 0.70 = 0.15

    telemetry = tracker.get_telemetry_summary()
    assert telemetry.total_experiments_evaluated == 1
    assert telemetry.mean_calibration_error_bits == 0.05
    assert telemetry.mean_selection_regret_bits == 0.15


def test_calibrated_eig_planner_selection():
    """Verifies that CalibratedEIGPlanner integrates empirical likelihoods and ranks experiments by utility."""
    likelihood_engine = EmpiricalOutcomeLikelihoodEngine()
    planner = CalibratedEIGPlanner(likelihood_engine=likelihood_engine)
    generator = ExperimentDesignGenerator()

    candidates = generator.generate_candidate_battery(component_id="L8_N412")
    current_probs = {"H1_Relational": 0.25, "H2_BroadTopic": 0.25, "H3_LexicalTrigger": 0.25, "H4_PositionalArtifact": 0.25}

    selection = planner.evaluate_and_select_best_experiment(current_probs, candidates)
    assert isinstance(selection, CalibratedOptimalSelection)
    assert selection.best_experiment is not None
    assert selection.best_experiment.expected_information_gain_eig > 0.0
    assert len(selection.ranked_candidates) == len(candidates)


def test_calibrated_closed_loop_scientist_end_to_end():
    """Verifies full execution of CalibratedClosedLoopScientist with online learning and calibration telemetry."""
    runtime = InMemoryRuntime(
        model_id="gpt2",
        device="cpu",
        precision=PrecisionProfile(weight_dtype="float32", activation_dtype="float32"),
    )
    scientist = CalibratedClosedLoopScientist(runtime=runtime, model_id="gpt2", device="cpu")

    result = scientist.run_calibrated_investigation(
        component_layer=8,
        component_index=412,
        behavior_name="french_capital_calibrated_science",
        clean_prompt="The capital of France is",
        target_token=" Paris",
        max_iterations=3,
    )

    assert isinstance(result, CalibratedInvestigationResult)
    assert result.total_iterations_run > 0
    assert result.final_entropy_bits < result.initial_entropy_bits
    assert result.calibration_telemetry.total_experiments_evaluated > 0
    assert result.calibration_telemetry.mean_calibration_error_bits >= 0.0
    assert result.calibration_telemetry.mean_selection_regret_bits >= 0.0
    assert len(result.formatted_certificate_markdown) > 100
