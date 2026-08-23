"""Unit and integration tests for Phase 25: Model Criticism & Robust Uncertainty."""

import pytest
import torch

from backend.discovery.calibration_policy import CalibrationPolicyConfig
from backend.discovery.distribution_family_models import (
    DistributionFamilyType,
    MultiFamilyDistributionFitter,
)
from backend.discovery.model_criticism_engine import (
    CriticEpistemicState,
    CriticResult,
    ModelCriticismEngine,
)
from backend.discovery.robust_closed_loop_scientist import (
    RobustClosedLoopScientist,
    RobustScientistResult,
)
from backend.discovery.robust_eig_planner import (
    CalibrationAwareEIGPlanner,
    CalibrationAwareOptimalSelection,
    ExperimentDecision,
)
from backend.discovery.experiment_design_generator import ExperimentDesignGenerator
from backend.runtime.in_memory_runtime import InMemoryRuntime
from backend.runtime.interfaces import PrecisionProfile


def test_multi_family_distribution_fitting():
    """Verifies that MultiFamilyDistributionFitter fits Gaussian, Student-t, GMM, and KDE models."""
    fitter = MultiFamilyDistributionFitter()
    # Deterministic synthetic bimodal distribution
    samples = [0.02, 0.03, 0.02, 0.04, 0.28, 0.30, 0.29, 0.31]

    prof_g = fitter.fit_gaussian(samples)
    prof_t = fitter.fit_student_t(samples, fixed_nu=4.0)
    prof_gmm = fitter.fit_2component_gmm(samples)
    prof_kde = fitter.fit_empirical_kde(samples)

    assert prof_g.family_type == DistributionFamilyType.GAUSSIAN
    assert prof_t.family_type == DistributionFamilyType.STUDENT_T
    assert prof_gmm.family_type == DistributionFamilyType.GAUSSIAN_MIXTURE_GMM
    assert prof_kde.family_type == DistributionFamilyType.EMPIRICAL_KDE

    # GMM should capture bimodal clusters with higher log-likelihood than single Gaussian
    assert prof_gmm.log_likelihood > prof_g.log_likelihood


def test_model_criticism_3tier_policy_and_abstention():
    """Verifies that ModelCriticismEngine enforces 3-tier policy: VALID_CONFIDENT, VALID_UNCERTAIN, ABSTAIN."""
    policy = CalibrationPolicyConfig(
        valid_confident_threshold=0.12,
        valid_uncertain_threshold=0.20,
        min_samples_required=5,
    )
    critic = ModelCriticismEngine(policy)

    # 1. High-coverage well-calibrated data -> VALID_CONFIDENT
    train_cal = [0.28, 0.29, 0.27, 0.30, 0.28, 0.29]
    test_cal = [0.28, 0.29, 0.28]
    res_conf = critic.critique_predictive_outcome_model("H1_Relational", "POSITIONAL_PERTURBATION", train_cal, test_cal)
    assert res_conf.epistemic_state == CriticEpistemicState.VALID_CONFIDENT
    assert res_conf.mean_coverage_error <= 0.12
    assert len(res_conf.calibration_probe_agenda) == 0

    # 2. Severe undercoverage (outliers breaking 90% bounds) -> ABSTAIN
    train_mis = [0.28, 0.28, 0.28, 0.28]
    test_mis = [0.01, 0.95, 0.02]  # Heavy tail shocks
    res_abstain = critic.critique_predictive_outcome_model("H1_Relational", "POSITIONAL_PERTURBATION", train_mis, test_mis)
    assert res_abstain.epistemic_state == CriticEpistemicState.ABSTAIN
    assert res_abstain.mean_coverage_error > 0.20
    assert len(res_abstain.calibration_probe_agenda) > 0


def test_calibration_aware_eig_planner_state_behavior():
    """Verifies that CalibrationAwareEIGPlanner reacts to CriticResult states and emits auditable ExperimentDecision records."""
    planner = CalibrationAwareEIGPlanner()
    generator = ExperimentDesignGenerator()
    candidates = generator.generate_candidate_battery(component_id="L8_N412")
    current_probs = {"H1_Relational": 0.25, "H2_BroadTopic": 0.25, "H3_LexicalTrigger": 0.25, "H4_PositionalArtifact": 0.25}

    # Case A: Well-calibrated data -> normal EIG
    train_good = [0.28, 0.29, 0.27, 0.30, 0.28, 0.29]
    test_good = [0.28, 0.29]
    sel_good = planner.select_experiment_with_criticism(current_probs, candidates, train_good, test_good)
    assert sel_good.decision_record.is_calibration_probe_mode is False
    assert sel_good.decision_record.calibrated_eig_bits > 0.0
    assert sel_good.decision_record.abstention_reason is None

    # Case B: Insufficient data -> ABSTAIN -> Probe Mode
    sel_sparse = planner.select_experiment_with_criticism(current_probs, candidates, [0.28], [])
    assert sel_sparse.decision_record.calibration_state == CriticEpistemicState.ABSTAIN.value
    assert sel_sparse.decision_record.is_calibration_probe_mode is True
    assert sel_sparse.decision_record.abstention_reason is not None
    assert len(sel_sparse.decision_record.calibration_probe_agenda) > 0


def test_robust_closed_loop_scientist_end_to_end():
    """Verifies full execution of RobustClosedLoopScientist with self-critical model validation."""
    runtime = InMemoryRuntime(
        model_id="gpt2",
        device="cpu",
        precision=PrecisionProfile(weight_dtype="float32", activation_dtype="float32"),
    )
    scientist = RobustClosedLoopScientist(runtime=runtime, model_id="gpt2", device="cpu")

    result = scientist.run_robust_investigation(
        component_layer=8,
        component_index=412,
        behavior_name="french_capital_robust_science",
        clean_prompt="The capital of France is",
        target_token=" Paris",
        max_iterations=3,
    )

    assert isinstance(result, RobustScientistResult)
    assert result.total_iterations_run > 0
    assert result.final_entropy_bits < result.initial_entropy_bits
    assert result.last_critic_result is not None
    assert len(result.decision_history) > 0
    assert len(result.formatted_certificate_markdown) > 100
