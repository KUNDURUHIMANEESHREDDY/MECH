"""Calibration-Aware EIG Planner for MECH.

Consumes CriticResult directly:
- VALID_CONFIDENT: EIG_critic = EIG * (1 - C_err)
- VALID_UNCERTAIN: EIG_critic = EIG * max(0.10, 1 - 2.5 * C_err)
- ABSTAIN: Switches to Calibration Probe Exploration Mode
Emits auditable ExperimentDecision records.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .belief_entropy_engine import BeliefEntropyEngine, HypothesisEntropyProfile
from .calibration_policy import CalibrationPolicyConfig
from .continuous_eig_planner import ContinuousEIGPlanner, ContinuousScoredExperimentCandidate
from .experiment_design_generator import CandidateExperimentDesign
from .model_criticism_engine import CriticEpistemicState, CriticResult, ModelCriticismEngine


@dataclass
class ExperimentDecision:
    """Immutable, auditable record capturing the planner's decision and epistemic state."""
    selected_experiment_id: str
    experiment_category: str
    distribution_family: str
    calibration_state: str
    predicted_mean_delta_z: float
    predicted_variance_delta_z: float
    predicted_eig_bits: float
    calibrated_eig_bits: float
    causal_relevance_weight: float
    validity_weight: float
    compute_passes: int
    total_scientific_utility_u: float
    is_calibration_probe_mode: bool
    abstention_reason: Optional[str]
    calibration_probe_agenda: List[str]
    policy_provenance: Dict[str, Any]
    decision_rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_experiment_id": self.selected_experiment_id,
            "experiment_category": self.experiment_category,
            "distribution_family": self.distribution_family,
            "calibration_state": self.calibration_state,
            "predicted_mean_delta_z": round(self.predicted_mean_delta_z, 4),
            "predicted_variance_delta_z": round(self.predicted_variance_delta_z, 6),
            "predicted_eig_bits": round(self.predicted_eig_bits, 4),
            "calibrated_eig_bits": round(self.calibrated_eig_bits, 4),
            "causal_relevance_weight": self.causal_relevance_weight,
            "validity_weight": self.validity_weight,
            "compute_passes": self.compute_passes,
            "total_scientific_utility_u": round(self.total_scientific_utility_u, 4),
            "is_calibration_probe_mode": self.is_calibration_probe_mode,
            "abstention_reason": self.abstention_reason,
            "calibration_probe_agenda": self.calibration_probe_agenda,
            "policy_provenance": self.policy_provenance,
            "decision_rationale": self.decision_rationale,
        }


@dataclass
class CalibrationAwareOptimalSelection:
    """Optimal experiment decision made under calibration-aware model criticism."""
    best_experiment: ContinuousScoredExperimentCandidate
    critic_result: CriticResult
    decision_record: ExperimentDecision

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_experiment": self.best_experiment.to_dict(),
            "critic_result": self.critic_result.to_dict(),
            "decision_record": self.decision_record.to_dict(),
        }


class CalibrationAwareEIGPlanner:
    """Plans experiments while factoring model criticism and 3-tier epistemic state."""

    def __init__(
        self,
        critic_engine: Optional[ModelCriticismEngine] = None,
        continuous_planner: Optional[ContinuousEIGPlanner] = None,
        policy: Optional[CalibrationPolicyConfig] = None,
    ) -> None:
        self.policy = policy or (critic_engine.policy if critic_engine else CalibrationPolicyConfig())
        self.critic_engine = critic_engine or ModelCriticismEngine(self.policy)
        self.continuous_planner = continuous_planner or ContinuousEIGPlanner()

    def select_experiment_with_criticism(
        self,
        current_hypothesis_probabilities: Dict[str, float],
        candidate_battery: List[CandidateExperimentDesign],
        historical_training_samples: List[float],
        recent_heldout_samples: List[float],
    ) -> CalibrationAwareOptimalSelection:
        """Selects optimal experiment or diverts to calibration probe agenda if ABSTAIN is active."""
        # 1. Evaluate baseline continuous EIG across candidates
        cont_selection = self.continuous_planner.evaluate_and_select_best_continuous_experiment(
            current_hypothesis_probabilities=current_hypothesis_probabilities,
            candidate_battery=candidate_battery,
        )
        best_cand = cont_selection.best_experiment
        raw_eig = best_cand.continuous_eig_bits

        dom_hyp = max(current_hypothesis_probabilities.items(), key=lambda x: x[1])[0]

        # 2. Run Model Criticism on dominant hypothesis & selected experiment category
        critic_res = self.critic_engine.critique_predictive_outcome_model(
            hypothesis_id=dom_hyp,
            experiment_category=best_cand.candidate_design.category.value,
            training_samples=historical_training_samples,
            heldout_samples=recent_heldout_samples,
        )

        # 3. Apply 3-Tier Policy
        cost_norm = math.sqrt(max(1, best_cand.compute_passes))
        if critic_res.epistemic_state == CriticEpistemicState.VALID_CONFIDENT:
            cal_factor = max(0.10, 1.0 - critic_res.mean_coverage_error)
            cal_eig = raw_eig * cal_factor
            utility_u = (cal_eig * best_cand.causal_relevance * best_cand.validity_score) / cost_norm
            is_probe = False
            abstain_reason = None
            rationale = f"VALID_CONFIDENT: Planning under verified {critic_res.selected_family.value} distribution. Calibrated EIG={cal_eig:.4f} bits."
        elif critic_res.epistemic_state == CriticEpistemicState.VALID_UNCERTAIN:
            cal_factor = max(0.10, 1.0 - self.policy.uncertain_utility_penalty_multiplier * critic_res.mean_coverage_error)
            cal_eig = raw_eig * cal_factor
            utility_u = (cal_eig * best_cand.causal_relevance * best_cand.validity_score) / cost_norm
            is_probe = False
            abstain_reason = None
            rationale = f"VALID_UNCERTAIN: Applying conservative penalty (C_err={critic_res.mean_coverage_error:.3f}). Calibrated EIG={cal_eig:.4f} bits."
        else:  # ABSTAIN
            cal_eig = 0.05  # Minimal EIG for hypothesis discrimination
            utility_u = 0.05
            is_probe = True
            abstain_reason = critic_res.criticism_verdict_rationale
            rationale = f"ABSTAIN TRIGGERED: Refusing high-stakes hypothesis claims. Switching to Calibration Probe Agenda ({len(critic_res.calibration_probe_agenda)} probes)."

        decision = ExperimentDecision(
            selected_experiment_id=best_cand.candidate_design.experiment_id,
            experiment_category=best_cand.candidate_design.category.value,
            distribution_family=critic_res.selected_family.value,
            calibration_state=critic_res.epistemic_state.value,
            predicted_mean_delta_z=best_cand.predicted_mean_delta_z,
            predicted_variance_delta_z=best_cand.marginal_variance_delta_z,
            predicted_eig_bits=raw_eig,
            calibrated_eig_bits=cal_eig,
            causal_relevance_weight=best_cand.causal_relevance,
            validity_weight=best_cand.validity_score,
            compute_passes=best_cand.compute_passes,
            total_scientific_utility_u=utility_u,
            is_calibration_probe_mode=is_probe,
            abstention_reason=abstain_reason,
            calibration_probe_agenda=critic_res.calibration_probe_agenda,
            policy_provenance=self.policy.to_dict(),
            decision_rationale=rationale,
        )

        return CalibrationAwareOptimalSelection(
            best_experiment=best_cand,
            critic_result=critic_res,
            decision_record=decision,
        )
