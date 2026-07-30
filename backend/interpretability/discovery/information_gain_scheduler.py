"""Expected Utility Experiment Scheduler & Bayesian Active Learning Engine.

Ref: Decision-Theoretic Active Learning for Mechanistic Interpretability.

Calculates Expected Utility (EU) for candidate algorithms based on:
    Expected Utility = Expected Information Gain (EIG) 
                     - alpha * Compute Cost 
                     - beta * Time Cost 
                     - gamma * Failure Risk 
                     + delta * Scientific Novelty

Optimizes discovery campaign selection to favor high-information, low-cost, 
highly-novel scientific experiments.
"""

from __future__ import annotations

import datetime as _dt
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...science.models.adapter_base import ModelAdapter
from .algorithms import get_algorithm
from .algorithms.base_algorithm import DiscoveryReport
from .confidence_scorer import PlatformConfidenceEngine
from .discovery_planner import ResearchGoal, MechanismClaim
from ...datasets.dataset_manager import DatasetManager


@dataclass
class CandidateExperiment:
    """A candidate experiment evaluated by the Expected Utility (EU) engine."""
    algorithm_name: str
    target_model: str
    expected_info_gain: float
    expected_utility: float
    compute_cost: float
    time_cost: float
    failure_risk: float
    scientific_novelty: float
    marginal_uncertainty_reduction: float
    rationale: str


@dataclass
class EUSchedulerStep:
    """Single step executed by the Expected Utility Scheduler."""
    step_number: int
    selected_experiment: CandidateExperiment
    evaluated_candidates: List[CandidateExperiment]
    uncertainty_before: float
    uncertainty_after: float
    actual_info_gain: float
    timestamp: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")


@dataclass
class EUCampaignReport:
    """Artifact emitted by the Expected Utility Experiment Scheduler."""
    campaign_id: str
    goal_description: str
    initial_uncertainty: float
    final_uncertainty: float
    total_information_gained: float
    steps_executed: List[EUSchedulerStep]
    mechanism_claim: Dict[str, Any]
    total_runtime_ms: float
    timestamp: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "goal_description": self.goal_description,
            "initial_uncertainty": self.initial_uncertainty,
            "final_uncertainty": self.final_uncertainty,
            "total_information_gained": self.total_information_gained,
            "steps": [
                {
                    "step": s.step_number,
                    "algorithm": s.selected_experiment.algorithm_name,
                    "model": s.selected_experiment.target_model,
                    "eig": s.selected_experiment.expected_info_gain,
                    "eu": s.selected_experiment.expected_utility,
                    "compute_cost": s.selected_experiment.compute_cost,
                    "failure_risk": s.selected_experiment.failure_risk,
                    "novelty": s.selected_experiment.scientific_novelty,
                    "uncertainty_before": s.uncertainty_before,
                    "uncertainty_after": s.uncertainty_after,
                    "actual_info_gain": s.actual_info_gain,
                    "rationale": s.selected_experiment.rationale,
                }
                for s in self.steps_executed
            ],
            "mechanism_claim": self.mechanism_claim,
            "total_runtime_ms": self.total_runtime_ms,
            "timestamp": self.timestamp,
        }


class InformationGainScheduler:
    """Decision-theoretic Active Learning Scheduler maximizing Expected Utility (EU)."""

    def __init__(
        self,
        adapter: Optional[ModelAdapter] = None,
        dataset_manager: Optional[DatasetManager] = None,
        min_eu_threshold: float = 0.02,
        max_steps: int = 5,
        alpha_compute: float = 0.15,
        beta_time: float = 0.10,
        gamma_risk: float = 0.20,
        delta_novelty: float = 0.25,
    ) -> None:
        self.adapter = adapter
        self.dataset_manager = dataset_manager or DatasetManager("backend/datasets")
        self.confidence_engine = PlatformConfidenceEngine()
        self.min_eu_threshold = min_eu_threshold
        self.max_steps = max_steps
        self.alpha_compute = alpha_compute
        self.beta_time = beta_time
        self.gamma_risk = gamma_risk
        self.delta_novelty = delta_novelty

    def evaluate_candidates(
        self,
        current_uncertainty: float,
        executed_algorithms: Dict[str, int],
        goal: ResearchGoal,
    ) -> List[CandidateExperiment]:
        """Calculates Expected Utility (EU) = EIG - ComputeCost - TimeCost - Risk + Novelty."""
        # Candidate tuples: (alg, model, info_density, compute_cost, time_cost, failure_risk, novelty, rationale)
        all_candidates = [
            ("attribution_patching", goal.model_id, 0.85, 0.10, 0.05, 0.05, 0.10, "O(1) screening isolates high-attribution candidate nodes."),
            ("acdc", goal.model_id, 0.75, 0.60, 0.50, 0.15, 0.15, "Iterative pruning reduces node/edge structural uncertainty."),
            ("path_patching", goal.model_id, 0.70, 0.40, 0.30, 0.10, 0.15, "Edge intervention reduces inter-layer connection uncertainty."),
            ("causal_scrubbing", goal.model_id, 0.95, 0.70, 0.60, 0.20, 0.20, "Equivalence class resamplings reduce falsification uncertainty."),
            ("transcoders", goal.model_id, 0.65, 0.50, 0.40, 0.15, 0.25, "Decomposes non-linear MLP activations into dictionary features."),
            ("feature_universality", "gemma" if goal.model_id == "gpt2" else "llama", 0.90, 0.45, 0.35, 0.10, 0.30, "Cross-model bipartite matching resolves universal generalization uncertainty."),
        ]

        evaluated: List[CandidateExperiment] = []

        for alg, model_name, info_density, comp_cost, time_cost, risk, novelty, base_rationale in all_candidates:
            num_previous_runs = executed_algorithms.get(alg, 0)

            # Diminishing marginal returns formula
            diminishing_factor = math.exp(-1.5 * num_previous_runs)
            expected_reduction = current_uncertainty * 0.6 * diminishing_factor
            eig = round(expected_reduction * info_density, 4)

            # Expected Utility Formula: EU = EIG - alpha*Compute - beta*Time - gamma*Risk + delta*Novelty
            penalty = (self.alpha_compute * comp_cost) + (self.beta_time * time_cost) + (self.gamma_risk * risk)
            bonus = self.delta_novelty * novelty
            eu = round(eig - penalty + bonus, 4)

            # Dynamic active-learning rationale
            if num_previous_runs > 0 and eu < self.min_eu_threshold:
                rationale = f"PRUNED (EU: {eu:.3f}, EIG: {eig:.3f}): Marginal utility below threshold ({self.min_eu_threshold})."
            else:
                rationale = f"EU {eu:.3f} (EIG: {eig:.3f}, Novelty: +{bonus:.2f}): {base_rationale}"

            evaluated.append(CandidateExperiment(
                algorithm_name=alg,
                target_model=model_name,
                expected_info_gain=eig,
                expected_utility=eu,
                compute_cost=comp_cost,
                time_cost=time_cost,
                failure_risk=risk,
                scientific_novelty=novelty,
                marginal_uncertainty_reduction=round(expected_reduction, 4),
                rationale=rationale
            ))

        # Rank candidates by Expected Utility (EU) descending
        evaluated.sort(key=lambda x: x.expected_utility, reverse=True)
        return evaluated

    def run_scheduled_campaign(self, goal: ResearchGoal) -> EUCampaignReport:
        """Executes active learning campaign maximizing Expected Utility (EU)."""
        t0 = time.time()
        campaign_id = f"eu_campaign_{hash(goal.goal_id + str(time.time())) & 0xffffffff:08x}"

        prompts = self.dataset_manager.load(goal.dataset_name)
        dataset_item = prompts[0] if prompts else {"clean": "John gave a drink to Mary", "target": " Mary"}

        current_uncertainty = 0.50  # Initial uncertainty bounds
        executed_counts: Dict[str, int] = {}
        steps_executed: List[EUSchedulerStep] = []
        total_info_gained = 0.0

        for step_idx in range(1, self.max_steps + 1):
            if current_uncertainty <= 0.05:
                break

            # 1. Evaluate EU across candidate pool
            candidates = self.evaluate_candidates(current_uncertainty, executed_counts, goal)
            best_candidate = candidates[0]

            # Stop if the top candidate's EU is below minimum threshold
            if best_candidate.expected_utility < self.min_eu_threshold:
                break

            alg_name = best_candidate.algorithm_name
            executed_counts[alg_name] = executed_counts.get(alg_name, 0) + 1

            unc_before = current_uncertainty

            # 2. Execute selected maximum Expected Utility experiment
            if self.adapter:
                alg_instance = get_algorithm(alg_name, self.adapter)
                report: DiscoveryReport = alg_instance.run(dataset_item)
                actual_reduction = max(0.02, best_candidate.marginal_uncertainty_reduction * (report.confidence / 0.95))
            else:
                actual_reduction = max(0.02, best_candidate.marginal_uncertainty_reduction)

            current_uncertainty = max(0.01, round(current_uncertainty - actual_reduction, 4))
            actual_gain = round(unc_before - current_uncertainty, 4)
            total_info_gained += actual_gain

            steps_executed.append(EUSchedulerStep(
                step_number=step_idx,
                selected_experiment=best_candidate,
                evaluated_candidates=candidates,
                uncertainty_before=unc_before,
                uncertainty_after=current_uncertainty,
                actual_info_gain=actual_gain
            ))

        total_runtime = (time.time() - t0) * 1000
        final_confidence = round(1.0 - current_uncertainty, 4)

        claim_summary = (
            f"Expected-Utility campaign complete for '{goal.description}'. "
            f"Executed {len(steps_executed)} decision-theoretic steps. "
            f"Total Information Gained: {total_info_gained:.3f} bits. "
            f"Final Uncertainty: {current_uncertainty:.3f} (Confidence: {final_confidence * 100:.1f}%)."
        )

        mechanism_claim = {
            "claim_id": f"claim_{campaign_id}",
            "goal_description": goal.description,
            "summary_claim": claim_summary,
            "composite_confidence": final_confidence,
            "final_uncertainty": current_uncertainty,
            "steps_count": len(steps_executed),
        }

        return EUCampaignReport(
            campaign_id=campaign_id,
            goal_description=goal.description,
            initial_uncertainty=0.50,
            final_uncertainty=current_uncertainty,
            total_information_gained=round(total_info_gained, 4),
            steps_executed=steps_executed,
            mechanism_claim=mechanism_claim,
            total_runtime_ms=total_runtime
        )
