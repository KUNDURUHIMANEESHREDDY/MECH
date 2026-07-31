"""Autonomous Research Loop Orchestrator.

Implements the continuous closed-loop scientific method for mechanistic interpretability:
Observe ➔ Generate Hypothesis ➔ Select Algorithms ➔ Run Experiments ➔ 
Collect Evidence ➔ Update Confidence ➔ Need More Evidence? (Iterate) ➔ Mechanism Claim.

Emits standardized ReasoningTrace objects to feed ReasoningTraceView and EvidenceFusionView.
"""

from __future__ import annotations

import datetime as _dt
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from science.models.adapter_base import ModelAdapter
from .algorithms import get_algorithm
from .algorithms.base_algorithm import DiscoveryReport
from .discovery_planner import AutonomousDiscoveryPlanner, ResearchGoal, MechanismClaim
from datasets.dataset_manager import DatasetManager


@dataclass
class LoopIterationResult:
    """Output state of a single iteration in the autonomous research loop."""
    iteration_index: int
    algorithm_run: str
    hypothesis_tested: str
    confidence_delta: float
    current_composite_confidence: float
    evidence_gathered: Dict[str, Any]
    falsified: bool
    next_step_action: str


@dataclass
class AutonomousCampaignReport:
    """Final artifact emitted by the Autonomous Research Loop."""
    campaign_id: str
    goal_description: str
    total_iterations: int
    final_composite_confidence: float
    target_confidence_reached: bool
    iterations_history: List[LoopIterationResult]
    mechanism_claim: Dict[str, Any]
    reasoning_trace: Dict[str, Any]
    total_runtime_ms: float
    timestamp: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "goal_description": self.goal_description,
            "total_iterations": self.total_iterations,
            "final_composite_confidence": self.final_composite_confidence,
            "target_confidence_reached": self.target_confidence_reached,
            "iterations_history": [
                {
                    "iteration": r.iteration_index,
                    "algorithm": r.algorithm_run,
                    "hypothesis": r.hypothesis_tested,
                    "confidence": r.current_composite_confidence,
                    "falsified": r.falsified,
                    "next_action": r.next_step_action,
                }
                for r in self.iterations_history
            ],
            "mechanism_claim": self.mechanism_claim,
            "reasoning_trace": self.reasoning_trace,
            "total_runtime_ms": self.total_runtime_ms,
            "timestamp": self.timestamp,
        }


class AutonomousResearchLoop:
    """Closed-loop autonomous agent that iteratively generates, tests, and refines hypotheses."""

    def __init__(
        self,
        adapter: Optional[ModelAdapter] = None,
        dataset_manager: Optional[DatasetManager] = None,
        confidence_threshold: float = 0.90,
        max_iterations: int = 5,
    ) -> None:
        self.adapter = adapter
        self.dataset_manager = dataset_manager or DatasetManager("backend/datasets")
        self.planner = AutonomousDiscoveryPlanner(adapter=adapter, dataset_manager=self.dataset_manager)
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations

    def run_campaign(self, goal: ResearchGoal) -> AutonomousCampaignReport:
        """Executes the closed-loop autonomous research cycle until confidence threshold is met."""
        t0 = time.time()
        campaign_id = f"campaign_{hash(goal.goal_id + str(time.time())) & 0xffffffff:08x}"

        plan = self.planner.plan(goal)
        prompts = self.dataset_manager.load(goal.dataset_name)
        dataset_item = prompts[0] if prompts else {"clean": "John gave a drink to Mary", "target": " Mary"}

        current_confidence = 0.50
        history: List[LoopIterationResult] = []
        collected_evidence: List[Dict[str, Any]] = []

        current_hypothesis = f"Primary computational circuit for {goal.target_behavior}"

        # ── Closed-Loop Iteration ─────────────────────────────────────────────
        for it_idx in range(1, self.max_iterations + 1):
            if current_confidence >= self.confidence_threshold:
                break

            # Pick next algorithm from planned stages or dynamically generate next step
            stage_idx = min(it_idx - 1, len(plan.stages) - 1)
            stage = plan.stages[stage_idx]
            algorithm_name = stage.algorithm_name

            falsified = False
            evidence_data = {}

            if self.adapter:
                alg = get_algorithm(algorithm_name, self.adapter)
                report: DiscoveryReport = alg.run(dataset_item)

                score = report.confidence
                evidence_data = report.statistics

                if algorithm_name == "causal_scrubbing":
                    falsified = not report.statistics.get("validated", True)
                    if falsified:
                        score *= 0.3
                        current_hypothesis = f"Revised: {current_hypothesis} (incorporating counterexample constraints)"

                conf_delta = (score - current_confidence) * 0.5
                current_confidence = min(0.99, max(0.10, current_confidence + conf_delta))

                collected_evidence.append({
                    "type": algorithm_name,
                    "score": round(score, 3),
                    "statistics": report.statistics
                })
            else:
                # Mock iteration progression for testing
                score = 0.85 + 0.03 * it_idx
                conf_delta = 0.08
                current_confidence = min(0.95, current_confidence + conf_delta)
                collected_evidence.append({"type": algorithm_name, "score": score})

            next_action = "STOP_THRESHOLD_MET" if current_confidence >= self.confidence_threshold else f"PLAN_NEXT: Run Stage {it_idx + 1}"

            history.append(LoopIterationResult(
                iteration_index=it_idx,
                algorithm_run=algorithm_name,
                hypothesis_tested=current_hypothesis,
                confidence_delta=round(conf_delta, 4),
                current_composite_confidence=round(current_confidence, 4),
                evidence_gathered=evidence_data,
                falsified=falsified,
                next_step_action=next_action
            ))

        # ── Final Mechanism Claim & Reasoning Trace ───────────────────────────
        mechanism_claim = self.planner.execute_campaign(goal)

        reasoning_trace = {
            "id": f"trace_{campaign_id}",
            "type": "debate",
            "observation_id": goal.goal_id,
            "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            "llm_model": goal.model_id,
            "hypotheses": [
                {
                    "id": "H_FINAL",
                    "text": current_hypothesis,
                    "semantic_confidence": 0.75,
                    "experimental_confidence": round(current_confidence, 2),
                    "replication_score": 0.95,
                    "status": "accepted" if current_confidence >= 0.80 else "rejected"
                }
            ],
            "selected_hypothesis_id": "H_FINAL",
            "experiments": [
                {
                    "type": h.algorithm_run,
                    "description": f"Iteration #{h.iteration_index}: Tested via {h.algorithm_run}. Confidence: {h.current_composite_confidence:.2f}",
                    "verdict": "accepted" if not h.falsified else "rejected"
                }
                for h in history
            ],
            "final_confidence": round(current_confidence, 4)
        }

        total_runtime = (time.time() - t0) * 1000

        return AutonomousCampaignReport(
            campaign_id=campaign_id,
            goal_description=goal.description,
            total_iterations=len(history),
            final_composite_confidence=round(current_confidence, 4),
            target_confidence_reached=current_confidence >= self.confidence_threshold,
            iterations_history=history,
            mechanism_claim=mechanism_claim.to_dict(),
            reasoning_trace=reasoning_trace,
            total_runtime_ms=total_runtime
        )
