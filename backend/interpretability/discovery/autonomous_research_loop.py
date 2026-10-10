"""Autonomous Research Loop Orchestrator.

Implements the continuous closed-loop scientific method for mechanistic interpretability:
Observe ➔ Generate Hypothesis ➔ Select Algorithms ➔ Run Experiments ➔ 
Collect Evidence ➔ Update Confidence ➔ Need More Evidence? (Iterate) ➔ Mechanism Claim.

Emits standardized ReasoningTrace objects to feed ReasoningTraceView and EvidenceFusionView.
"""

from __future__ import annotations

from backend.core.identifiers import entity_id

import datetime as _dt
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.science.models.adapter_base import ModelAdapter
from .algorithms import get_algorithm
from .algorithms.base_algorithm import DiscoveryReport
from .discovery_planner import AutonomousDiscoveryPlanner, ResearchGoal, MechanismClaim
from backend.research_datasets.dataset_manager import DatasetManager


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
    #: Whether this iteration actually measured anything. `confidence_delta` is
    #: 0.0 when it did not, so a delta of 0.0 alone cannot distinguish "no change"
    #: from "nothing measured" -- this can.
    measured: bool = False


@dataclass
class AutonomousCampaignReport:
    """Final artifact emitted by the Autonomous Research Loop."""
    campaign_id: str
    goal_description: str
    total_iterations: int
    #: Optional. None when no iteration measured anything, which is not the same
    #: as a campaign that measured and fell short.
    final_composite_confidence: Optional[float]
    target_confidence_reached: bool
    #: Whether `target_confidence_reached` was assessable at all.
    target_confidence_assessed: bool = False
    #: How many iterations produced a measurement.
    evidence_gathered: int = 0
    iterations_history: List[LoopIterationResult] = field(default_factory=list)
    mechanism_claim: Dict[str, Any] = field(default_factory=dict)
    reasoning_trace: Dict[str, Any] = field(default_factory=dict)
    total_runtime_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "goal_description": self.goal_description,
            "total_iterations": self.total_iterations,
            "final_composite_confidence": self.final_composite_confidence,
            "target_confidence_reached": self.target_confidence_reached,
            "target_confidence_assessed": self.target_confidence_assessed,
            "evidence_gathered": self.evidence_gathered,
            "iterations_history": [
                {
                    "iteration": r.iteration_index,
                    "algorithm": r.algorithm_run,
                    "hypothesis": r.hypothesis_tested,
                    "confidence": r.current_composite_confidence,
                    "measured": r.measured,
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
        self.dataset_manager = dataset_manager or DatasetManager("backend/research_datasets")
        self.planner = AutonomousDiscoveryPlanner(adapter=adapter, dataset_manager=self.dataset_manager)
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations

    def run_campaign(self, goal: ResearchGoal) -> AutonomousCampaignReport:
        """Executes the closed-loop autonomous research cycle until confidence threshold is met."""
        t0 = time.time()
        campaign_id = entity_id("campaign_")

        plan = self.planner.plan(goal)
        prompts = self.dataset_manager.load(goal.dataset_name)
        dataset_item = prompts[0] if prompts else {"clean": "John gave a drink to Mary", "target": " Mary"}

        current_confidence = 0.50
        history: List[LoopIterationResult] = []
        collected_evidence: List[Dict[str, Any]] = []
        # Defined before the loop: with no adapter the loop exits on its first
        # check, so anything computed inside it may never be assigned. The
        # summary below depends on this.
        measured_so_far = False

        current_hypothesis = f"Primary computational circuit for {goal.target_behavior}"

        # ── Closed-Loop Iteration ─────────────────────────────────────────────
        for it_idx in range(1, self.max_iterations + 1):
            if current_confidence >= self.confidence_threshold:
                break
            # Without an adapter no stage can measure, so further iterations
            # cannot either. Stopping here rather than looping `max_iterations`
            # times is not only cheaper: it means the history says the campaign
            # stopped for lack of evidence instead of recording N identical
            # no-evidence iterations.
            if self.adapter is None:
                break

            # Pick next algorithm from planned stages or dynamically generate next step
            stage_idx = min(it_idx - 1, len(plan.stages) - 1)
            stage = plan.stages[stage_idx]
            algorithm_name = stage.algorithm_name

            falsified = False
            evidence_data = {}
            iteration_measured = False

            if self.adapter:
                alg = get_algorithm(algorithm_name, self.adapter)
                report: DiscoveryReport = alg.run(dataset_item)

                score = report.confidence
                evidence_data = report.statistics
                # The report's own verdict on whether it measured takes
                # precedence over the confidence value. ACDC always reports
                # confidence 0.0 -- calibrated to "no confidence", never a
                # measurement -- and declares provenance.measured False when
                # its sweep evaluated nothing (e.g. mock adapter). Reading
                # `score is not None` alone counted that 0.0 as evidence and
                # marked the iteration measured. Algorithms whose reports
                # carry no measured key keep the legacy rule.
                prov = (report.provenance
                        if isinstance(report.provenance, dict) else {})
                if "measured" in prov:
                    iteration_measured = prov.get("measured") is True
                    if not iteration_measured:
                        score = None
                else:
                    iteration_measured = score is not None

                if algorithm_name == "causal_scrubbing":
                    # `get("validated", True)` made a scrubbing report with no
                    # `validated` key count as a PASSED falsification. Silence
                    # from the one stage that can refute a hypothesis is not
                    # support for it.
                    verdict = report.statistics.get("validated")
                    if verdict is None:
                        iteration_measured = False
                    elif not verdict:
                        falsified = True
                        if score is not None:
                            score *= 0.3
                        current_hypothesis = (
                            f"Revised: {current_hypothesis} "
                            f"(incorporating counterexample constraints)")

                if iteration_measured:
                    conf_delta = (score - current_confidence) * 0.5
                    current_confidence = min(0.99, max(0.10,
                                                       current_confidence + conf_delta))
                else:
                    # A stage that measured nothing must not move the loop's
                    # confidence in either direction. Treating None as a score
                    # would either crash or, with a 0.0 default, drive the
                    # confidence down for a stage that was merely honest.
                    conf_delta = 0.0

                collected_evidence.append({
                    "type": algorithm_name,
                    "score": None if score is None else round(score, 3),
                    "measured": iteration_measured,
                    "statistics": report.statistics
                })
            else:
                # No adapter: nothing ran, so nothing is learned.
                #
                # Was `score = 0.85 + 0.03 * it_idx`, `conf_delta = 0.08`, and
                # `current_confidence += 0.08` capped at 0.95 -- a synthetic
                # climb that started at 0.50 and reached any threshold within
                # `max_iterations`, then emitted `STOP_THRESHOLD_MET`. The loop
                # was guaranteed to report success having measured nothing, and
                # `final_composite_confidence` was that invented number.
                iteration_measured = False
                score = None
                conf_delta = 0.0
                collected_evidence.append({
                    "type": algorithm_name,
                    "score": None,
                    "measured": False,
                    "reason": ("No adapter connected, so no algorithm ran. "
                               "This iteration measured nothing."),
                })

            # The stopping condition has to distinguish "reached the threshold" from
            # "ran out of iterations without ever measuring". Otherwise a loop
            # that measured nothing reports a clean stop at its starting
            # confidence and a caller cannot tell the two apart.
            measured_so_far = measured_so_far or any(
                e.get("measured") for e in collected_evidence)
            if current_confidence >= self.confidence_threshold and measured_so_far:
                next_action = "STOP_THRESHOLD_MET"
            elif not measured_so_far:
                next_action = "STOP_NO_EVIDENCE"
            else:
                next_action = f"PLAN_NEXT: Run Stage {it_idx + 1}"

            history.append(LoopIterationResult(
                iteration_index=it_idx,
                algorithm_run=algorithm_name,
                hypothesis_tested=current_hypothesis,
                confidence_delta=round(conf_delta, 4),
                current_composite_confidence=round(current_confidence, 4),
                evidence_gathered=evidence_data,
                falsified=falsified,
                measured=iteration_measured,
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
                    # Was the literals `semantic_confidence: 0.75` and
                    # `replication_score: 0.95`. Neither was computed by
                    # anything. `replication_score` in particular claimed
                    # replication for a single-model, single-seed campaign that
                    # performs none -- the same rung that was removed from
                    # `live_discovery` for exactly this reason.
                    "semantic_confidence": None,
                    "experimental_confidence": (round(current_confidence, 2)
                                                if measured_so_far else None),
                    "replication_score": None,
                    "replication_performed": False,
                    "replication_reason": (
                        "Single model, single seed, no repeated runs: this "
                        "campaign performs no replication to score."),
                    "status": ("accepted" if (measured_so_far
                                              and current_confidence >= 0.80)
                               else "rejected" if measured_so_far
                               else "unevaluated"),
                }
            ],
            # Only selected once something was actually measured. Selecting an
            # unevaluated hypothesis is a decision the trace did not make.
            "selected_hypothesis_id": ("H_FINAL" if measured_so_far else None),
            "experiments": [
                {
                    "type": h.algorithm_run,
                    "description": (
                        f"Iteration #{h.iteration_index}: tested via "
                        f"{h.algorithm_run}. "
                        + (f"Confidence: {h.current_composite_confidence:.2f}"
                           if h.measured else "measured nothing")),
                    "measured": h.measured,
                    "verdict": ("unevaluated" if not h.measured
                                else "rejected" if h.falsified else "accepted"),
                }
                for h in history
            ],
            "measured": measured_so_far,
            "final_confidence": (round(current_confidence, 4)
                                 if measured_so_far else None),
        }

        total_runtime = (time.time() - t0) * 1000

        return AutonomousCampaignReport(
            campaign_id=campaign_id,
            goal_description=goal.description,
            total_iterations=len(history),
            # `final_composite_confidence` was `round(current_confidence, 4)`
            # unconditionally -- and with no adapter `current_confidence` was the
            # synthetic 0.50 + 0.08 per iteration. `target_confidence_reached`
            # was then computed against that invented number, so a campaign that
            # measured nothing could report having reached its target.
            #
            # Both are None unless something was measured, and
            # `target_confidence_reached` is False when unmeasured: "did not
            # reach the target" and "cannot say" must not collapse.
            final_composite_confidence=(round(current_confidence, 4)
                                        if measured_so_far else None),
            target_confidence_reached=(current_confidence >= self.confidence_threshold
                                       if measured_so_far else False),
            target_confidence_assessed=measured_so_far,
            evidence_gathered=len([e for e in collected_evidence
                                   if e.get("measured")]),
            iterations_history=history,
            mechanism_claim=mechanism_claim.to_dict(),
            reasoning_trace=reasoning_trace,
            total_runtime_ms=total_runtime
        )
