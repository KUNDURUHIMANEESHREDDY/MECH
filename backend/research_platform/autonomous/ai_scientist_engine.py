"""Central AI Scientist Platform Orchestrator (Sprint 5).

Coordinates multi-agent research society, scientific debate, research programs,
scientific validation, traceable evidence graph, capability registry, research critic,
uncertainty manager, experiment recommender, discovery prioritizer, roadmap generator,
consensus engine, literature integrator, and research governance.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional

from backend.core.capability_registry import CapabilityRegistry
from backend.core.evidence_graph import TraceableEvidenceGraph
from backend.core.workflow_dsl import DeclarativeWorkflowEngine
from backend.validation.validation_engine import ScientificValidationEngine
from .debate_engine import ScientificDebateEngine
from .discovery_prioritizer import DiscoveryPrioritizerEngine
from .experiment_recommender import ExperimentRecommendationEngine
from .literature_integrator import AutonomousLiteratureIntegrator
from .multi_agent_society import MultiAgentResearchSociety
from .research_critic_agent import ResearchCriticAgent
from .research_governance import ResearchGovernanceEngine
from .research_program_manager import LongTermResearchProgramManager
from .research_roadmap_generator import ResearchRoadmapGenerator
from .scientific_consensus_engine import ScientificConsensusEngine
from .uncertainty_manager import UncertaintyManagerEngine, UncertaintyPolicy


# Fallbacks used only when validation returns nothing usable. The score is 0.0
# rather than an optimistic default so a missing measurement can never be
# mistaken for a passing one; `evidence_missing` routes the decision to
# "More experiments" instead.
_NO_CONFIDENCE_SCORE = 0.0


def _extract_confidence(val_res: Any) -> tuple[Dict[str, Any], bool]:
    """Pull a usable confidence block out of a validation response.

    Returns ``(confidence, evidence_missing)``. ``evidence_missing`` is True
    when the response could not supply a numeric score and a well-formed
    two-element interval, meaning the campaign has no validation evidence.
    """
    block = val_res.get("confidence") if isinstance(val_res, dict) else None
    if not isinstance(block, dict):
        block = {}

    raw_score = block.get("confidence_score")
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
        return ({"confidence_score": _NO_CONFIDENCE_SCORE,
                 "uncertainty_interval": None}, True)
    confidence_score = float(raw_score)

    raw_interval = block.get("uncertainty_interval")
    if (isinstance(raw_interval, (list, tuple)) and len(raw_interval) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    for v in raw_interval)):
        interval: Optional[List[float]] = [float(raw_interval[0]),
                                           float(raw_interval[1])]
    else:
        interval = None

    missing = interval is None
    return ({"confidence_score": confidence_score,
             "uncertainty_interval": interval}, missing)


class AIScientistEngine:
    """Master Autonomous AI Scientist Engine."""

    def __init__(self, policy: UncertaintyPolicy | Dict[str, Any] | None = None) -> None:
        self.capabilities = CapabilityRegistry()
        self.evidence_graph = TraceableEvidenceGraph()
        self.workflow_dsl = DeclarativeWorkflowEngine()
        self.validation_engine = ScientificValidationEngine()
        self.agent_society = MultiAgentResearchSociety()
        self.debate_engine = ScientificDebateEngine()
        self.program_manager = LongTermResearchProgramManager()

        # AI 1 Scientific Reasoning Extensions
        self.critic_agent = ResearchCriticAgent()
        self.uncertainty_manager = UncertaintyManagerEngine(policy=policy)
        self.recommender = ExperimentRecommendationEngine()
        self.discovery_prioritizer = DiscoveryPrioritizerEngine()
        self.roadmap_generator = ResearchRoadmapGenerator()
        self.consensus_engine = ScientificConsensusEngine()
        self.literature_integrator = AutonomousLiteratureIntegrator()
        self.governance_engine = ResearchGovernanceEngine()

    def run_scientific_campaign(
        self,
        question: str = "Why does GPT-2 predict Paris for capital of France?",
        policy: UncertaintyPolicy | Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        # 1. Multi-agent collaboration & debate
        society_res = self.agent_society.run_society_collaboration(goal=question)
        debate_res = self.debate_engine.debate_hypotheses(
            hypothesis_a="L8_N402 mediates geographic capital retrieval",
            hypothesis_b="Layer 0 embeddings direct country logits",
        )

        # 2. Research Critic & Governance
        critique = self.critic_agent.critique_hypothesis(
            hypothesis=debate_res["consensus_winner"],
            evidence=[{"id": "ev_1", "score": 0.95}, {"id": "ev_2", "score": 0.92}],
        )
        gov = self.governance_engine.validate_and_approve(experiment_id="exp_s5_gov_1")

        # 3. Experiment recommendation & Literature integration
        rec = self.recommender.recommend_next_experiment(research_goal=question)
        lit = self.literature_integrator.integrate_literature(discovery_title="IOI Capital Retrieval Circuit")

        # 4. Workflow DSL execution
        dsl_res = self.workflow_dsl.execute_workflow()

        # 5. Scientific Validation Layer
        val_res = self.validation_engine.validate_discovery(
            discovery_id="disc_s5_master",
            hypothesis_statement=debate_res["consensus_winner"],
        )

        # 6. Uncertainty Manager decision logic with Configurable Policy
        # Validation is an external call and may return nothing usable ({} ,
        # None, a bare string, a truncated interval). Blind nested indexing
        # turned every one of those into an AttributeError/TypeError/
        # IndexError that killed the campaign. Extract defensively instead,
        # and treat an unusable response as *absent evidence* rather than as
        # a score: defaulting a missing measurement optimistically would let
        # the campaign reach "Publish" with no validation behind it.
        confidence, evidence_missing = _extract_confidence(val_res)
        uncertainty_decision = self.uncertainty_manager.evaluate_uncertainty(
            confidence_score=confidence["confidence_score"],
            uncertainty_interval=confidence["uncertainty_interval"],
            sample_size=0 if evidence_missing else 5,
            variance=0.02,
            custom_policy=policy,
            evidence_missing=evidence_missing,
        )

        # 7. Closed-Loop Planner Integration (Uncertainty ➔ Planner Loop)
        planner_feedback_loop = None
        if uncertainty_decision["action"] == "More experiments":
            planner_feedback_loop = {
                "triggered": True,
                "target_action": "experiment_recommender",
                "next_recommended_experiment": self.recommender.recommend_next_experiment(
                    research_goal=f"Follow-up for {question} due to {uncertainty_decision['decision']}"
                ),
            }
        elif uncertainty_decision["action"] == "Debate":
            planner_feedback_loop = {
                "triggered": True,
                "target_action": "scientific_debate",
                "followup_debate": self.debate_engine.debate_hypotheses(
                    hypothesis_a=debate_res["consensus_winner"],
                    hypothesis_b="Alternative variance explanation",
                ),
            }

        # 8. Roadmap & Consensus Synthesis
        roadmap = self.roadmap_generator.generate_roadmap(research_theme=question)
        consensus = self.consensus_engine.synthesize_consensus(experimental_outcomes=[val_res])

        return {
            "question": question,
            "society": society_res,
            "debate": debate_res,
            "critique": critique,
            "governance": gov,
            "recommendation": rec,
            "literature": lit,
            "workflow": dsl_res,
            "validation": val_res,
            "uncertainty_decision": uncertainty_decision,
            "planner_feedback_loop": planner_feedback_loop,
            "roadmap": roadmap,
            "consensus": consensus,
            "evidence_graph": self.evidence_graph.to_dict(),
            "capabilities": self.capabilities.list_capabilities(),
            "status": "ScientificCampaignCompleted",
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "ai_scientist_status": "active",
            "capabilities": self.capabilities.list_capabilities(),
            "evidence_nodes_count": self.evidence_graph.to_dict()["nodes_count"],
            "governance_audit_count": len(self.governance_engine.get_audit_trail()),
        }
