"""Epic 1 — Meta Research Engine.

Top-level coordinator studying research performance, triggering reflection, literature learning,
strategy optimization, policy repository versioning, campaign vector embeddings, skill library quality,
experience replay with 'Why' provenance, and curriculum planning in a closed feedback control loop.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional

from .campaign_embeddings import CampaignEmbeddingsEngine
from .experience_replay import ResearchExperienceReplay
from .literature_learning_pipeline import LiteratureLearningPipeline
from .multi_agent_evolution import MultiAgentEvolutionEngine
from .policy_repository import PolicyRepository
from .research_curriculum import AutonomousResearchCurriculum
from .research_strategy_optimizer import ResearchStrategyOptimizer
from .scientific_skill_library import ScientificSkillLibrary
from .self_reflection_engine import SelfReflectionEngine


class MetaResearchEngine:
    """Top-level controller for the Self-Improving AI Scientist closed feedback loop."""

    def __init__(self) -> None:
        self.reflection_engine = SelfReflectionEngine()
        self.literature_pipeline = LiteratureLearningPipeline()
        self.strategy_optimizer = ResearchStrategyOptimizer()
        self.agent_evolution = MultiAgentEvolutionEngine()
        self.skill_library = ScientificSkillLibrary()
        self.curriculum = AutonomousResearchCurriculum()
        self.experience_replay = ResearchExperienceReplay()
        self.policy_repository = PolicyRepository()
        self.campaign_embeddings = CampaignEmbeddingsEngine()

        # No seeded campaign. This used to ship a fabricated record for
        # "camp_s6_ioi" -- 127 hypotheses tested, 41 discoveries, 13
        # publications, average_confidence 0.91 -- describing a run that never
        # happened. Because every aggregate in analyze_meta_performance() sums
        # this list, that fiction was silently folded into the reported
        # success rate and publication count. An engine that has run nothing
        # must report nothing.
        self.campaign_history: List[Dict[str, Any]] = []
        # Also a literal, and also presented as a measured adaptation factor.
        # Exposed as unmeasured until a loop can actually derive one.
        self.strategy_adaptation_factor: Optional[float] = None

    def record_campaign_performance(
        self,
        campaign_id: str,
        topic: str,
        hypotheses_tested: int,
        discoveries_count: int,
        compute_used_vram_gb: float,
        failures_count: int = 0,
    ) -> Dict[str, Any]:
        record = {
            "campaign_id": campaign_id,
            "topic": topic,
            "hypotheses_tested": hypotheses_tested,
            "discoveries_count": discoveries_count,
            "compute_used_vram_gb": compute_used_vram_gb,
            "failures_count": failures_count,
            "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
        }
        self.campaign_history.append(record)

        # Index campaign vector embedding
        self.campaign_embeddings.index_campaign(
            campaign_id=campaign_id,
            topic=topic,
            summary_text=f"Campaign on {topic}: {discoveries_count} discoveries out of {hypotheses_tested} hypotheses.",
        )
        return record

    def run_closed_feedback_loop(self, campaign_id: str = "camp_s6_ioi") -> Dict[str, Any]:
        """Executes full closed feedback loop: Campaign -> Analytics -> Reflection -> Experience Replay -> Strategy Optimizer -> Policy Repo -> Skills -> Curriculum."""
        replay_res = self.experience_replay.replay_campaign(campaign_id)
        refl_res = self.reflection_engine.generate_reflection_report(campaign_id=campaign_id)
        strat_res = self.strategy_optimizer.recommend_strategy(domain="circuit_discovery")

        # Save immutable versioned policy to PolicyRepository.
        # success_rate=0.94 was a literal: nothing in this loop measured an
        # effectiveness rate, and the strategy it saves is a static template
        # whose own score `recommend_strategy` now reports as unmeasured.
        new_pol = self.policy_repository.save_policy(
            domain="circuit_discovery",
            workflow=strat_res["recommended_workflow"],
            success_rate=None,
            average_cost_usd=None,
            average_runtime_min=None,
        )

        return {
            "status": "ClosedFeedbackLoopCompleted",
            "campaign_id": campaign_id,
            "replay": replay_res,
            "reflection": refl_res,
            "recommended_strategy": strat_res,
            "saved_policy_version": new_pol["version"],
            "policy_id": new_pol["policy_id"],
            "policy_metrics_measured": False,
            "provenance": "unavailable",
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": (
                "This loop replays an illustrative trajectory and applies a "
                "static strategy template. No effectiveness, cost, or runtime "
                "was measured, so the saved policy carries no metrics."
            ),
            "loop_state": "Ready for Next Autonomous Campaign Execution",
        }

    def analyze_meta_performance(self) -> Dict[str, Any]:
        total_campaigns = len(self.campaign_history)
        total_discoveries = sum(c.get("discoveries_count", 0) for c in self.campaign_history)
        total_hypotheses = sum(c.get("hypotheses_tested", 0) for c in self.campaign_history) or 1
        success_rate = round(total_discoveries / total_hypotheses, 4)

        # `c.get("published_count", 1)` counted one publication for every
        # campaign that had never recorded one, so the summary reported
        # publications that did not happen. An absent key means zero.
        published_count = sum(int(c.get("published_count") or 0)
                              for c in self.campaign_history)

        return {
            "status": "Analyzed",
            "campaign_summary": {
                "total_campaigns": total_campaigns,
                "hypotheses_tested": total_hypotheses,
                "discoveries_count": total_discoveries,
                "rejected_count": sum(c.get("failures_count", 0) for c in self.campaign_history),
                "published_count": published_count,
                # These three were literals (8.4, 0.91, and two fixed workflow
                # strings) presented as aggregates over the campaign history.
                # Nothing here times a run or scores a confidence, so they are
                # reported as unmeasured instead of invented.
                "average_runtime_min": None,
                "average_confidence": None,
                "best_workflow": None,
                "worst_workflow": None,
                "metrics_measured": False,
                "reason": (
                    "Runtime, confidence, and workflow ranking are not "
                    "aggregated from any recorded measurement in this loop."
                ),
            },
            "overall_success_rate": success_rate,
            "overall_success_rate_measured": total_campaigns > 0,
            "strategy_adaptation_factor": self.strategy_adaptation_factor,
            "curriculum": self.curriculum.generate_curriculum(),
            "latest_policy": self.policy_repository.get_latest_policy("circuit_discovery"),
            "active_skills_count": len(self.skill_library.list_skills()),
            "evolving_agents_count": len(self.agent_evolution.list_evolving_agents()),
        }
