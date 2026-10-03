"""Historical Research Society facade.

The original implementation was a synthetic demonstration scaffold. Every agent
returned a fixed, confident-sounding value regardless of its input:

    Planner.create_plan    -> ["Step 1 to achieve {goal}", "Step 2 ..."]
    GoalOptimizer.optimize -> "Optimized: {goal}"
    ResearchAgent          -> {"status": "executed", "finding": "Discovered new feature"}
    Critic.evaluate        -> 0.95
    DebateEngine.debate    -> "Consensus reached after rigorous debate."
    ConsensusEngine        -> {"synthesis": "Unified theory established"}
    Reviewer.review        -> {"approved": True, "feedback": "Solid methodology"}
    RoadmapGenerator       -> ["Next goal: Induction Heads", "Next goal: SAE Scaling"]

None of these computed anything. `Critic.evaluate` returning 0.95 is the
clearest instance: a confidence score with no rubric behind it, reachable by
any caller and indistinguishable from a real evaluation.

Rather than delete the classes -- they are part of the public surface and
something may still import them -- each one now refuses. An agent that cannot
do its job reports that it cannot, which is the only safe behaviour for a
component whose output is a verdict.

`ResearchSociety.run_society_collaboration` was already fail-closed and is
unchanged. Active Society work uses
:class:`backend.agents.society.ResearchSocietyV2`.
"""

from typing import Any, Dict, List, Optional

# One message per agent, naming what the method would need in order to answer.
_UNIMPLEMENTED = {
    "Planner.create_plan": (
        "plan decomposition requires a planner that can reason about the "
        "goal; use backend.agents.planner.Planner, which returns a labelled "
        "lifecycle skeleton"
    ),
    "GoalOptimizer.optimize": (
        "optimising a goal requires a search over objectives, which is not "
        "implemented"
    ),
    "ResearchAgent.execute_step": (
        "executing a step requires an executor bound to a live model; use "
        "backend.agents.society.ResearchSocietyV2"
    ),
    "Critic.evaluate": (
        "scoring a finding requires a rubric and the finding's measurements; "
        "no rubric exists here, so no confidence can be assigned"
    ),
    "DebateEngine.debate": (
        "a debate outcome requires agents that can actually argue; no such "
        "agents are wired up, so no consensus can be reported"
    ),
    "ConsensusEngine.synthesize": (
        "synthesis requires evaluated inputs; none were produced"
    ),
    "Reviewer.review": (
        "review requires reading the report against criteria; no criteria are "
        "defined, so no approval can be given"
    ),
    "RoadmapGenerator.generate": (
        "a roadmap requires a record of completed goals; none was supplied"
    ),
}


def _refuse(method: str, **context: Any) -> Dict[str, Any]:
    """The uniform refusal shape every agent here returns."""
    return {
        "status": "unavailable",
        "implemented": False,
        "provenance": "unavailable",
        "validation_eligible": False,
        "publication_eligible": False,
        "method": method,
        "reason": _UNIMPLEMENTED[method],
        "context": context,
    }


class _UnimplementedAgent:
    """Base for the historical stubs: fails closed with a named reason."""

    def _refuse(self, method: str, **context: Any) -> Dict[str, Any]:
        return _refuse(method, **context)


class Planner(_UnimplementedAgent):
    """Was: returned ["Step 1 to achieve {goal}", "Step 2 to achieve {goal}"]."""

    def create_plan(self, goal: str) -> Dict[str, Any]:
        return _refuse("Planner.create_plan", goal=goal)


class GoalOptimizer(_UnimplementedAgent):
    """Was: returned f"Optimized: {goal}"."""

    def optimize(self, goal: str) -> Dict[str, Any]:
        return _refuse("GoalOptimizer.optimize", goal=goal)


class ResearchAgent(_UnimplementedAgent):
    """Was: returned {"status": "executed", "finding": "Discovered new feature"}."""

    def execute_step(self, step: str) -> Dict[str, Any]:
        return _refuse("ResearchAgent.execute_step", step=step)


class Critic(_UnimplementedAgent):
    """Was: returned 0.95 for any finding whatsoever.

    A confidence without a rubric is a decoration, so this returns None rather
    than a number. `Optional[float]` is the honest type; callers that treat 0.0
    as "bad finding" and None as "no verdict" both get what they need.
    """

    def evaluate(self, finding: Dict[str, Any]) -> Optional[float]:
        _refuse("Critic.evaluate", finding_keys=sorted(finding or {}))
        return None


class DebateEngine(_UnimplementedAgent):
    """Was: returned "Consensus reached after rigorous debate." for any topic."""

    def debate(self, topic: str) -> Dict[str, Any]:
        return _refuse("DebateEngine.debate", topic=topic)


class ConsensusEngine(_UnimplementedAgent):
    """Was: returned {"synthesis": "Unified theory established"}."""

    def synthesize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        return _refuse("ConsensusEngine.synthesize",
                       input_count=len(results or []))


class Reviewer(_UnimplementedAgent):
    """Was: returned {"approved": True, "feedback": "Solid methodology"}."""

    def review(self, report: str) -> Dict[str, Any]:
        return _refuse("Reviewer.review", report_chars=len(report or ""))


class RoadmapGenerator(_UnimplementedAgent):
    """Was: returned ["Next goal: Induction Heads", "Next goal: SAE Scaling"]."""

    def generate(self, past_goals: List[str]) -> Dict[str, Any]:
        return _refuse("RoadmapGenerator.generate",
                       past_goal_count=len(past_goals or []))


class ResearchSociety:
    """The central orchestrator managing all specialized AI sub-agents."""

    def __init__(self) -> None:
        self.planner = Planner()
        self.goal_optimizer = GoalOptimizer()
        self.agent = ResearchAgent()
        self.critic = Critic()
        self.debate_engine = DebateEngine()
        self.consensus_engine = ConsensusEngine()
        self.reviewer = Reviewer()
        self.roadmap_generator = RoadmapGenerator()

    def run_society_collaboration(self, goal: str) -> Dict[str, Any]:
        """Disable the historical stub instead of returning synthetic claims."""
        return {
            "status": "blocked",
            "provenance": "unavailable",
            "validation_eligible": False,
            "publication_eligible": False,
            "goal": goal,
            "reason": (
                "The historical Society stub is disabled; use the guarded "
                "ResearchSocietyV2 workflow with live evidence."
            ),
        }
