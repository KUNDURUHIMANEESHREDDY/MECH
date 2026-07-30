"""Epic 5 — Multi-Agent Evolution Engine.

Tracks agent performance (success rate, disagreement, contribution) and dynamically evolves agent behaviors over time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class EvolvingAgentState:
    """Dataclass storing evolutionary metrics and behavior parameters for a research agent."""

    agent_id: str
    role: str
    version: str = "v1.0"
    success_rate: float = 0.85
    disagreement_rate: float = 0.12
    contribution_score: float = 0.90
    generations_evolved: int = 0
    prompt_adaptations: List[str] | None = None

    def __post_init__(self) -> None:
        if self.prompt_adaptations is None:
            self.prompt_adaptations = ["Default Initial Strategy"]


class MultiAgentEvolutionEngine:
    """Manages the independent evolution and parameter adaptation of specialized multi-agent society members."""

    def __init__(self) -> None:
        self.agents: Dict[str, EvolvingAgentState] = {
            "research_critic": EvolvingAgentState(agent_id="research_critic", role="Research Critic", success_rate=0.88, disagreement_rate=0.15, contribution_score=0.92),
            "planner": EvolvingAgentState(agent_id="planner", role="Execution Planner", success_rate=0.91, disagreement_rate=0.08, contribution_score=0.95),
            "reviewer": EvolvingAgentState(agent_id="reviewer", role="Scientific Reviewer", success_rate=0.86, disagreement_rate=0.18, contribution_score=0.89),
        }

    def record_agent_interaction(
        self,
        agent_id: str,
        task_success: bool = True,
        disagreed_with_peer: bool = False,
        contribution_delta: float = 0.05,
    ) -> Dict[str, Any]:
        agent = self.agents.get(agent_id)
        if not agent:
            agent = EvolvingAgentState(agent_id=agent_id, role=agent_id.capitalize())
            self.agents[agent_id] = agent

        # Exponential moving update
        alpha = 0.2
        succ_val = 1.0 if task_success else 0.0
        dis_val = 1.0 if disagreed_with_peer else 0.0

        agent.success_rate = round((1 - alpha) * agent.success_rate + alpha * succ_val, 4)
        agent.disagreement_rate = round((1 - alpha) * agent.disagreement_rate + alpha * dis_val, 4)
        agent.contribution_score = round(min(1.0, agent.contribution_score + contribution_delta), 4)

        # Trigger evolutionary adaptation if enough interactions take place
        if agent.success_rate > 0.80 and agent.disagreement_rate < 0.25:
            agent.generations_evolved += 1
            agent.version = f"v1.{agent.generations_evolved}"
            adapt = f"Gen {agent.generations_evolved}: Prioritize high-confidence causal intervention checks."
            if adapt not in (agent.prompt_adaptations or []):
                agent.prompt_adaptations = (agent.prompt_adaptations or []) + [adapt]

        return asdict(agent)

    def list_evolving_agents(self) -> List[Dict[str, Any]]:
        return [asdict(a) for a in self.agents.values()]
