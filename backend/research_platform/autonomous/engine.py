"""Autonomous Research Engine.

Central execution loop orchestrating Goal -> Plan -> Experiment -> Evidence -> Knowledge -> Memory -> Report.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List

from .autonomous_planner import AutonomousPlannerEngine
from .hypothesis_generator import HypothesisGeneratorEngine
from .knowledge_base import KnowledgeBaseEngine
from .research_dashboard import ResearchDashboardEngine
from .research_graph import TypedResearchGraph
from .research_memory import ResearchMemoryEngine


class AutonomousResearchEngine:
    """Central orchestrator driving the autonomous research loop."""

    def __init__(self) -> None:
        self.graph = TypedResearchGraph()
        self.knowledge_base = KnowledgeBaseEngine()
        self.memory = ResearchMemoryEngine()
        self.hypothesis_generator = HypothesisGeneratorEngine()
        self.planner = AutonomousPlannerEngine()
        self.dashboard = ResearchDashboardEngine()
        self.execution_history: List[Dict[str, Any]] = []

    def execute_goal(self, goal: str = "Investigate IOI Circuit in GPT-2") -> Dict[str, Any]:
        t0 = _dt.datetime.utcnow().isoformat() + "Z"

        # 1. Understand Goal & Generate Hypotheses
        hypotheses = self.hypothesis_generator.generate_hypotheses(context_prompt=goal)

        # 2. Create Dependency Plan
        plan = self.planner.create_plan(goal=goal)

        # 3. Record what was actually generated.
        #
        # This used to store a hardcoded fact -- `entity="GPT-2 L8_N402"`,
        # `prop="Circuit Mediation"`, `value="IOI Indirect Object Name Retrieval"`
        # at `confidence=0.95` -- on every call, for every goal. Nothing measured
        # it, and nothing about it varied with `goal` or with the hypotheses that
        # had just been generated. It was a scientific claim written into the
        # knowledge base by the orchestrator as a side effect of doing nothing
        # in particular.
        #
        # What is stored now is the one thing this method actually knows: how many
        # hypotheses were generated, and what the goal was.
        fact = self.knowledge_base.store_fact(
            entity=goal,
            prop="Hypotheses generated",
            value=len(hypotheses),
            confidence=None,
        )

        # 4. Record Experiential Memory
        #
        # The description asserted "Executed goal cleanly. Identified 2 candidate
        # hypotheses" -- a fixed string. "2" contradicted `len(hypotheses)`, which
        # the same method reports three lines later, and "cleanly" was asserted
        # unconditionally by a method with no error handling. `utility_score=0.92`
        # was likewise invented.
        #
        # The count is now the real one, and no success or quality claim is made.
        mem = self.memory.record_memory(
            category="autonomous_discovery",
            description=(
                f"Processed goal '{goal}': generated {len(hypotheses)} "
                f"hypothesis(es), created plan {plan['plan_id']}."
            ),
        )

        # 5. Update Research Graph
        self.graph.add_node(node_id=f"g_{len(self.execution_history)}", node_type="Question", label=goal)

        record = {
            "goal": goal,
            "hypotheses_count": len(hypotheses),
            "plan_id": plan["plan_id"],
            "fact_stored": fact["id"],
            "memory_recorded": mem["memory_id"],
            "status": "completed",
            "executed_at": t0,
        }
        self.execution_history.append(record)
        return record

    def get_status(self) -> Dict[str, Any]:
        return {
            "status": "ready",
            "graph": self.graph.to_dict(),
            "dashboard": self.dashboard.get_summary(),
            "execution_history_count": len(self.execution_history),
        }
