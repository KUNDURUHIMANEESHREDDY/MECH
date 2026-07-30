"""Autonomous Research Agent Brain."""

from __future__ import annotations

from typing import Any, Dict
from .engine import AutonomousResearchEngine


class AutonomousResearchAgent:
    """Event-driven autonomous agent brain orchestrating platform components."""

    def __init__(self, engine: AutonomousResearchEngine | None = None) -> None:
        self.engine = engine or AutonomousResearchEngine()

    def run_agent_workflow(self, goal: str) -> Dict[str, Any]:
        return self.engine.execute_goal(goal=goal)
