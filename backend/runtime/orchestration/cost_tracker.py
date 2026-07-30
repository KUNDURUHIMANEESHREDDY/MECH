"""Experiment Cost Tracking Engine."""

from __future__ import annotations

from typing import Any, Dict


class ExperimentCostTracker:
    """Tracks real-time GPU/CPU dollar costs and enforces organization compute budgets."""

    def __init__(self) -> None:
        self.accumulated_cost = 0.0

    def track_cost(self, gpu_hours: float, rate_per_hour: float = 2.45) -> Dict[str, Any]:
        cost = round(gpu_hours * rate_per_hour, 4)
        self.accumulated_cost += cost
        return {
            "session_cost_usd": cost,
            "accumulated_cost_usd": round(self.accumulated_cost, 4),
            "budget_limit_usd": 1000.0,
            "within_budget": self.accumulated_cost <= 1000.0,
        }
