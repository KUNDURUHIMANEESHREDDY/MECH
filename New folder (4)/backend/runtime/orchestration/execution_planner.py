"""Execution Planner Engine & Execution Plan Dataclass.

Separates execution planning from execution orchestration.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict

from .data_locality_manager import DataLocalityManager
from .learned_runtime_optimizer import LearnedRuntimeOptimizer


@dataclass
class ExecutionPlan:
    """Explicit container representing a compiled execution plan."""

    plan_id: str
    experiment_id: str
    target_backend: str
    locality_tier: str
    predicted_resources: Dict[str, Any]
    checkpoint_frequency: int
    strategy: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExecutionPlannerEngine:
    """Compiles research requests into explicit ExecutionPlans using resource optimization and data locality rules."""

    def __init__(self, optimizer: LearnedRuntimeOptimizer | None = None, locality_manager: DataLocalityManager | None = None) -> None:
        self.optimizer = optimizer or LearnedRuntimeOptimizer()
        self.locality_manager = locality_manager or DataLocalityManager()

    def create_execution_plan(
        self,
        experiment_id: str,
        model_name: str = "GPT-2 Small",
        num_prompts: int = 1000,
        strategy: str = "Balanced",
    ) -> ExecutionPlan:
        pred = self.optimizer.predict_execution(model_name=model_name, num_prompts=num_prompts)
        locality = self.locality_manager.allocate_tensor_locality(tensor_id=f"tensor_{experiment_id}")

        return ExecutionPlan(
            plan_id=f"plan_{uuid.uuid4().hex[:8]}",
            experiment_id=experiment_id,
            target_backend=pred["recommended_backend"],
            locality_tier=locality.get("allocated_tier", "GPU_VRAM"),
            predicted_resources=pred,
            checkpoint_frequency=100,
            strategy=strategy,
        )
