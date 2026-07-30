"""Planner Policy Optimizer.

Updates Dynamic Discovery Planner execution policies based on meta-learning recommendations.

Workflow:
Research Goal ➔ Meta-Learning Engine ➔ Policy Optimization ➔ Optimized Dynamic DAG
"""

from __future__ import annotations

import datetime as _dt
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..interpretability.discovery.dag_discovery_planner import DynamicDAGPlanner, DynamicDiscoveryDAG, DAGNode
from ..interpretability.discovery.discovery_planner import ResearchGoal
from .meta_learning_engine import MetaLearningEngine, OptimizedPlannerPolicy


class PlannerOptimizer:
    """Optimizes Dynamic Discovery Planner using meta-learned execution policies."""

    def __init__(
        self,
        planner: Optional[DynamicDAGPlanner] = None,
        meta_engine: Optional[MetaLearningEngine] = None
    ) -> None:
        self.planner = planner or DynamicDAGPlanner()
        self.meta_engine = meta_engine or MetaLearningEngine()

    def build_meta_optimized_dag(self, goal: ResearchGoal) -> DynamicDiscoveryDAG:
        """Constructs a Dynamic Discovery DAG optimized by historical meta-learning policy."""
        policy: OptimizedPlannerPolicy = self.meta_engine.learn_policy(task_category=goal.dataset_name)

        # Build baseline DAG from planner
        dag = self.planner.build_dag(goal)

        # Apply meta-learning priority weights & anti-pattern pruning
        weights = policy.algorithm_priority_weights

        for node_id, node in dag.nodes.items():
            alg_weight = weights.get(node.algorithm_name, 1.0)
            node.expected_info_gain = round(node.expected_info_gain * alg_weight, 4)

            # Prevent failure anti-pattern: Ensure Universality ALWAYS depends on ACDC or Transcoders
            if node.algorithm_name == "feature_universality":
                if not node.prerequisites:
                    node.prerequisites = ["acdc"]

        return dag
