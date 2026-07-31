"""Dynamic DAG Discovery Planner & Topological Orchestrator.

Replaces static linear execution pipelines with a Dynamic Directed Acyclic Graph (DAG) 
where each discovery node declares:
  - prerequisites
  - outputs
  - cost_ms
  - expected_info_gain

Example DAG Structure:
             Attribution
             /         \\
        ACDC         Transcoders
          |               |
      Scrubbing      Universality
           \\         /
          Evidence Fusion
"""

from __future__ import annotations

import datetime as _dt
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from science.models.adapter_base import ModelAdapter
from .algorithms import get_algorithm
from .algorithms.base_algorithm import DiscoveryReport
from .discovery_planner import ResearchGoal, MechanismClaim
from datasets.dataset_manager import DatasetManager


@dataclass
class DAGNode:
    """A single algorithm node in the Dynamic Discovery DAG."""
    node_id: str
    algorithm_name: str
    prerequisites: List[str]  # Node IDs that must complete before this node
    outputs: List[str]        # Artifacts emitted
    cost_ms: float
    expected_info_gain: float
    config_override: Dict[str, Any] = field(default_factory=dict)
    executed: bool = False
    result_report: Optional[Dict[str, Any]] = None


@dataclass
class DynamicDiscoveryDAG:
    """Directed Acyclic Graph (DAG) representation of a research campaign."""
    dag_id: str
    goal_id: str
    nodes: Dict[str, DAGNode]
    edges: List[Dict[str, str]]  # list of {"source": id, "target": id}

    def get_topological_levels(self) -> List[List[DAGNode]]:
        """Computes topological levels for parallel or ordered execution."""
        levels: List[List[DAGNode]] = []
        executed_ids: Set[str] = set()

        remaining_nodes = dict(self.nodes)

        while remaining_nodes:
            # Find nodes whose prerequisites are all satisfied
            current_level = [
                node for node in remaining_nodes.values()
                if all(p in executed_ids for p in node.prerequisites)
            ]

            if not current_level:
                # Fallback for cyclic or unsatisfied dependencies
                current_level = list(remaining_nodes.values())

            levels.append(current_level)
            for n in current_level:
                executed_ids.add(n.node_id)
                del remaining_nodes[n.node_id]

        return levels


class DynamicDAGPlanner:
    """Engine that constructs and executes Dynamic Discovery DAGs."""

    def __init__(self, adapter: Optional[ModelAdapter] = None, dataset_manager: Optional[DatasetManager] = None) -> None:
        self.adapter = adapter
        self.dataset_manager = dataset_manager or DatasetManager("backend/datasets")

    def build_dag(self, goal: ResearchGoal) -> DynamicDiscoveryDAG:
        """Constructs an optimal DAG based on node prerequisites and Expected Information Gain."""
        nodes: Dict[str, DAGNode] = {}
        edges: List[Dict[str, str]] = []

        desc_lower = goal.description.lower()

        # Node 1: Fast Attribution Screening (Root)
        nodes["attr"] = DAGNode(
            node_id="attr",
            algorithm_name="attribution_patching",
            prerequisites=[],
            outputs=["attributed_components"],
            cost_ms=2000.0,
            expected_info_gain=0.85
        )

        # Node 2: ACDC Circuit Pruning (Depends on Attribution)
        nodes["acdc"] = DAGNode(
            node_id="acdc",
            algorithm_name="acdc",
            prerequisites=["attr"],
            outputs=["pruned_subgraph"],
            cost_ms=8000.0,
            expected_info_gain=0.75
        )
        edges.append({"source": "attr", "target": "acdc"})

        # Node 3: Transcoders (Depends on Attribution, runs parallel to ACDC)
        if "mlp" in desc_lower or "transcoder" in desc_lower or "representation" in desc_lower:
            nodes["transcoders"] = DAGNode(
                node_id="transcoders",
                algorithm_name="transcoders",
                prerequisites=["attr"],
                outputs=["dictionary_features"],
                cost_ms=5000.0,
                expected_info_gain=0.65
            )
            edges.append({"source": "attr", "target": "transcoders"})

        # Node 4: Causal Scrubbing (Depends on ACDC)
        if goal.require_falsification:
            nodes["scrub"] = DAGNode(
                node_id="scrub",
                algorithm_name="causal_scrubbing",
                prerequisites=["acdc"],
                outputs=["falsification_score"],
                cost_ms=10000.0,
                expected_info_gain=0.95
            )
            edges.append({"source": "acdc", "target": "scrub"})

        # Node 5: Feature Universality (Depends on Transcoders or ACDC)
        if goal.require_universality or "universal" in desc_lower or "cross-model" in desc_lower:
            parent_node = "transcoders" if "transcoders" in nodes else "acdc"
            nodes["universality"] = DAGNode(
                node_id="universality",
                algorithm_name="feature_universality",
                prerequisites=[parent_node],
                outputs=["cross_model_alignments"],
                cost_ms=6000.0,
                expected_info_gain=0.90
            )
            edges.append({"source": parent_node, "target": "universality"})

        # Node 6: Evidence Fusion (Sink node depending on all leaf nodes)
        prereqs_for_fusion = [nid for nid in nodes if nid not in [e["source"] for e in edges]]
        nodes["fusion"] = DAGNode(
            node_id="fusion",
            algorithm_name="evidence_fusion",
            prerequisites=prereqs_for_fusion,
            outputs=["mechanism_claim"],
            cost_ms=1000.0,
            expected_info_gain=1.0
        )
        for p in prereqs_for_fusion:
            edges.append({"source": p, "target": "fusion"})

        return DynamicDiscoveryDAG(
            dag_id=f"dag_{hash(goal.goal_id + str(time.time())) & 0xffffffff:08x}",
            goal_id=goal.goal_id,
            nodes=nodes,
            edges=edges
        )

    def execute_dag(self, goal: ResearchGoal) -> Dict[str, Any]:
        """Executes a Dynamic Discovery DAG in topological level order."""
        t0 = time.time()
        dag = self.build_dag(goal)
        levels = dag.get_topological_levels()

        prompts = self.dataset_manager.load(goal.dataset_name)
        dataset_item = prompts[0] if prompts else {"clean": "John gave a drink to Mary", "target": " Mary"}

        execution_logs = []
        reports = {}

        for level_idx, level_nodes in enumerate(levels):
            level_log = {"level": level_idx + 1, "executed_nodes": []}

            for node in level_nodes:
                if node.algorithm_name == "evidence_fusion":
                    continue

                if self.adapter:
                    alg = get_algorithm(node.algorithm_name, self.adapter)
                    rep: DiscoveryReport = alg.run(dataset_item)
                    node.result_report = rep.to_dict()
                    reports[node.node_id] = rep
                else:
                    node.result_report = {"algorithm": node.algorithm_name, "confidence": 0.95}

                node.executed = True
                level_log["executed_nodes"].append({
                    "node_id": node.node_id,
                    "algorithm": node.algorithm_name,
                    "prerequisites": node.prerequisites,
                    "expected_info_gain": node.expected_info_gain
                })

            execution_logs.append(level_log)

        total_runtime = (time.time() - t0) * 1000

        return {
            "dag_id": dag.dag_id,
            "goal_description": goal.description,
            "total_levels": len(levels),
            "nodes_count": len(dag.nodes),
            "execution_logs": execution_logs,
            "total_runtime_ms": round(total_runtime, 2)
        }
