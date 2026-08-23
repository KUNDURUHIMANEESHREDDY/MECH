"""Cache Resolver for Execution DAGs.

Analyzes an ExecutionDAG against the ArtifactStore, identifying exactly which
nodes have valid cached artifacts and compiling the minimal uncomputed subgraph.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..artifacts.cas_store import ArtifactStore, get_artifact_store
from ..artifacts.models import Provenance
from .execution_dag import DAGNode, ExecutionDAG

logger = logging.getLogger("MECH.cache_resolver")


@dataclass
class ResolutionPlan:
    """Result of analyzing an ExecutionDAG against the artifact cache."""
    total_nodes_count: int
    cached_nodes_count: int
    uncomputed_nodes_count: int
    cached_node_ids: List[str]
    uncomputed_nodes: List[DAGNode]
    node_cas_keys: Dict[str, str]  # node_id -> computed CAS key
    reusable_artifacts: Dict[str, str]  # node_id -> artifact_id

    def is_fully_cached(self) -> bool:
        return self.uncomputed_nodes_count == 0

    def summary(self) -> Dict[str, Any]:
        return {
            "total_nodes": self.total_nodes_count,
            "cached_nodes": self.cached_nodes_count,
            "uncomputed_nodes": self.uncomputed_nodes_count,
            "cache_hit_rate": round(
                (self.cached_nodes_count / max(1, self.total_nodes_count)) * 100.0, 2
            ),
            "reusable_nodes": self.cached_node_ids,
            "nodes_to_execute": [n.node_id for n in self.uncomputed_nodes],
        }


class CacheResolver:
    """Resolves DAG execution requirements against the Content-Addressed Storage."""

    def __init__(self, store: Optional[ArtifactStore] = None) -> None:
        self.store = store or get_artifact_store()

    def resolve(self, dag: ExecutionDAG, provenance: Provenance) -> ResolutionPlan:
        """Determines which nodes in the DAG require execution.
        
        Invariant:
        - If a node's parents change, its CAS key changes deterministically.
        - If its CAS key is not found in the store, it (and all downstream nodes)
          must be executed.
        - If its CAS key exists in the store, it can be loaded directly from L1/L2.
        """
        sorted_nodes = dag.topological_sort()
        prov_digest = provenance.compute_digest()

        node_cas_keys: Dict[str, str] = {}
        cached_node_ids: List[str] = []
        uncomputed_nodes: List[DAGNode] = []
        reusable_artifacts: Dict[str, str] = {}

        for node in sorted_nodes:
            # 1. Gather parent CAS keys
            parent_cas_keys = []
            for p_id in node.parent_ids:
                if p_id in node_cas_keys:
                    parent_cas_keys.append(node_cas_keys[p_id])
                else:
                    # Parent was not mapped; generate fallback
                    parent_cas_keys.append(f"unmapped_{p_id}")

            # 2. Compute deterministic CAS key for this node
            node_key = node.compute_cas_key(
                parent_cas_keys=parent_cas_keys,
                provenance_digest=prov_digest,
            )
            node_cas_keys[node.node_id] = node_key

            # 3. Check ArtifactStore for this key
            if self.store.exists(node_key):
                cached_node_ids.append(node.node_id)
                reusable_artifacts[node.node_id] = node_key
            else:
                uncomputed_nodes.append(node)

        return ResolutionPlan(
            total_nodes_count=len(sorted_nodes),
            cached_nodes_count=len(cached_node_ids),
            uncomputed_nodes_count=len(uncomputed_nodes),
            cached_node_ids=cached_node_ids,
            uncomputed_nodes=uncomputed_nodes,
            node_cas_keys=node_cas_keys,
            reusable_artifacts=reusable_artifacts,
        )
