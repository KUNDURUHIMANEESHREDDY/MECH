"""Provenance Graph Traversal & Lineage Engine.

Constructs end-to-end scientific provenance chains:
Paper ➔ Mechanism Claim ➔ Evidence ➔ Circuit ➔ Attention Head ➔ SAE Feature
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .ontology import NodeType, EdgeType
from .graph_store import GraphStore, KGNode, KGEdge


@dataclass
class ProvenanceChain:
    """End-to-end scientific provenance lineage chain."""
    root_node_id: str
    target_node_id: str
    path_nodes: List[Dict[str, Any]]
    path_edges: List[Dict[str, Any]]
    provenance_length: int
    confidence_score: float


class ProvenanceGraph:
    """Provenance Graph Lineage Engine."""

    def __init__(self, store: Optional[GraphStore] = None) -> None:
        self.store = store or GraphStore()

    def Trace_provenance_chain(self, start_node_id: str, max_depth: int = 4) -> List[ProvenanceChain]:
        """Traces all provenance lineage chains starting from start_node_id."""
        if start_node_id not in self.store.nodes:
            return []

        chains: List[ProvenanceChain] = []
        visited: Set[str] = set()

        def dfs(curr_id: str, current_nodes: List[KGNode], current_edges: List[KGEdge], depth: int):
            if depth >= max_depth:
                return

            visited.add(curr_id)
            out_edges = self.store.get_outgoing_edges(curr_id)

            if not out_edges and len(current_nodes) > 1:
                # Leaf node reached: record provenance chain
                conf = round(sum(e.weight for e in current_edges) / max(1, len(current_edges)), 3)
                chains.append(ProvenanceChain(
                    root_node_id=start_node_id,
                    target_node_id=curr_id,
                    path_nodes=[n.to_dict() for n in current_nodes],
                    path_edges=[e.to_dict() for e in current_edges],
                    provenance_length=len(current_nodes),
                    confidence_score=conf
                ))
                return

            for edge in out_edges:
                target_node = self.store.nodes.get(edge.target_id)
                if target_node and edge.target_id not in visited:
                    dfs(edge.target_id, current_nodes + [target_node], current_edges + [edge], depth + 1)

        root = self.store.nodes[start_node_id]
        dfs(start_node_id, [root], [], 0)
        return chains
