"""Typed Research Graph Engine.

Models typed research entities (Question, Hypothesis, Experiment, Evidence, Circuit, Feature, Conclusion, Publication)
and typed relationships (supports, contradicts, discovered_by, derived_from, validates, generated).
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List


class TypedResearchGraph:
    """DAG mapping research entities and relationship edges."""

    def __init__(self, graph_id: str = "rg_default") -> None:
        self.graph_id = graph_id
        self.nodes: List[Dict[str, Any]] = [
            {"id": "q_1", "type": "Question", "label": "Why does GPT-2 predict Paris for capital of France?"},
            {"id": "h_1", "type": "Hypothesis", "label": "Layer 8 MLP Neuron #402 mediates geographic capital retrieval"},
            {"id": "e_1", "type": "Experiment", "label": "Activation Patching L8_N402 over clean vs corrupted prompts"},
            {"id": "ev_1", "type": "Evidence", "label": "Logit delta -4.2 on ' Paris' upon zeroing L8_N402"},
            {"id": "c_1", "type": "Circuit", "label": "IOI Geographic Retrieval Circuit"},
            {"id": "conc_1", "type": "Conclusion", "label": "L8_N402 is an essential component of geographic capital prediction"},
        ]
        self.edges: List[Dict[str, Any]] = [
            {"source": "q_1", "target": "h_1", "type": "generated"},
            {"source": "h_1", "target": "e_1", "type": "derived_from"},
            {"source": "e_1", "target": "ev_1", "type": "discovered_by"},
            {"source": "ev_1", "target": "c_1", "type": "supports"},
            {"source": "c_1", "target": "conc_1", "type": "validates"},
        ]

    def add_node(self, node_id: str, node_type: str, label: str) -> Dict[str, Any]:
        node = {"id": node_id, "type": node_type, "label": label}
        self.nodes.append(node)
        return node

    def add_edge(self, source: str, target: str, edge_type: str) -> Dict[str, Any]:
        edge = {"source": source, "target": target, "type": edge_type}
        self.edges.append(edge)
        return edge

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "nodes_count": len(self.nodes),
            "edges_count": len(self.edges),
            "nodes": self.nodes,
            "edges": self.edges,
            "updated_at": _dt.datetime.utcnow().isoformat() + "Z",
        }
