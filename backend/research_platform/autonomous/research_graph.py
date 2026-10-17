"""Typed Research Graph Engine.

Models typed research entities (Question, Hypothesis, Experiment, Evidence, Circuit, Feature, Conclusion, Publication)
and typed relationships (supports, contradicts, discovered_by, derived_from, validates, generated).
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List


class TypedResearchGraph:
    """DAG mapping research entities and relationship edges.

    Starts empty. This constructor used to seed six entities and five edges
    describing one specific IOI result -- including the evidence node
    "Logit delta -4.2 on ' Paris' upon zeroing L8_N402" and the conclusion
    "L8_N402 is an essential component of geographic capital prediction" --
    with nothing marking them as an illustration. `to_dict()["nodes_count"]`
    therefore counted six findings that nobody had made, and
    `AutonomousResearchEngine` appends its real nodes to the same graph, so
    the invented ones were indistinguishable from the recorded ones.
    """

    def __init__(self, graph_id: str = "rg_default") -> None:
        self.graph_id = graph_id
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []

    #: Illustrative shape only. Kept for UI fixtures and tests that need a
    #: populated graph, available through `demo()` and never by default.
    _EXHIBIT_NODES: List[Dict[str, Any]] = [
        {"id": "q_1", "type": "Question", "label": "Why does GPT-2 predict Paris for capital of France?"},
        {"id": "h_1", "type": "Hypothesis", "label": "Layer 8 MLP Neuron #402 mediates geographic capital retrieval"},
        {"id": "e_1", "type": "Experiment", "label": "Activation Patching L8_N402 over clean vs corrupted prompts"},
        {"id": "ev_1", "type": "Evidence", "label": "Logit delta -4.2 on ' Paris' upon zeroing L8_N402"},
        {"id": "c_1", "type": "Circuit", "label": "IOI Geographic Retrieval Circuit"},
        {"id": "conc_1", "type": "Conclusion", "label": "L8_N402 is an essential component of geographic capital prediction"},
    ]

    _EXHIBIT_EDGES: List[Dict[str, Any]] = [
        {"source": "q_1", "target": "h_1", "type": "generated"},
        {"source": "h_1", "target": "e_1", "type": "derived_from"},
        {"source": "e_1", "target": "ev_1", "type": "discovered_by"},
        {"source": "ev_1", "target": "c_1", "type": "supports"},
        {"source": "c_1", "target": "conc_1", "type": "validates"},
    ]

    @classmethod
    def demo(cls, graph_id: str = "rg_demo") -> "TypedResearchGraph":
        """The illustrative graph, opted into explicitly.

        Every seeded node carries `measured: False` and
        `provenance: "reference"`, so a consumer reading the graph cannot
        mistake an example for a result.
        """
        graph = cls(graph_id)
        graph.nodes.extend(
            {**node, "measured": False, "provenance": "reference"}
            for node in cls._EXHIBIT_NODES
        )
        graph.edges.extend(dict(edge) for edge in cls._EXHIBIT_EDGES)
        return graph

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
