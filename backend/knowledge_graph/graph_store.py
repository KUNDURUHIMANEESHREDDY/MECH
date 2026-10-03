"""In-Memory Multi-Relational Knowledge Graph Store with JSON Persistence.

Stores entity nodes and directional edges, supporting multi-hop graph traversal,
type indexing, and persistent JSON storage.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .ontology import NodeType, EdgeType


@dataclass
class KGNode:
    """Node entity in the knowledge graph."""
    node_id: str
    node_type: str  # NodeType enum value
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "label": self.label,
            "properties": self.properties,
            "created_at": self.created_at,
        }


@dataclass
class KGEdge:
    """Directional edge relationship in the knowledge graph."""
    edge_id: str
    source_id: str
    target_id: str
    edge_type: str  # EdgeType enum value
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type,
            "weight": self.weight,
            "properties": self.properties,
            "created_at": self.created_at,
        }


class GraphStore:
    """Multi-relational graph store with persistent JSON storage."""

    def __init__(self, storage_path: str = "backend/research_datasets/knowledge_graph_index.json") -> None:
        self.storage_path = storage_path
        self.nodes: Dict[str, KGNode] = {}
        self.edges: Dict[str, KGEdge] = {}
        self.adjacency_out: Dict[str, Set[str]] = {}  # source_id ➔ set of edge_ids
        self.adjacency_in: Dict[str, Set[str]] = {}   # target_id ➔ set of edge_ids
        self.type_index: Dict[str, Set[str]] = {}     # node_type ➔ set of node_ids
        self._load_or_initialize()

    def add_node(self, node: KGNode) -> None:
        """Inserts or updates a node in the graph."""
        self.nodes[node.node_id] = node
        if node.node_type not in self.type_index:
            self.type_index[node.node_type] = set()
        self.type_index[node.node_type].add(node.node_id)
        self.save()

    def add_edge(self, edge: KGEdge) -> None:
        """Inserts an edge connecting source_id ➔ target_id."""
        self.edges[edge.edge_id] = edge
        
        if edge.source_id not in self.adjacency_out:
            self.adjacency_out[edge.source_id] = set()
        self.adjacency_out[edge.source_id].add(edge.edge_id)

        if edge.target_id not in self.adjacency_in:
            self.adjacency_in[edge.target_id] = set()
        self.adjacency_in[edge.target_id].add(edge.edge_id)

        self.save()

    def get_nodes_by_type(self, node_type: str) -> List[KGNode]:
        """Returns all nodes matching a specific NodeType."""
        node_ids = self.type_index.get(node_type, set())
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]

    def get_outgoing_edges(self, node_id: str) -> List[KGEdge]:
        """Returns all outgoing edges from node_id."""
        edge_ids = self.adjacency_out.get(node_id, set())
        return [self.edges[eid] for eid in edge_ids if eid in self.edges]

    def get_incoming_edges(self, node_id: str) -> List[KGEdge]:
        """Returns all incoming edges to node_id."""
        edge_ids = self.adjacency_in.get(node_id, set())
        return [self.edges[eid] for eid in edge_ids if eid in self.edges]

    def save(self) -> None:
        """Persists graph store state to disk."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        data = {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges.values()],
        }
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load_or_initialize(self) -> None:
        """Loads graph from JSON or initializes default baseline nodes."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for nd in data.get("nodes", []):
                    node = KGNode(**nd)
                    self.nodes[node.node_id] = node
                    if node.node_type not in self.type_index:
                        self.type_index[node.node_type] = set()
                    self.type_index[node.node_type].add(node.node_id)
                for ed in data.get("edges", []):
                    edge = KGEdge(**ed)
                    self.edges[edge.edge_id] = edge
                    if edge.source_id not in self.adjacency_out:
                        self.adjacency_out[edge.source_id] = set()
                    self.adjacency_out[edge.source_id].add(edge.edge_id)
                    if edge.target_id not in self.adjacency_in:
                        self.adjacency_in[edge.target_id] = set()
                    self.adjacency_in[edge.target_id].add(edge.edge_id)
                return
            except Exception:
                pass

        self._build_default_baseline()

    def _build_default_baseline(self) -> None:
        """Build a placeholder baseline graph so the UI has nodes to render.

        Every node here is a transcription, and none of it is a MECH measurement.
        The previous version asserted a chain of invented numbers as if it were
        the platform's own accumulated evidence:

            claim      confidence 0.962, status "Validated"
            circuit    logit_diff 3.55
            experiment algorithm "acdc", fidelity 0.972
            campaign   status "Completed"
            feature    sparsity 0.0012

        and then wired them together so that the fabricated experiment SUPPORTS
        the fabricated claim (edge e6). Read end to end, that subgraph asserted:
        an ACDC edge-pruning run at 0.972 fidelity validated the IOI Name Mover
        circuit at 0.962 confidence.

        None of it was run. The real IOI numbers are faithfulness 0.724 for the
        discovered circuit and 0.598 for the published circuit through this
        harness, recorded by the IOI pipeline -- neither of which is 0.972.

        Unlike the seeded graph in science/explorer/knowledge_graph.py, this one
        is live: `_build_default_baseline` runs from `__init__`, so the fabricated
        baseline was present in every freshly constructed graph.

        The citations and the paper's own head IDs stay, because those are the
        provenance. The invented measurements are gone, and the whole baseline is
        marked so nothing downstream can rank it above a real result.
        """
        seeded = {
            "provenance": "seeded",
            "validation_eligible": False,
            "publication_eligible": False,
            "measured": False,
            "source": "Wang et al. 2022 IOI circuit, transcribed as a placeholder",
        }

        # 1. Nodes
        p1 = KGNode("paper_wang2022", NodeType.PAPER, "Wang et al. (2022) IOI Paper",
                    {"title": "Interpretability in the Wild: a Circuit for Indirect Object Identification",
                     "year": 2022, **seeded})
        c1 = KGNode("claim_ioi_01", NodeType.MECHANISM_CLAIM, "IOI Name Mover Circuit (reported)",
                    {"status": "Reported", **seeded})
        cr1 = KGNode("circuit_ioi", NodeType.CIRCUIT, "Indirect Object Identification Subgraph (reported)",
                     {"num_heads": 3, **seeded})
        h1 = KGNode("head_l9h9", NodeType.ATTENTION_HEAD, "GPT-2 L9H9 Name Mover Head",
                    {"layer": 9, "head": 9, "model": "GPT2-S", **seeded})
        h2 = KGNode("head_l10h0", NodeType.ATTENTION_HEAD, "GPT-2 L10H0 Name Mover Head",
                    {"layer": 10, "head": 0, "model": "GPT2-S", **seeded})
        f1 = KGNode("feature_sae_4096", NodeType.SAE_FEATURE, "SAE Feature #4096 (Proper Nouns)",
                    {"model": "GPT2-S", **seeded})
        cmp1 = KGNode("camp_ioi_01", NodeType.CAMPAIGN, "IOI Discovery Campaign",
                      {"status": "Placeholder", **seeded})
        exp1 = KGNode("exp_acdc_01", NodeType.EXPERIMENT, "ACDC Edge Pruning Run (placeholder)",
                      {"algorithm": "acdc", **seeded})

        for n in [p1, c1, cr1, h1, h2, f1, cmp1, exp1]:
            self.add_node(n)

        # 2. Edges
        e1 = KGEdge("e1", p1.node_id, c1.node_id, EdgeType.CITES, properties=dict(seeded))
        e2 = KGEdge("e2", c1.node_id, cr1.node_id, EdgeType.REFERENCES, properties=dict(seeded))
        e3 = KGEdge("e3", cr1.node_id, h1.node_id, EdgeType.CONTAINS, properties=dict(seeded))
        e4 = KGEdge("e4", cr1.node_id, h2.node_id, EdgeType.CONTAINS, properties=dict(seeded))
        e5 = KGEdge("e5", h1.node_id, f1.node_id, EdgeType.ACTIVATES, properties=dict(seeded))
        # Placeholder nodes, and the edge carries the placeholder marking, so a
        # SUPPORTS relationship here cannot be read as measured support.
        e6 = KGEdge("e6", exp1.node_id, c1.node_id, EdgeType.SUPPORTS, properties=dict(seeded))
        e7 = KGEdge("e7", cmp1.node_id, exp1.node_id, EdgeType.CONTAINS, properties=dict(seeded))

        for e in [e1, e2, e3, e4, e5, e6, e7]:
            self.add_edge(e)
