"""Mechanistic Knowledge Graph Backend.

The central data model connecting all research artifacts. Provides a directed graph
system where nodes support metadata, provenance, confidence scores, and bidirectional links.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class GraphNode:
    """A node in the Mechanistic Knowledge Graph."""
    id: str
    type: str                     # Paper, Experiment, Circuit, AttentionHead, Neuron, SAEFeature, Evidence, Concept, ConceptFamily, KnowledgeDomain, Dataset, DatasetVersion, ResearchPackage
    label: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance_manifest_id: Optional[str] = None
    # Optional, and defaulting to None rather than 1.0. This is the evidence
    # store, and 1.0 is a claim of certainty: every node added without an
    # explicit confidence was asserting total, measured confidence in a
    # mechanistic claim. Nothing measured it -- `add_node` accepts the field but
    # never computes it, so the default was the only value most nodes ever had.
    # None means unmeasured, which is the honest state and is distinguishable
    # from a measured 1.0.
    confidence_score: Optional[float] = None  # 0.0 to 1.0 when measured
    created_at: str = field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat())


@dataclass
class GraphEdge:
    """A directed edge connecting two nodes."""
    source_id: str
    target_id: str
    relationship: str             # MEMBER_OF, SUPPORTS, ACTIVATES, CONTRADICTS, generalizes, specializes, evolves_into, aligns_with, represented_by
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class MechanisticKnowledgeGraph:
    """In-memory (for now) Knowledge Graph engine."""

    _instance = None

    def __new__(cls) -> MechanisticKnowledgeGraph:
        if cls._instance is None:
            cls._instance = super(MechanisticKnowledgeGraph, cls).__new__(cls)
            cls._instance.nodes = {}
            cls._instance.edges = []
        return cls._instance

    def add_node(
        self,
        type: str,
        label: str,
        metadata: Optional[Dict[str, Any]] = None,
        provenance_manifest_id: Optional[str] = None,
        confidence_score: Optional[float] = None,
        node_id: Optional[str] = None,
    ) -> GraphNode:
        if node_id is None:
            node_id = f"node_{uuid.uuid4().hex[:8]}"

        # Ensure provenance is tracked in metadata if provided
        final_metadata = metadata or {}
        if provenance_manifest_id:
            final_metadata["provenance_manifest_id"] = provenance_manifest_id

        node = GraphNode(
            id=node_id,
            type=type,
            label=label,
            metadata=final_metadata,
            provenance_manifest_id=provenance_manifest_id,
            confidence_score=confidence_score,
        )
        self.nodes[node_id] = node
        return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship: str,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GraphEdge:
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError("Source or target node does not exist.")
        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            relationship=relationship,
            weight=weight,
            metadata=metadata or {},
        )
        self.edges.append(edge)
        return edge

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self.nodes.get(node_id)

    def get_subgraph(self, root_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """Traverse the graph from a root node up to max_depth, returning a localized graph."""
        if root_id not in self.nodes:
            return {"nodes": [], "edges": []}

        visited_nodes: Set[str] = set()
        edges_included: List[GraphEdge] = []
        
        queue = [(root_id, 0)]
        while queue:
            current_id, depth = queue.pop(0)
            if current_id in visited_nodes:
                continue
            
            visited_nodes.add(current_id)

            if depth < max_depth:
                # Find outgoing edges
                outgoing = [e for e in self.edges if e.source_id == current_id]
                for edge in outgoing:
                    if edge not in edges_included:
                        edges_included.append(edge)
                    if edge.target_id not in visited_nodes:
                        queue.append((edge.target_id, depth + 1))
                
                # Find incoming edges (backlinks)
                incoming = [e for e in self.edges if e.target_id == current_id]
                for edge in incoming:
                    if edge not in edges_included:
                        edges_included.append(edge)
                    if edge.source_id not in visited_nodes:
                        queue.append((edge.source_id, depth + 1))

        return {
            "nodes": [self.nodes[nid].__dict__ for nid in visited_nodes],
            "edges": [e.__dict__ for e in edges_included]
        }

    def clear(self) -> None:
        self.nodes.clear()
        self.edges.clear()

    def _seed_mock_data(self) -> None:
        """Seed a placeholder knowledge graph so the UI has something to render.

        Everything here is a *transcription of Wang et al. 2022*, not a result
        MECH produced: the paper node is theirs, the circuit is theirs, and the
        three heads (L9H9, L10H0, L7H3) are from their published list.

        It is seeded because an empty graph gives the explorer nothing to draw.
        Every node and edge therefore carries the standard fixture markers
        (`provenance: seeded`, and ineligibility for validation and publication)
        so that nothing downstream can mistake this for a measured circuit.

        The `SUPPORTS` edge previously carried `confidence_score: 0.95`. Nothing
        measured that. It was a typed-in number on a graph whose whole purpose is
        to hold evidence, which is worse than having no score: a consumer
        filtering on confidence would have ranked this placeholder above real
        measurements. The number is gone rather than labelled, because there is no
        measurement it could be a label *for*. Recorded for the record: 0.95 is
        the same value the validation layer was explicitly prevented from
        defaulting to earlier, so the two ends of the codebase were making
        opposite claims about the same magic number.
        """
        if self.nodes:
            return

        # Marks every seeded node and edge. See the docstring: this content is a
        # paper transcription, and must never read as MECH evidence.
        seeded = {
            "provenance": "seeded",
            "validation_eligible": False,
            "publication_eligible": False,
            "source": "Wang et al. 2022, transcribed for UI development",
        }

        paper = self.add_node("Paper", "IOI Paper (Wang et al.)", node_id="paper_ioi",
                              metadata=dict(seeded))
        circuit = self.add_node("Circuit", "IOI Circuit (from published paper)",
                                node_id="circuit_ioi", metadata=dict(seeded))

        # Heads
        nmh1 = self.add_node("AttentionHead", "L9H9 (Name Mover)", node_id="head_L9H9",
                             metadata={**seeded, "layer": 9, "head": 9})
        nmh2 = self.add_node("AttentionHead", "L10H0 (Name Mover)", node_id="head_L10H0",
                             metadata={**seeded, "layer": 10, "head": 0})
        s2i = self.add_node("AttentionHead", "L7H3 (S2 Inhibition)", node_id="head_L7H3",
                            metadata={**seeded, "layer": 7, "head": 3})

        # Evidence
        ev1 = self.add_node("Evidence", "Path Patching -> NMH", node_id="ev_path_nmh",
                            metadata=dict(seeded))

        # Edges
        self.add_edge(paper.id, circuit.id, "DESCRIBES", metadata=dict(seeded))
        self.add_edge(circuit.id, nmh1.id, "MEMBER_OF", metadata=dict(seeded))
        self.add_edge(circuit.id, nmh2.id, "MEMBER_OF", metadata=dict(seeded))
        self.add_edge(circuit.id, s2i.id, "MEMBER_OF", metadata=dict(seeded))

        self.add_edge(s2i.id, nmh1.id, "INHIBITS", weight=-1.0, metadata=dict(seeded))
        self.add_edge(s2i.id, nmh2.id, "INHIBITS", weight=-1.0, metadata=dict(seeded))

        # No confidence_score. This edge asserts that path patching supports the
        # circuit, which is true of the paper and unmeasured here.
        self.add_edge(ev1.id, circuit.id, "SUPPORTS", metadata=dict(seeded))
