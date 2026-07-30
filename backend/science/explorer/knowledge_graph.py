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
    confidence_score: float = 1.0 # 0.0 to 1.0
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
        confidence_score: float = 1.0,
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
        """Seed initial knowledge graph for UI dev."""
        if self.nodes:
            return
            
        paper = self.add_node("Paper", "IOI Paper (Wang et al.)", node_id="paper_ioi")
        circuit = self.add_node("Circuit", "IOI Circuit", node_id="circuit_ioi")
        
        # Heads
        nmh1 = self.add_node("AttentionHead", "L9H9 (Name Mover)", node_id="head_L9H9", metadata={"layer": 9, "head": 9})
        nmh2 = self.add_node("AttentionHead", "L10H0 (Name Mover)", node_id="head_L10H0", metadata={"layer": 10, "head": 0})
        s2i = self.add_node("AttentionHead", "L7H3 (S2 Inhibition)", node_id="head_L7H3", metadata={"layer": 7, "head": 3})
        
        # Evidence
        ev1 = self.add_node("Evidence", "Path Patching -> NMH", node_id="ev_path_nmh")
        
        # Edges
        self.add_edge(paper.id, circuit.id, "DESCRIBES")
        self.add_edge(circuit.id, nmh1.id, "MEMBER_OF")
        self.add_edge(circuit.id, nmh2.id, "MEMBER_OF")
        self.add_edge(circuit.id, s2i.id, "MEMBER_OF")
        
        self.add_edge(s2i.id, nmh1.id, "INHIBITS", weight=-1.0)
        self.add_edge(s2i.id, nmh2.id, "INHIBITS", weight=-1.0)
        
        self.add_edge(ev1.id, circuit.id, "SUPPORTS", metadata={"confidence_score": 0.95})
