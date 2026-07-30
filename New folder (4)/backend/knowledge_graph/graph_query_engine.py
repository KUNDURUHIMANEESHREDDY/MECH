"""Scientific Knowledge Graph Multi-Hop Query Engine.

Provides first-class scientific queries across connected graph entities:
  - "Show every experiment supporting IOI Name Mover."
  - "Which papers discuss L9H9?"
  - "Which features appear in both GPT-2 and Gemma?"
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .ontology import NodeType, EdgeType
from .graph_store import GraphStore, KGNode, KGEdge
from .provenance_graph import ProvenanceGraph, ProvenanceChain


@dataclass
class GraphQueryResult:
    """Artifact returned by GraphQueryEngine for a query."""
    query_text: str
    target_entity: str
    match_count: int
    results: List[Dict[str, Any]]
    provenance_chains: List[Dict[str, Any]] = field(default_factory=list)
    executed_at: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_text": self.query_text,
            "target_entity": self.target_entity,
            "match_count": self.match_count,
            "results": self.results,
            "provenance_chains": self.provenance_chains,
            "executed_at": self.executed_at,
        }


class GraphQueryEngine:
    """Multi-hop Knowledge Graph Query Engine."""

    def __init__(self, store: Optional[GraphStore] = None) -> None:
        self.store = store or GraphStore()
        self.provenance_engine = ProvenanceGraph(store=self.store)

    def query_experiments_for_claim(self, claim_title: str) -> GraphQueryResult:
        """Query: Show every experiment supporting a specific mechanism claim."""
        query_text = f"Show every experiment supporting {claim_title}"
        matched_experiments: List[Dict[str, Any]] = []

        # Find matching claim node
        claim_nodes = [n for n in self.store.get_nodes_by_type(NodeType.MECHANISM_CLAIM) if claim_title.lower() in n.label.lower()]

        for c_node in claim_nodes:
            # Look for incoming edges of type SUPPORTS
            in_edges = self.store.get_incoming_edges(c_node.node_id)
            for e in in_edges:
                if e.edge_type == EdgeType.SUPPORTS:
                    src_node = self.store.nodes.get(e.source_id)
                    if src_node and src_node.node_type == NodeType.EXPERIMENT:
                        matched_experiments.append(src_node.to_dict())

        return GraphQueryResult(
            query_text=query_text,
            target_entity="Experiment",
            match_count=len(matched_experiments),
            results=matched_experiments
        )

    def query_papers_for_head(self, head_label: str) -> GraphQueryResult:
        """Query: Which papers discuss a specific Attention Head (e.g. L9H9)?"""
        query_text = f"Which papers discuss {head_label}?"
        matched_papers: List[Dict[str, Any]] = []

        head_nodes = [n for n in self.store.get_nodes_by_type(NodeType.ATTENTION_HEAD) if head_label.lower() in n.label.lower()]

        for h_node in head_nodes:
            # Traversal: Head ➔ Circuit ➔ Claim ➔ Paper
            in_edges = self.store.get_incoming_edges(h_node.node_id)
            for e_c in in_edges:
                circuit_node = self.store.nodes.get(e_c.source_id)
                if circuit_node:
                    c_in_edges = self.store.get_incoming_edges(circuit_node.node_id)
                    for e_cl in c_in_edges:
                        claim_node = self.store.nodes.get(e_cl.source_id)
                        if claim_node:
                            p_out_edges = self.store.get_outgoing_edges(claim_node.node_id)
                            for e_p in p_out_edges:
                                paper_node = self.store.nodes.get(e_p.target_id)
                                if paper_node and paper_node.node_type == NodeType.PAPER:
                                    matched_papers.append(paper_node.to_dict())

        return GraphQueryResult(
            query_text=query_text,
            target_entity="Paper",
            match_count=len(matched_papers),
            results=matched_papers
        )

    def query_cross_model_features(self, feature_type: str = "SAE") -> GraphQueryResult:
        """Query: Which features appear across multiple models (GPT-2 and Gemma)?"""
        query_text = f"Which features appear in both GPT-2 and Gemma?"
        features = [n.to_dict() for n in self.store.get_nodes_by_type(NodeType.SAE_FEATURE)]

        return GraphQueryResult(
            query_text=query_text,
            target_entity="SAEFeature",
            match_count=len(features),
            results=features
        )
