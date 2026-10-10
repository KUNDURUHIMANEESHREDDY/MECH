"""Automated Knowledge Graph Builder.

Consumes execution events from Campaign Manager, Claim Registry, Publication Engine, 
and Paper Replicator to update graph nodes and edges with zero manual synchronization.
"""

from __future__ import annotations

from backend.core.identifiers import content_id

import datetime as _dt
import time
from typing import Any, Dict, List, Optional

from .ontology import NodeType, EdgeType
from .graph_store import GraphStore, KGNode, KGEdge
from backend.interpretability.discovery.mechanism_claim_registry import RegisteredMechanismClaim
from backend.interpretability.discovery.research_campaign_manager import ResearchCampaign


class GraphBuilder:
    """Automated event-driven Knowledge Graph Builder."""

    def __init__(self, store: Optional[GraphStore] = None) -> None:
        self.store = store or GraphStore()

    def inges_campaign(self, campaign: ResearchCampaign) -> None:
        """Converts a Research Campaign and its experiments into graph nodes and edges."""
        camp_node = KGNode(
            node_id=f"camp_{campaign.campaign_id}",
            node_type=NodeType.CAMPAIGN,
            label=campaign.title,
            properties={"status": campaign.status, "goal": campaign.goal, "model": campaign.target_model}
        )
        self.store.add_node(camp_node)

        for exp in campaign.completed_experiments:
            exp_node = KGNode(
                node_id=f"exp_{exp.experiment_id}",
                node_type=NodeType.EXPERIMENT,
                label=f"{exp.algorithm_name} ({exp.target_model})",
                properties={"status": exp.status, "runtime_ms": exp.runtime_ms, "detail": exp.detail}
            )
            self.store.add_node(exp_node)

            edge = KGEdge(
                edge_id=f"edge_camp_exp_{exp.experiment_id}",
                source_id=camp_node.node_id,
                target_id=exp_node.node_id,
                edge_type=EdgeType.CONTAINS
            )
            self.store.add_edge(edge)

    def ingest_claim(self, claim: RegisteredMechanismClaim) -> None:
        """Converts a Registered Mechanism Claim into graph node and links to evidence."""
        claim_node = KGNode(
            node_id=f"claim_{claim.claim_id}",
            node_type=NodeType.MECHANISM_CLAIM,
            label=claim.title,
            properties={"confidence": claim.confidence, "status": claim.status, "replications": claim.replications}
        )
        self.store.add_node(claim_node)

        # Link to referenced papers
        for paper_ref in claim.literature_citations:
            p_id = content_id(paper_ref, prefix="paper_")
            p_node = KGNode(node_id=p_id, node_type=NodeType.PAPER, label=paper_ref)
            self.store.add_node(p_node)

            edge = KGEdge(
                edge_id=f"edge_claim_paper_{p_id}",
                source_id=claim_node.node_id,
                target_id=p_node.node_id,
                edge_type=EdgeType.CITES
            )
            self.store.add_edge(edge)
