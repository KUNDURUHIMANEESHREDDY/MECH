"""Execution Directed Acyclic Graph (DAG) for Mechanistic Interpretability.

Represents fine-grained computation steps (layers, hooks, SAEs, interventions,
attributions) as a graph of deterministic operations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..artifacts.models import Provenance


@dataclass
class DAGNode:
    """A single deterministic computation step in an interpretability experiment."""
    node_id: str
    node_type: str  # e.g., "embedding", "layer", "sae_encode", "intervention", "logit_lens", "attribution"
    parent_ids: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    layer_idx: Optional[int] = None
    component: str = "residual"

    def compute_cas_key(self, parent_cas_keys: List[str], provenance_digest: str) -> str:
        """Computes deterministic CAS key given parent CAS addresses and node config."""
        payload = {
            "node_type": self.node_type,
            "parent_cas_keys": sorted(parent_cas_keys),
            "params": self.params,
            "layer_idx": self.layer_idx,
            "component": self.component,
            "provenance_digest": provenance_digest,
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "parent_ids": self.parent_ids,
            "params": self.params,
            "layer_idx": self.layer_idx,
            "component": self.component,
        }


class ExecutionDAG:
    """Graph of interpretability operations with dependency tracking."""

    def __init__(self, name: str = "experiment_dag") -> None:
        self.name = name
        self.nodes: Dict[str, DAGNode] = {}
        self.adjacency: Dict[str, List[str]] = {}  # parent_id -> list of child_ids

    def add_node(self, node: DAGNode) -> None:
        self.nodes[node.node_id] = node
        if node.node_id not in self.adjacency:
            self.adjacency[node.node_id] = []
        for p in node.parent_ids:
            if p not in self.adjacency:
                self.adjacency[p] = []
            self.adjacency[p].append(node.node_id)

    def topological_sort(self) -> List[DAGNode]:
        """Returns nodes in topologically sorted execution order."""
        visited: Set[str] = set()
        temp_mark: Set[str] = set()
        order: List[DAGNode] = []

        def visit(nid: str):
            if nid in temp_mark:
                raise ValueError(f"Cycle detected in ExecutionDAG at node: {nid}")
            if nid not in visited:
                temp_mark.add(nid)
                # Visit all parent dependencies first
                node = self.nodes[nid]
                for p in node.parent_ids:
                    if p in self.nodes:
                        visit(p)
                temp_mark.remove(nid)
                visited.add(nid)
                order.append(node)

        for nid in self.nodes:
            if nid not in visited:
                visit(nid)

        return order

    def get_upstream_subgraph(self, target_node_id: str) -> Set[str]:
        """Returns all ancestor node IDs required by target_node_id."""
        ancestors: Set[str] = set()
        queue = [target_node_id]
        while queue:
            curr = queue.pop(0)
            if curr in self.nodes:
                for p in self.nodes[curr].parent_ids:
                    if p not in ancestors:
                        ancestors.add(p)
                        queue.append(p)
        return ancestors

    @classmethod
    def build_transformer_pipeline(
        cls,
        num_layers: int,
        prompt_tokens_hash: str,
        interventions: Optional[Dict[int, Dict[str, Any]]] = None,
        sae_layers: Optional[List[int]] = None,
    ) -> ExecutionDAG:
        """Helper to construct a full transformer forward + intervention + SAE DAG."""
        dag = cls(name="transformer_pipeline")
        interventions = interventions or {}
        sae_layers = sae_layers or []

        # 1. Embedding node
        embed_node = DAGNode(
            node_id="embed",
            node_type="embedding",
            params={"prompt_tokens_hash": prompt_tokens_hash},
            layer_idx=None,
            component="embedding",
        )
        dag.add_node(embed_node)

        prev_node_id = "embed"

        # 2. Sequential Layers
        for l in range(num_layers):
            layer_node_id = f"layer_{l}"
            layer_node = DAGNode(
                node_id=layer_node_id,
                node_type="layer",
                parent_ids=[prev_node_id],
                params={"layer": l},
                layer_idx=l,
                component="residual",
            )
            dag.add_node(layer_node)
            curr_layer_out = layer_node_id

            # Check if intervention attaches to this layer
            if l in interventions:
                int_node_id = f"intervention_L{l}"
                int_node = DAGNode(
                    node_id=int_node_id,
                    node_type="intervention",
                    parent_ids=[layer_node_id],
                    params=interventions[l],
                    layer_idx=l,
                    component="residual_intervened",
                )
                dag.add_node(int_node)
                curr_layer_out = int_node_id

            # Check if SAE attaches to this layer
            if l in sae_layers:
                sae_node_id = f"sae_L{l}"
                sae_node = DAGNode(
                    node_id=sae_node_id,
                    node_type="sae_encode",
                    parent_ids=[curr_layer_out],
                    params={"sae_version": "v1", "layer": l},
                    layer_idx=l,
                    component="sae_latents",
                )
                dag.add_node(sae_node)

            prev_node_id = curr_layer_out

        # 3. Logit head node
        logit_node = DAGNode(
            node_id="unembed_logits",
            node_type="logit_lens",
            parent_ids=[prev_node_id],
            params={},
            layer_idx=num_layers - 1,
            component="logits",
        )
        dag.add_node(logit_node)

        return dag
