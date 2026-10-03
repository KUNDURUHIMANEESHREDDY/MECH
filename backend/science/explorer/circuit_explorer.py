"""Circuit Explorer — Circuit-centric exploration.

Starts from a named circuit and expands to its attention heads, MLP neurons,
SAE features, tokens, and output logits with cross-navigation links.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class CircuitNode:
    node_id: str
    node_type: str          # "attention_head", "mlp_neuron", "sae_feature", "token", "logit"
    label: str
    layer: int
    index: int
    importance_score: float
    description: str
    links: List[str]        # node_ids this node connects to


@dataclass
class CircuitGraph:
    circuit_id: str
    circuit_name: str
    paper_reference: str
    arxiv_id: str
    model_id: str
    description: str
    nodes: List[CircuitNode]
    edges: List[Dict[str, Any]]  # {"source", "target", "weight", "edge_type"}
    faithfulness: float
    completeness: float
    minimality: float


_CIRCUITS: Dict[str, Dict[str, Any]] = {
    "ioi_circuit": {
        "name": "Indirect Object Identification Circuit",
        "paper": "Wang et al. 2022",
        "arxiv_id": "2211.00593",
        "description": "Identifies the indirect object in 'When Mary and John went to the store, John gave a drink to ___'.",
        "faithfulness": 0.86, "completeness": 0.80, "minimality": 0.92,
        "nodes": [
            ("attn_L5H1",  "attention_head", "L5H1 Induction",        5, 1,  0.91, "Induction head — copies previous token pattern"),
            ("attn_L5H5",  "attention_head", "L5H5 Induction",        5, 5,  0.88, "Induction head — attends to prior occurrence"),
            ("attn_L5H8",  "attention_head", "L5H8 Prev Token",       5, 8,  0.74, "Previous token head — provides positional context"),
            ("attn_L5H9",  "attention_head", "L5H9 S-Inhibition",     5, 9,  0.82, "Suppresses subject name tokens at output"),
            ("attn_L9N9",  "attention_head", "L9H9 Name Mover",       9, 9,  0.95, "Primary name mover — writes IO name to residual stream"),
            ("attn_L10H0", "attention_head", "L10H0 Name Mover",     10, 0,  0.93, "Secondary name mover — reinforces IO name"),
            ("attn_L9H6",  "attention_head", "L9H6 Neg Name Mover",   9, 6,  0.71, "Negative name mover — suppresses subject token"),
            ("mlp_L8N42",  "mlp_neuron",     "L8N42 Positional MLP",  8, 42, 0.68, "MLP neuron encoding token position information"),
        ],
        "edges": [
            ("attn_L5H1",  "attn_L9N9",  0.85, "residual"),
            ("attn_L5H5",  "attn_L9N9",  0.80, "residual"),
            ("attn_L5H9",  "attn_L9N9",  0.77, "suppression"),
            ("attn_L9N9",  "attn_L10H0", 0.82, "residual"),
            ("mlp_L8N42",  "attn_L9N9",  0.65, "positional"),
        ],
    },
    "greater_than_circuit": {
        "name": "Greater-Than Arithmetic Circuit",
        "paper": "Hanna et al. 2023",
        "arxiv_id": "2305.00586",
        "description": "Computes greater-than comparisons on year pairs in GPT-2 Small.",
        "faithfulness": 0.82, "completeness": 0.78, "minimality": 0.88,
        "nodes": [
            ("mlp_L7N100", "mlp_neuron",     "L7N100 Year Encoder",   7, 100, 0.88, "Encodes numeric year value in residual stream"),
            ("mlp_L8N50",  "mlp_neuron",     "L8N50 Comparator",      8, 50,  0.91, "Primary comparison unit — fires for year > start"),
            ("mlp_L9N200", "mlp_neuron",     "L9N200 Output Writer",  9, 200, 0.82, "Writes comparison result to output distribution"),
            ("attn_L3H5",  "attention_head", "L3H5 Digit Head",       3, 5,   0.72, "Attends to digit tokens in year string"),
        ],
        "edges": [
            ("attn_L3H5",  "mlp_L7N100", 0.70, "residual"),
            ("mlp_L7N100", "mlp_L8N50",  0.88, "residual"),
            ("mlp_L8N50",  "mlp_L9N200", 0.85, "residual"),
        ],
    },
}


def _reference_fields(*fields: str) -> Dict[str, str]:
    return {field: "reference" for field in fields}


class CircuitExplorer:
    """Circuit-centric exploration starting from circuit → components → tokens."""

    def list_circuits(self) -> List[Dict[str, Any]]:
        """The published circuit reference catalogue.

        `_CIRCUITS` holds figures transcribed from Wang et al. 2022. That is
        legitimate reference material, but the three metrics were not named as
        published values, so a caller could read "faithfulness: 0.86" as
        something this system measured on this model. They are now labelled
        with their source and marked as not measured here.
        """
        return [
            {
                "circuit_id": cid,
                "name": c["name"],
                "paper": c["paper"],
                "arxiv_id": c["arxiv_id"],
                "description": c["description"],
                "faithfulness": c["faithfulness"],
                "completeness": c["completeness"],
                "minimality": c["minimality"],
                "node_count": len(c["nodes"]),
                "edge_count": len(c["edges"]),
                "provenance": "reference",
                "measured_here": False,
                "metrics_source": (
                    f"{c['paper']} (arXiv:{c['arxiv_id']}), transcribed"
                ),
                "metrics_reason": (
                    "These are the published reference values for this circuit. "
                    "They were not re-measured on this model, so they describe "
                    "the paper and not this system's results."
                ),
                "field_provenance": _reference_fields(
                    "circuit_id", "name", "paper", "arxiv_id", "description",
                    "faithfulness", "completeness", "minimality", "node_count", "edge_count",
                ),
            }
            for cid, c in _CIRCUITS.items()
        ]

    def get_circuit(self, circuit_id: str) -> Optional[Dict[str, Any]]:
        c = _CIRCUITS.get(circuit_id)
        if not c:
            return None

        nodes = [
            CircuitNode(
                node_id=n[0], node_type=n[1], label=n[2], layer=n[3], index=n[4],
                importance_score=n[5], description=n[6],
                links=[e[1] for e in c["edges"] if e[0] == n[0]],
            )
            for n in c["nodes"]
        ]
        edges = [
            {"source": e[0], "target": e[1], "weight": e[2], "edge_type": e[3]}
            for e in c["edges"]
        ]

        graph = CircuitGraph(
            circuit_id=circuit_id,
            circuit_name=c["name"],
            paper_reference=c["paper"],
            arxiv_id=c["arxiv_id"],
            model_id="gpt2-small",
            description=c["description"],
            nodes=nodes,
            edges=edges,
            faithfulness=c["faithfulness"],
            completeness=c["completeness"],
            minimality=c["minimality"],
        )
        result = asdict(graph)
        result["provenance"] = "reference"
        # Same distinction as list_circuits: published values, not local
        # measurements. The node importance scores are likewise transcribed.
        result["measured_here"] = False
        result["metrics_source"] = (
            f"{c['paper']} (arXiv:{c['arxiv_id']}), transcribed"
        )
        result["metrics_reason"] = (
            "Transcribed from the source paper. This circuit was not discovered "
            "or measured by this system; no forward pass produced these values."
        )
        result["validation_eligible"] = False
        result["publication_eligible"] = False
        result["field_provenance"] = _reference_fields(
            "circuit_id", "circuit_name", "paper_reference", "arxiv_id",
            "model_id", "description", "nodes", "edges", "faithfulness",
            "completeness", "minimality",
        )
        return result

    def get_component_detail(self, circuit_id: str, node_id: str) -> Dict[str, Any]:
        """Navigate from circuit → component with links to other circuits/neurons."""
        circuit = _CIRCUITS.get(circuit_id, {})
        node = next((n for n in circuit.get("nodes", []) if n[0] == node_id), None)
        if not node:
            return {"error": f"Node '{node_id}' not found in circuit '{circuit_id}'"}
        return {
            "node_id": node_id,
            "circuit_id": circuit_id,
            "node_type": node[1],
            "label": node[2],
            "layer": node[3],
            "index": node[4],
            "importance_score": node[5],
            "description": node[6],
            "navigate_to_neuron": f"/explorer/neuron?model=gpt2-small&layer={node[3]}&neuron={node[4]}",
            "other_circuits_containing_this_node": [
                cid for cid, c in _CIRCUITS.items()
                if cid != circuit_id and any(n[0] == node_id for n in c.get("nodes", []))
            ],
            "provenance": "reference",
            "field_provenance": _reference_fields(
                "node_id", "circuit_id", "node_type", "label", "layer", "index",
                "importance_score", "description", "navigate_to_neuron",
                "other_circuits_containing_this_node",
            ),
        }
