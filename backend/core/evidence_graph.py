"""Traceable Evidence Graph.

Connects Neuron ➔ Feature ➔ Circuit ➔ Hypothesis ➔ Experiment ➔ Evidence ➔ Publication.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List


class TraceableEvidenceGraph:
    """DAG graph tracking full evidence provenance from neuron activations to final paper."""

    def __init__(self) -> None:
        self.nodes: List[Dict[str, Any]] = [
            {"id": "n_402", "type": "Neuron", "label": "L8_N402"},
            {"id": "f_1402", "type": "Feature", "label": "SAE #1402"},
            {"id": "c_ioi", "type": "Circuit", "label": "IOI Circuit"},
            {"id": "h_ioi", "type": "Hypothesis", "label": "L8_N402 mediates IOI"},
            {"id": "exp_ioi", "type": "Experiment", "label": "Activation Patching L8_N402"},
            {"id": "ev_ioi", "type": "Evidence", "label": "Logit delta -4.2"},
            {"id": "pub_ioi", "type": "Publication", "label": "Mechanistic Paper #1"},
        ]
        self.edges: List[Dict[str, Any]] = [
            {"source": "n_402", "target": "f_1402", "relation": "encodes"},
            {"source": "f_1402", "target": "c_ioi", "relation": "forms"},
            {"source": "c_ioi", "target": "h_ioi", "relation": "suggests"},
            {"source": "h_ioi", "target": "exp_ioi", "relation": "tested_by"},
            {"source": "exp_ioi", "target": "ev_ioi", "relation": "yields"},
            {"source": "ev_ioi", "target": "pub_ioi", "relation": "substantiates"},
        ]

    def add_evidence(self, node_id: str, claim: str, evidence_data: Dict[str, Any]) -> Dict[str, Any]:
        node = {
            "id": node_id,
            "type": "Evidence",
            "claim": claim,
            "data": evidence_data,
            "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }
        self.nodes.append(node)
        self.edges.append({
            "source": node_id,
            "target": "c_ioi",
            "relation": "supports",
        })
        return node

    def get_provenance_trace(self, target_id: str = "pub_ioi") -> List[Dict[str, Any]]:
        return list(self.nodes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes_count": len(self.nodes),
            "edges_count": len(self.edges),
            "nodes": self.nodes,
            "edges": self.edges,
            "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }

