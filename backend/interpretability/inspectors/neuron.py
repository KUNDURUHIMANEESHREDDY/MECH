"""Neuron, Attention, and Residual Inspectors."""

from __future__ import annotations

from typing import Any
from backend.repository.activation_repository import activation_repo
from backend.science.statistics.stats_engine import stats_engine


class NeuronInspector:
    def inspect_neuron(self, session_id: str, layer: int, neuron_idx: int) -> dict[str, Any]:
        records = activation_repo.query(session_id=session_id, layer=layer, component="mlp")
        if not records or records[0].tensor is None:
            return {"layer": layer, "neuron_idx": neuron_idx, "activation": 0.0}

        tensor = records[0].tensor
        # Mean activation for selected neuron
        if tensor.dim() >= 2:
            val = float(tensor[..., neuron_idx].abs().mean().item()) if neuron_idx < tensor.size(-1) else 0.0
        else:
            val = float(tensor.mean().item())

        return {
            "session_id": session_id,
            "layer": layer,
            "neuron_idx": neuron_idx,
            "activation_value": round(val, 4),
            "stats": stats_engine.compute_tensor_stats(tensor),
        }


class AttentionInspector:
    def inspect_attention(self, session_id: str, layer: int, head: int) -> dict[str, Any]:
        records = activation_repo.query(session_id=session_id, layer=layer, component="attention", head=head)
        if not records or records[0].tensor is None:
            return {"layer": layer, "head": head, "matrix": []}

        matrix = records[0].tensor[0].numpy().tolist() if records[0].tensor.dim() == 3 else records[0].tensor.numpy().tolist()
        return {
            "session_id": session_id,
            "layer": layer,
            "head": head,
            "matrix": matrix,
            "stats": stats_engine.compute_tensor_stats(records[0].tensor),
        }


class ResidualInspector:
    def inspect_residual(self, session_id: str, layer: int) -> dict[str, Any]:
        records = activation_repo.query(session_id=session_id, layer=layer, component="residual")
        if not records or records[0].tensor is None:
            return {"layer": layer, "norm": 0.0}

        stats = stats_engine.compute_tensor_stats(records[0].tensor)
        return {
            "session_id": session_id,
            "layer": layer,
            "stats": stats,
            "l2_norm": stats["l2_norm"],
        }


neuron_inspector = NeuronInspector()
attention_inspector = AttentionInspector()
residual_inspector = ResidualInspector()
