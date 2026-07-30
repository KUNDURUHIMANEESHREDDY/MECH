"""Model Tree Builder — Layer / Head / Neuron navigation tree.

Builds the hierarchical model structure for the Neural Explorer's
left-panel navigation: Model → Layers → (Attention Heads | MLP Neurons).
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..models.adapter_registry import ModelAdapterRegistry


_KNOWN_CIRCUITS: Dict[int, List[str]] = {
    5: ["ioi_circuit"],
    6: ["ioi_circuit"],
    7: ["ioi_circuit", "greater_than_circuit"],
    8: ["greater_than_circuit"],
    9: ["greater_than_circuit"],
}


class ModelTreeBuilder:
    """Builds the model layer/head/neuron navigation tree."""

    def __init__(self) -> None:
        self._registry = ModelAdapterRegistry()

    def build_tree(self, model_id: str = "gpt2-small") -> Dict[str, Any]:
        adapter = self._registry.get_adapter(model_id, mock_mode=True)
        spec = adapter.spec

        layers = []
        for layer_idx in range(spec.num_layers):
            # Attention heads
            heads = [
                {
                    "head_index": h,
                    "label": f"L{layer_idx}H{h}",
                    "known_role": _head_role(layer_idx, h),
                    "is_induction_head": _is_induction_head(layer_idx, h),
                }
                for h in range(spec.num_heads)
            ]
            # MLP neurons — show first 32 in tree (full list paginated via API)
            neurons_preview = [
                {
                    "neuron_index": n,
                    "label": f"N{n}",
                    "circuit_memberships": _KNOWN_CIRCUITS.get(layer_idx, []),
                }
                for n in range(min(32, spec.d_mlp))
            ]
            layers.append({
                "layer_index": layer_idx,
                "label": f"Layer {layer_idx}",
                "num_attention_heads": spec.num_heads,
                "num_mlp_neurons": spec.d_mlp,
                "attention_heads_preview": heads,
                "mlp_neurons_preview": neurons_preview,
                "known_circuits": _KNOWN_CIRCUITS.get(layer_idx, []),
                "residual_stream_dim": spec.d_model,
            })

        return {
            "model_id": model_id,
            "family": spec.family,
            "num_layers": spec.num_layers,
            "num_heads": spec.num_heads,
            "d_model": spec.d_model,
            "d_mlp": spec.d_mlp,
            "vocab_size": spec.vocab_size,
            "context_length": spec.context_length,
            "hf_repo_id": spec.hf_repo_id,
            "layers": layers,
        }

    def list_neurons_in_layer(
        self,
        model_id: str,
        layer: int,
        page: int = 0,
        page_size: int = 64,
    ) -> Dict[str, Any]:
        adapter = self._registry.get_adapter(model_id, mock_mode=True)
        spec = adapter.spec
        start = page * page_size
        end = min(start + page_size, spec.d_mlp)
        neurons = [
            {"neuron_index": n, "label": f"N{n}", "circuit_memberships": _KNOWN_CIRCUITS.get(layer, [])}
            for n in range(start, end)
        ]
        return {
            "model_id": model_id,
            "layer": layer,
            "page": page,
            "page_size": page_size,
            "total_neurons": spec.d_mlp,
            "neurons": neurons,
        }


# ── Helpers ──────────────────────────────────────────────────────────────────

_INDUCTION_HEADS = {(5, 1), (5, 5), (6, 9), (7, 2), (7, 10)}  # GPT-2 Small canonical

def _is_induction_head(layer: int, head: int) -> bool:
    return (layer, head) in _INDUCTION_HEADS

def _head_role(layer: int, head: int) -> str:
    roles = {
        (4, 4): "Duplicate Token Head",
        (4, 11): "Induction Head (weak)",
        (5, 1): "Induction Head",
        (5, 5): "Induction Head",
        (5, 8): "Previous Token Head",
        (5, 9): "S-Inhibition Head",
        (6, 9): "Induction Head",
        (7, 2): "Induction Head",
        (7, 10): "Induction Head",
        (9, 9): "Name Mover Head (IOI)",
        (10, 0): "Name Mover Head (IOI)",
        (9, 6): "Negative Name Mover Head (IOI)",
    }
    return roles.get((layer, head), "")
