"""Layer Inspector — Composable layer breakdown.

Components:
    - Attention Summary
    - MLP Summary
    - Residual Summary
    - Layer Metadata
"""

from __future__ import annotations

from typing import Any
from backend.repository.activation_repository import activation_repo
from backend.science.statistics.stats_engine import stats_engine


class LayerInspector:
    """Inspects a specific layer using composable summaries."""

    def inspect_layer(self, session_id: str, layer_idx: int) -> dict[str, Any]:
        records = activation_repo.query(session_id=session_id, layer=layer_idx)

        attn_records = [r for r in records if r.component == "attention"]
        mlp_records = [r for r in records if r.component == "mlp"]
        res_records = [r for r in records if r.component == "residual"]

        # Attention Summary
        attn_summary = {
            "head_count": len(attn_records),
            "heads": [
                {
                    "head": r.head,
                    "stats": stats_engine.compute_tensor_stats(r.tensor),
                }
                for r in attn_records if r.head is not None
            ],
        }

        # MLP Summary
        mlp_stats = stats_engine.compute_tensor_stats(mlp_records[0].tensor) if mlp_records else {}
        mlp_summary = {
            "active": len(mlp_records) > 0,
            "stats": mlp_stats,
            "sparsity": mlp_stats.get("sparsity", 0.0),
        }

        # Residual Summary
        res_stats = stats_engine.compute_tensor_stats(res_records[0].tensor) if res_records else {}
        res_summary = {
            "active": len(res_records) > 0,
            "stats": res_stats,
            "l2_norm": res_stats.get("l2_norm", 0.0),
        }

        # Layer Metadata
        layer_metadata = {
            "layer_idx": layer_idx,
            "total_cached_components": len(records),
            "session_id": session_id,
        }

        return {
            "metadata": layer_metadata,
            "attention_summary": attn_summary,
            "mlp_summary": mlp_summary,
            "residual_summary": res_summary,
        }


layer_inspector = LayerInspector()
