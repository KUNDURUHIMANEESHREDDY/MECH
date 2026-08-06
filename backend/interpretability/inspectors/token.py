"""Token Inspector — Examines single token embeddings, logits, attention, and residual stream."""

from __future__ import annotations

from typing import Any
from backend.repository.activation_repository import activation_repo
from backend.interpretability.statistics.stats_engine import stats_engine


class TokenInspector:
    """Inspects token-level representations across layers."""

    def inspect_token(self, session_id: str, token_idx: int) -> dict[str, Any]:
        activations = activation_repo.query(session_id=session_id)

        embedding_act = next((a for a in activations if a.component == "embedding"), None)
        emb_stats = stats_engine.compute_tensor_stats(embedding_act.tensor) if embedding_act else {}

        layer_residuals = []
        for a in activations:
            if a.component == "residual":
                stats = stats_engine.compute_tensor_stats(a.tensor)
                layer_residuals.append({
                    "layer": a.layer,
                    "l2_norm": stats["l2_norm"],
                    "mean": stats["mean"],
                })

        return {
            "token_idx": token_idx,
            "session_id": session_id,
            "embedding_stats": emb_stats,
            "residual_stream_progression": sorted(layer_residuals, key=lambda x: x["layer"]),
            "activation_count": len(activations),
        }


token_inspector = TokenInspector()
