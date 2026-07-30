"""Prediction Inspector — Analyzes model predictions, logits, probabilities, entropy, and logit lens.

Renamed from Logit Inspector for better scalability.
"""

from __future__ import annotations

import torch
from typing import Any
from repository.activation_repository import activation_repo
from interpretability.statistics.stats_engine import stats_engine


class PredictionInspector:
    """Inspects prediction distributions, top-K probabilities, and Logit Lens layer projections."""

    def inspect_predictions(
        self,
        session_id: str,
        model: Any = None,
        tokenizer: Any = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        activations = activation_repo.query(session_id=session_id)

        # Logit Lens projection if model unembedding (wte/lm_head) is available
        layer_projections = []
        if model is not None and tokenizer is not None and hasattr(model, "lm_head"):
            res_activations = [a for a in activations if a.component == "residual"]
            for a in sorted(res_activations, key=lambda x: x.layer):
                if a.tensor is not None:
                    with torch.no_grad():
                        # Project residual stream state through lm_head
                        last_token_res = a.tensor[0, -1, :] if a.tensor.dim() == 3 else a.tensor[-1, :]
                        logits = model.lm_head(last_token_res)
                        top_tokens = stats_engine.compute_top_k(logits, tokenizer, top_k=top_k)
                        probs = torch.softmax(logits, dim=-1)
                        entropy = stats_engine.compute_entropy(probs)

                        layer_projections.append({
                            "layer": a.layer,
                            "top_tokens": top_tokens,
                            "entropy": entropy,
                        })

        return {
            "session_id": session_id,
            "layer_projections": layer_projections,
            "has_logit_lens": len(layer_projections) > 0,
        }


prediction_inspector = PredictionInspector()
