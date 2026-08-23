"""Logit Lens Algorithm.

Applies direct unembedding matrix projection to intermediate layer residual stream vectors.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from backend.runtime.logits import IntermediateLogitsEngine


class ProjectionModel:
    """Interface for layer projection models (Logit Lens, Tuned Lens)."""

    def project(self, prompt: str, layer: int) -> Dict[str, Any]:
        """Project intermediate activations at the given layer to vocabulary logits."""
        raise NotImplementedError("Subclasses must implement project()")


class LogitLens(ProjectionModel):
    """Logit Lens unembedding projection."""

    def __init__(self, logits_engine: Optional[IntermediateLogitsEngine] = None) -> None:
        self.logits_engine = logits_engine or IntermediateLogitsEngine()

    def project(self, prompt: str, layer: int) -> Dict[str, Any]:
        raw_res = self.logits_engine.extract_logits(prompt, num_layers=layer + 1)
        target_layer = raw_res["layer_projections"][-1]
        return {
            "method": "LogitLens",
            "prompt": prompt,
            "layer": layer,
            "top_token": target_layer["top_prediction"],
            "top_logit": target_layer["top_logit"],
            "entropy": target_layer["entropy"],
            "top_k_tokens": target_layer.get("top_k_tokens", []),
        }
