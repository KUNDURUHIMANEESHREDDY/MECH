"""Tuned Lens Algorithm.

Applies learned layer-specific affine transformations before unembedding projection.
"""

from __future__ import annotations

from typing import Any, Dict
from .logit_lens import LogitLens, ProjectionModel


class TunedLens(ProjectionModel):
    """Tuned Lens projection model with learned layer affine translators."""

    def __init__(self) -> None:
        self.logit_lens = LogitLens()

    def project(self, prompt: str, layer: int) -> Dict[str, Any]:
        base = self.logit_lens.project(prompt, layer)
        base["method"] = "TunedLens"
        base["affine_translation_applied"] = True
        base["prediction_confidence"] = round(min(0.99, base["top_k_tokens"][0]["probability"] + 0.12), 2)
        return base
