"""Tuned Lens Algorithm.

NOT IMPLEMENTED. A tuned lens applies *learned* layer-specific affine
translators before the unembedding projection. No translators are trained,
loaded, or applied here.

This previously added a flat ``+0.12`` to the top-token probability and reported
``affine_translation_applied: True``, which inflated every confidence it
produced while claiming a learned transformation that never ran. It now
declares itself: the untrained lens is reported as unavailable, the confidence
inflation is gone, and no caller can mistake it for a measured projection.
"""

from __future__ import annotations

from typing import Any, Dict
from .logit_lens import LogitLens, ProjectionModel


class TunedLens(ProjectionModel):
    """Untrained. Reports the plain logit lens and says so."""

    def __init__(self) -> None:
        self.logit_lens = LogitLens()

    def project(self, prompt: str, layer: int) -> Dict[str, Any]:
        base = self.logit_lens.project(prompt, layer)
        base["method"] = "TunedLens"
        # No learned translators exist, so none were applied.
        base["affine_translation_applied"] = False
        base["tuned_lens_available"] = False
        base["provenance"] = "unavailable"
        base["validation_eligible"] = False
        base["publication_eligible"] = False
        base["reason"] = (
            "Tuned lens is not implemented: no layer-specific affine "
            "translators are trained or loaded. The projection below is the "
            "plain logit lens, and no confidence inflation was applied."
        )
        return base
