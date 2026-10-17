"""Prediction Inspector.

Analyzes output predictions, logits, probabilities, entropy, top-k tokens,
Logit Lens, and Tuned Lens intermediate layer predictions.
"""

from __future__ import annotations

from typing import Any, Dict, List


class PredictionInspector:
    """Inspector for logits, output probabilities, and intermediate predictions."""

    def inspect(self, prompt: str = "", top_k: int = 5) -> Dict[str, Any]:
        """Refuse: no model, session, or activation is bound to this inspector.

        Args:
            prompt: Echoed back for correlation.
            top_k: Echoed back for correlation.

        Returns:
            An unavailable record carrying no predictions. For the implemented
            logit-lens path see
            `prediction.PredictionInspector.inspect_predictions`.
        """
        # No model, session, or activation is bound to this inspector, so
        # there is nothing to inspect.
        #
        # This previously returned five invented predictions (" Paris" at
        # logit 14.8 / probability 0.82, and four more), a fixed entropy of
        # 0.65, and a logit lens mapping layers 0, 6 and 11 to fixed token
        # strings. Every number was a literal, and the shape was
        # indistinguishable from a real inspection.
        #
        # Real logit-lens inspection needs stored activations and a model: see
        # `prediction.PredictionInspector.inspect_predictions`, which projects
        # the residual stream through `lm_head` and computes top-k and entropy
        # from the result.
        return {
            "prompt": prompt,
            "top_k": top_k,
            "predictions": [],
            "entropy": None,
            "logit_lens": None,
            "status": "unavailable",
            "provenance": "unavailable",
            "measured": False,
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": (
                "Prediction inspection is not implemented in this inspector. "
                "No forward pass was run and no logits were read, so there are "
                "no predictions and no logit-lens projections to report."
            ),
        }
