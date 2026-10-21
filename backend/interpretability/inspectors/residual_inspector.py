"""Residual Stream Inspector.

Decomposes and analyzes residual stream vector norms, layer contributions,
and component direct logit attributions.
"""

from __future__ import annotations

from typing import Any, Dict


class ResidualInspector:
    """Inspector for residual stream state evolution across layers."""

    def inspect(self, layer: int, prompt: str = "") -> Dict[str, Any]:
        """Refuse: no model and no activation is bound to this inspector.

        Args:
            layer: Echoed back for correlation.
            prompt: Echoed back for correlation.

        Returns:
            An unavailable record carrying no residual statistics.

        This previously returned ``round(12.4 + layer * 1.8, 2)``,
        ``round(0.95 - layer * 0.05, 3)`` and a contribution percentage built
        from ``layer % 4``: three curves of the layer index, none of them read
        from a residual stream. The implemented path is
        ``gpt2_engine.layer_activations``, which returns the per-token
        ``resid_post`` tensor.
        """
        return {
            "layer": layer,
            "prompt": prompt,
            "norm": None,
            "cosine_similarity_with_input": None,
            "layer_contribution_pct": None,
            "status": "unavailable",
            "provenance": "unavailable",
            "measured": False,
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": (
                "Residual-stream inspection is not implemented in this "
                "inspector: no forward pass was run, so there is no state to "
                "describe. See gpt2_engine.layer_activations."
            ),
        }
