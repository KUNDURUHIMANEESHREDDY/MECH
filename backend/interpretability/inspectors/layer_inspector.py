"""Layer Inspector.

Profiles layer-level activations, MLP sub-layer vs Attention sub-layer
contributions, and activation norms.
"""

from __future__ import annotations

from typing import Any, Dict


class LayerInspector:
    """Inspector for entire transformer layers."""

    def inspect(self, layer: int) -> Dict[str, Any]:
        """Refuse: no model and no activation is bound to this inspector.

        Args:
            layer: Echoed back for correlation.

        Returns:
            An unavailable record carrying no layer statistics.

        This previously returned ``round(14.2 + layer, 2)`` and
        ``round(10.8 + layer * 0.9, 2)`` with ``status: "computed"`` -- the
        layer index dressed up as a measurement of that layer. The
        implemented path is ``gpt2_engine.layer_detail``.
        """
        return {
            "layer": layer,
            "mlp_activation_norm": None,
            "attention_output_norm": None,
            "active_neurons_count": None,
            "status": "unavailable",
            "provenance": "unavailable",
            "measured": False,
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": (
                "Layer inspection is not implemented in this inspector: no "
                "forward pass was run, so there are no norms to report. See "
                "gpt2_engine.layer_detail."
            ),
        }
