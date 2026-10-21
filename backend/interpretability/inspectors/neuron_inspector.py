"""Neuron Inspector.

Analyzes individual neuron activations, activation statistics, firing frequencies,
and top activating tokens across model layers.
"""

from __future__ import annotations

from typing import Any, Dict, List


class NeuronInspector:
    """Inspector for individual neurons and feature activations."""

    def inspect(self, layer: int, neuron_index: int, activations: List[float] | None = None) -> Dict[str, Any]:
        """Refuse: no model and no activation is bound to this inspector.

        Args:
            layer: Echoed back for correlation.
            neuron_index: Echoed back for correlation.
            activations: Accepted for signature compatibility and ignored.

        Returns:
            An unavailable record carrying no activation statistics.

        This previously summarised the literal list
        ``[0.12, 0.45, 2.41, 0.05, 0.88]`` whenever no activations were
        supplied, and ``backend/agents/inspector.py`` wrapped the result as
        ``status: "ok"``. Every number was a constant while the shape was
        indistinguishable from a real inspection. The implemented path is
        ``gpt2_engine.neuron_detail``, which reads one neuron's post-GELU
        activation out of a live forward pass.
        """
        return {
            "layer": layer,
            "neuron_index": neuron_index,
            "neuron_id": f"L{layer}_N{neuron_index}",
            "mean_activation": None,
            "max_activation": None,
            "firing_frequency": None,
            "feature_label": None,
            "status": "unavailable",
            "provenance": "unavailable",
            "measured": False,
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": (
                "Neuron inspection is not implemented in this inspector: no "
                "forward pass was run, so there are no activations to "
                "summarise. See gpt2_engine.neuron_detail."
            ),
        }
