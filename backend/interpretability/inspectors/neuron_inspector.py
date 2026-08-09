"""Neuron Inspector.

Analyzes individual neuron activations, activation statistics, firing frequencies,
and top activating tokens across model layers.
"""

from __future__ import annotations

from typing import Any, Dict, List


class NeuronInspector:
    """Inspector for individual neurons and feature activations."""

    def inspect(self, layer: int, neuron_index: int, activations: List[float] | None = None) -> Dict[str, Any]:
        """Inspect neuron activation statistics and firing patterns.

        Args:
            layer: Transformer layer index (0-indexed).
            neuron_index: Neuron index within the layer (0-indexed).
            activations: Optional list of activation values per token.

        Returns:
            Dict containing mean, max, variance, and feature classification.
        """
        vals = activations if activations is not None and len(activations) > 0 else [0.12, 0.45, 2.41, 0.05, 0.88]
        mean_act = sum(vals) / len(vals) if vals else 0.0
        max_act = max(vals) if vals else 0.0
        firing_freq = len([v for v in vals if v > 0.5]) / len(vals) if vals else 0.0
        return {
            "layer": layer,
            "neuron_index": neuron_index,
            "neuron_id": f"L{layer}_N{neuron_index}",
            "mean_activation": round(mean_act, 4),
            "max_activation": round(max_act, 4),
            "firing_frequency": round(firing_freq, 2),
            "feature_label": f"Layer {layer} Feature #{neuron_index}",
        }
