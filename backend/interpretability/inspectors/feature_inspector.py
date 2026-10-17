"""Feature Inspector DTO & Engine.

Returns structured FeatureInspection DTOs mapping Feature -> Activation -> Connected Neurons -> Examples.
"""

from __future__ import annotations

from typing import Any, Dict, List


class FeatureInspection:
    """Structured DTO for feature inspection results."""

    def __init__(
        self,
        feature_id: int,
        activation: float,
        label: str,
        connected_neurons: List[Dict[str, Any]],
        dataset_examples: List[str],
        statistics: Dict[str, Any],
    ) -> None:
        self.feature_id = feature_id
        self.activation = activation
        self.label = label
        self.connected_neurons = connected_neurons
        self.dataset_examples = dataset_examples
        self.statistics = statistics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "activation": self.activation,
            "label": self.label,
            "connected_neurons": self.connected_neurons,
            "dataset_examples": self.dataset_examples,
            "statistics": self.statistics,
        }


class FeatureInspector:
    """Inspector creating structured FeatureInspection DTOs."""

    def inspect(self, feature_id: int, prompt: str = "") -> Dict[str, Any]:
        """Refuse: no SAE and no activation is bound to this inspector.

        This previously built a fully populated FeatureInspection for any
        feature id: activation 4.12, connected neurons L8N402 (weight 0.85) and
        L9N112 (weight 0.62), two fixed dataset examples, and statistics
        (firing_freq 0.042, max_activation 4.12, mean_activation 1.25). The
        neuron ids are specific claims about specific neurons, and every field
        was a literal.

        Returns:
            An unavailable record carrying the same keys as a real inspection,
            with no activation, no connected neurons, and no examples. The
            `FeatureInspection` DTO itself is unchanged and is used by callers
            that hold real values.
        """
        return {
            "feature_id": feature_id,
            "activation": None,
            "label": None,
            "connected_neurons": [],
            "dataset_examples": [],
            "statistics": {},
            "status": "unavailable",
            "provenance": "unavailable",
            "measured": False,
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": (
                "Feature inspection is not implemented. No SAE encoder is "
                "loaded and no activation was read, so there is no activation "
                "value, no connected-neuron list, and no dataset evidence to "
                "report."
            ),
        }
