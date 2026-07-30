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
        dto = FeatureInspection(
            feature_id=feature_id,
            activation=4.12,
            label=f"SAE Feature #{feature_id}",
            connected_neurons=[
                {"layer": 8, "neuron": 402, "weight": 0.85},
                {"layer": 9, "neuron": 112, "weight": 0.62},
            ],
            dataset_examples=[
                "John gave a book to Mary",
                "The doctor called the nurse",
            ],
            statistics={
                "firing_freq": 0.042,
                "max_activation": 4.12,
                "mean_activation": 1.25,
            },
        )
        return dto.to_dict()
