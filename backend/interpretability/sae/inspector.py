"""SAE Feature Inspector.

Inspects feature activations, top firing neurons, and dataset examples,
and automatically generates semantic interpretations for features.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .feature_dictionary import FeatureDictionary
from backend.science.models.adapter_base import ModelAdapter
from ..discovery.feature_auto_interpreter import FeatureAutoInterpreter


class SAEInspector:
    """Inspector for Sparse Autoencoder feature activations.

    NOT IMPLEMENTED. No SAE encoder is loaded, so no feature activation, decoder
    weight, or top-firing neuron can be measured. `inspect_feature` previously
    returned the same two hardcoded neuron weights (0.85, 0.62) and the same two
    example prompts with activations (4.2, 3.8) for *every* feature id, so
    `max_act` and `n_examples` computed from them looked like statistics.
    """

    NOT_IMPLEMENTED_REASON = (
        "SAE feature inspection is not implemented: no encoder or decoder "
        "weights are loaded, so this feature was not analysed."
    )

    def __init__(self, adapter: Optional[ModelAdapter] = None) -> None:
        self.dictionary = FeatureDictionary()
        self.adapter = adapter
        self.interpreter = FeatureAutoInterpreter(adapter) if adapter else None

    def inspect_feature(self, feature_id: int) -> Dict[str, Any]:
        feat = self.dictionary.get_feature(feature_id)
        result = {
            "feature": feat,
            "connected_neurons": [],
            "feature_report": {"top_positive_examples": []},
            "status": "unavailable",
            "provenance": "unavailable",
            "inspected": False,
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": self.NOT_IMPLEMENTED_REASON,
        }

        # Only a real interpreter run may contribute activations.
        if self.interpreter is not None:
            try:
                result["feature_report"] = \
                    self.interpreter.generate_feature_report(feature_id)
            except Exception as exc:
                result["reason"] = (
                    f"{self.NOT_IMPLEMENTED_REASON} Interp"
                    f"retation also failed: {exc}")
        return result

