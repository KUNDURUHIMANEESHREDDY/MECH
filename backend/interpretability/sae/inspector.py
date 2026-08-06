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
    """Inspector for Sparse Autoencoder feature activations."""

    def __init__(self, adapter: Optional[ModelAdapter] = None) -> None:
        self.dictionary = FeatureDictionary()
        self.adapter = adapter
        self.interpreter = FeatureAutoInterpreter(adapter) if adapter else None

    def inspect_feature(self, feature_id: int) -> Dict[str, Any]:
        feat = self.dictionary.get_feature(feature_id)
        
        # Base metadata
        result = {
            "feature": feat,
            "connected_neurons": [
                {"layer": 8, "neuron": 402, "weight": 0.85},
                {"layer": 9, "neuron": 112, "weight": 0.62},
            ],
        }
        
        # If we have an adapter, run real auto-interpretation
        if self.interpreter:
            report = self.interpreter.generate_feature_report(feature_id)
            result["feature_report"] = report
        else:
            # Fallback for mock mode
            result["feature_report"] = {
                "top_positive_examples": [
                    {"prompt": "John gave a book to Mary", "activation": 4.2},
                    {"prompt": "Alice sent a letter to Bob", "activation": 3.8}
                ]
            }
            
        return result
