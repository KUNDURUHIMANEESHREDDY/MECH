"""SAE Feature Dictionary.

Manages feature metadata, activation frequency statistics, and feature indices.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SAEFeature:
    """Rich data model for an SAE feature."""
    id: str # UUID
    repo_id: str
    version: str
    index: int
    model_id: str
    layer: int
    label: str = "Unlabeled"
    description: str = ""
    firing_freq: float = 0.0
    max_activation: float = 0.0
    top_tokens: List[str] = field(default_factory=list)
    top_examples: List[Dict[str, Any]] = field(default_factory=list)
    neuronpedia_url: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


class FeatureProxy:
    """Lazy-loading proxy for SAE features."""

    def __init__(self, feature_id: str, dictionary: FeatureDictionary) -> None:
        self.id = feature_id
        self._dictionary = dictionary
        self._resolved: Optional[SAEFeature] = None

    def resolve(self) -> SAEFeature:
        """Loads the full feature data on demand."""
        if not self._resolved:
            self._resolved = self._dictionary._load_feature_detail(self.id)
        return self._resolved

    def __getattr__(self, name: str) -> Any:
        return getattr(self.resolve(), name)


class FeatureDictionary:
    """Dictionary mapping SAE feature indices to metadata with lazy loading support."""

    def __init__(self, model_id: str = "gpt2-small", layer: int = 8, size: int = 16384) -> None:
        self.model_id = model_id
        self.layer = layer
        self.size = size
        self._proxies: Dict[str, FeatureProxy] = {}
        self._cache: Dict[str, SAEFeature] = {}
        self._initialize_proxies()

    def _initialize_proxies(self) -> None:
        """Seed proxy list (e.g., from a manifest file)."""
        for i in range(self.size):
            fid = f"{self.model_id}_L{self.layer}_F{i}"
            self._proxies[fid] = FeatureProxy(fid, self)

    def _load_feature_detail(self, feature_id: str) -> SAEFeature:
        """Mock: Fetch full detail from local cache or remote API (Neuronpedia)."""
        # In real mode, this would call NeuronpediaClient
        # and handle index-to-UUID resolution
        index = int(feature_id.split("_F")[-1])
        return SAEFeature(
            id=str(uuid.uuid4()),
            repo_id="google/gemma-scope-2b-pt-res",
            version="1.0.0",
            index=index,
            layer=self.layer,
            model_id=self.model_id,
            label=f"Feature {index}",
            description=f"Automated interpretation for feature {index}"
        )

    def get_feature(self, feature_id: str) -> SAEFeature:
        return self._proxies.get(feature_id).resolve() if feature_id in self._proxies else None

    def find_nearest_neighbors(self, feature_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Mock: Returns features with highest cosine similarity to target."""
        # Real implementation would load encoder weights and use FAISS or similar
        return [
            {"id": f"{self.model_id}_L{self.layer}_F{random.randint(0, self.size)}", "similarity": 0.92},
            {"id": f"{self.model_id}_L{self.layer}_F{random.randint(0, self.size)}", "similarity": 0.88}
        ]

    def list_features(self, limit: int = 10) -> List[FeatureProxy]:
        return list(self._proxies.values())[:limit]
