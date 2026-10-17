"""Feature Repository.

Stores and indexes SAE feature metadata, activation values, descriptions,
and dataset examples.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class FeatureRepository:
    """Repository for SAE features and interpretability descriptions."""

    def __init__(self) -> None:
        # Empty by construction.
        #
        # This previously seeded two complete feature records -- feature 1402
        # "Indirect Object Identification" at activation 4.12, and feature 789
        # "Capital Cities Probe" at 3.85, each with a description and dataset
        # examples. No SAE was trained, no encoder was loaded, and no
        # activation was read for either, so `search_features` returned
        # invented features through the same path as stored ones, and a caller
        # filtering on `min_activation` got hits.
        #
        # Real records need a loader; until one exists this repository holds
        # nothing rather than a fixture.
        self._features: Dict[int, Dict[str, Any]] = {}

    def search_features(self, query: str = "", min_activation: Optional[float] = None) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        q = query.lower()
        for feat in self._features.values():
            if min_activation is not None and feat["activation"] < min_activation:
                continue
            if not q or (
                q in str(feat["feature_id"])
                or q in feat["label"].lower()
                or q in feat["description"].lower()
                or any(q in ex.lower() for ex in feat["dataset_examples"])
            ):
                results.append(dict(feat))
        return results

    def get_feature(self, feature_id: int) -> Optional[Dict[str, Any]]:
        return self._features.get(feature_id)


_feature_repo_instance: FeatureRepository | None = None


def get_feature_repository() -> FeatureRepository:
    global _feature_repo_instance
    if _feature_repo_instance is None:
        _feature_repo_instance = FeatureRepository()
    return _feature_repo_instance
