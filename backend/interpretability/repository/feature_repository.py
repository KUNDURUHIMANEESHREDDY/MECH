"""Feature Repository.

Stores and indexes SAE feature metadata, activation values, descriptions,
and dataset examples.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class FeatureRepository:
    """Repository for SAE features and interpretability descriptions."""

    def __init__(self) -> None:
        self._features: Dict[int, Dict[str, Any]] = {
            1402: {
                "feature_id": 1402,
                "label": "Indirect Object Identification",
                "activation": 4.12,
                "description": "Fires on indirect object tokens in IOI sequences",
                "dataset_examples": ["John gave a book to Mary", "Alice sent a letter to Bob"],
            },
            789: {
                "feature_id": 789,
                "label": "Capital Cities Probe",
                "activation": 3.85,
                "description": "Fires on capital city names",
                "dataset_examples": ["The capital of France is Paris", "Tokyo is in Japan"],
            },
        }

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
