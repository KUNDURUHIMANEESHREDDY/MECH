"""Feature Search Engine.

Searches SAE features across activation magnitudes, descriptions, and dataset examples.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from backend.interpretability.repository.feature_repository import get_feature_repository


class FeatureSearchEngine:
    """Engine for searching SAE feature vectors."""

    def search(self, query: str = "", min_activation: Optional[float] = None) -> List[Dict[str, Any]]:
        repo = get_feature_repository()
        return repo.search_features(query=query, min_activation=min_activation)
