"""Activation Search Engine.

Executes activation queries using ActivationQuery specification objects.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from repository.activation_repository import get_activation_repository


class ActivationQuery:
    """Query object for filtering model activations."""

    def __init__(
        self,
        layer: Optional[int] = None,
        component: Optional[str] = None,
        threshold: Optional[float] = None,
        token: Optional[str] = None,
        top_k: Optional[int] = None,
        sort_by: Optional[str] = None,
    ) -> None:
        self.layer = layer
        self.component = component
        self.threshold = threshold
        self.token = token
        self.top_k = top_k
        self.sort_by = sort_by

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "component": self.component,
            "threshold": self.threshold,
            "token": self.token,
            "top_k": self.top_k,
            "sort_by": self.sort_by,
        }


class ActivationSearchEngine:
    """Executes ActivationQuery objects against ActivationRepository."""

    def execute_query(self, query: ActivationQuery) -> List[Dict[str, Any]]:
        repo = get_activation_repository()
        return repo.query(
            layer=query.layer,
            component=query.component,
            threshold=query.threshold,
            token=query.token,
            top_k=query.top_k,
            sort_by=query.sort_by,
        )
