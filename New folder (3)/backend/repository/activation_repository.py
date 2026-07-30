"""Activation Repository — Data access layer for cached activations.

Responsibilities:
    - Store, retrieve, index, and filter cached activations.
    - Pure data access (NO statistics calculations here).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional
import torch


@dataclass
class ActivationRecord:
    activation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    experiment_id: str = ""
    model_name: str = ""
    prompt_id: str = ""
    layer: int = 0
    head: Optional[int] = None
    component: str = ""  # "attention", "mlp", "residual", "embedding"
    token_idx: Optional[int] = None
    shape: tuple[int, ...] = ()
    tensor: Optional[torch.Tensor] = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class ActivationRepository:
    """In-memory activation repository with indexing and parametric queries."""

    def __init__(self, max_records: int = 10_000):
        self._records: dict[str, ActivationRecord] = {}
        self._by_session: dict[str, set[str]] = {}
        self._by_layer: dict[int, set[str]] = {}
        self._by_component: dict[str, set[str]] = {}
        self._max_records = max_records

    def save(self, record: ActivationRecord) -> str:
        if len(self._records) >= self._max_records:
            oldest_id = next(iter(self._records))
            self.delete(oldest_id)

        self._records[record.activation_id] = record

        # Indexing
        if record.session_id:
            self._by_session.setdefault(record.session_id, set()).add(record.activation_id)
        self._by_layer.setdefault(record.layer, set()).add(record.activation_id)
        if record.component:
            self._by_component.setdefault(record.component, set()).add(record.activation_id)

        return record.activation_id

    def get(self, activation_id: str) -> Optional[ActivationRecord]:
        return self._records.get(activation_id)

    def delete(self, activation_id: str) -> bool:
        record = self._records.pop(activation_id, None)
        if record:
            if record.session_id in self._by_session:
                self._by_session[record.session_id].discard(activation_id)
            if record.layer in self._by_layer:
                self._by_layer[record.layer].discard(activation_id)
            if record.component in self._by_component:
                self._by_component[record.component].discard(activation_id)
            return True
        return False

    def query(
        self,
        session_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        layer: Optional[int] = None,
        component: Optional[str] = None,
        head: Optional[int] = None,
        token_idx: Optional[int] = None,
        min_layer: Optional[int] = None,
        max_layer: Optional[int] = None,
    ) -> list[ActivationRecord]:
        """Parametric query with indexing lookup."""
        candidate_ids: Optional[set[str]] = None

        if session_id is not None:
            candidate_ids = set(self._by_session.get(session_id, set()))
        if component is not None:
            comp_ids = self._by_component.get(component, set())
            candidate_ids = comp_ids if candidate_ids is None else candidate_ids & comp_ids
        if layer is not None:
            layer_ids = self._by_layer.get(layer, set())
            candidate_ids = layer_ids if candidate_ids is None else candidate_ids & layer_ids

        if candidate_ids is not None:
            results = [self._records[aid] for aid in candidate_ids if aid in self._records]
        else:
            results = list(self._records.values())

        # Filtering
        if experiment_id is not None:
            results = [r for r in results if r.experiment_id == experiment_id]
        if head is not None:
            results = [r for r in results if r.head == head]
        if token_idx is not None:
            results = [r for r in results if r.token_idx == token_idx]
        if min_layer is not None:
            results = [r for r in results if r.layer >= min_layer]
        if max_layer is not None:
            results = [r for r in results if r.layer <= max_layer]

        return results

    def clear(self) -> None:
        self._records.clear()
        self._by_session.clear()
        self._by_layer.clear()
        self._by_component.clear()

    def count(self) -> int:
        return len(self._records)


# Singleton instance
activation_repo = ActivationRepository()
