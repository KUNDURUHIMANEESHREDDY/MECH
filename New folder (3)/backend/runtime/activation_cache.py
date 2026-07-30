"""Epic 5: Activation Cache v2 — searchable metadata cache.

Each cached activation has rich metadata:
    - activation_id: UUID
    - session_id: str
    - prompt_id: str
    - layer: int
    - head: int | None
    - component: str (e.g. "attention", "mlp", "residual", "embedding")
    - shape: tuple[int, ...]
    - tensor: torch.Tensor (lazy-loaded if disk-backed)
    - timestamp: float
    - metadata: dict (extensible key-value store)

Supports searching by session, layer, component, etc.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from .errors import CacheError


@dataclass
class CachedActivation:
    activation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    prompt_id: str = ""
    layer: int = 0
    head: Optional[int] = None
    component: str = ""
    shape: tuple[int, ...] = ()
    tensor: Optional[torch.Tensor] = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class ActivationCache:
    """In-memory activation cache with search and eviction."""

    def __init__(self, max_entries: int = 10_000):
        self._entries: dict[str, CachedActivation] = {}
        self._max_entries = max_entries

    # ── CRUD ─────────────────────────────────────────────────────

    def put(self, activation: CachedActivation) -> str:
        if len(self._entries) >= self._max_entries:
            # FIFO eviction
            oldest_key = next(iter(self._entries))
            del self._entries[oldest_key]
        self._entries[activation.activation_id] = activation
        return activation.activation_id

    def get(self, activation_id: str) -> Optional[CachedActivation]:
        return self._entries.get(activation_id)

    def delete(self, activation_id: str) -> bool:
        if activation_id in self._entries:
            del self._entries[activation_id]
            return True
        return False

    def clear(self) -> None:
        self._entries.clear()

    def clear_session(self, session_id: str) -> int:
        keys = [aid for aid, a in self._entries.items() if a.session_id == session_id]
        for k in keys:
            del self._entries[k]
        return len(keys)

    # ── Search ───────────────────────────────────────────────────

    def search(
        self,
        session_id: str | None = None,
        layer: int | None = None,
        component: str | None = None,
        head: int | None = None,
        prompt_id: str | None = None,
    ) -> list[CachedActivation]:
        results = list(self._entries.values())
        if session_id is not None:
            results = [a for a in results if a.session_id == session_id]
        if layer is not None:
            results = [a for a in results if a.layer == layer]
        if component is not None:
            results = [a for a in results if a.component == component]
        if head is not None:
            results = [a for a in results if a.head == head]
        if prompt_id is not None:
            results = [a for a in results if a.prompt_id == prompt_id]
        return results

    def count(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"ActivationCache(entries={len(self._entries)})"


# Global singleton
cache = ActivationCache()
