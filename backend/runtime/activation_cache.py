"""Epic 5: Activation Cache v2 — searchable metadata & CAS artifact cache.

Each cached activation has rich metadata:
    - activation_id: UUID or SHA-256 CAS key
    - session_id: str
    - prompt_id: str
    - layer: int
    - head: int | None
    - component: str (e.g. "attention", "mlp", "residual", "embedding")
    - shape: tuple[int, ...]
    - tensor: torch.Tensor (lazy-loaded if disk-backed)
    - timestamp: float
    - metadata: dict (extensible key-value store)

Backed by the unified MECH Execution Artifact Store.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

from .artifacts import (
    ArtifactMetadata,
    ArtifactStore,
    ExecutionArtifact,
    Provenance,
    compute_artifact_key,
    get_artifact_store,
)
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
    """In-memory & CAS-backed activation cache with search and eviction."""

    def __init__(self, max_entries: int = 10_000, store: Optional[ArtifactStore] = None):
        self._entries: dict[str, CachedActivation] = {}
        self._max_entries = max_entries
        self._store = store or get_artifact_store()

    # ── CRUD ─────────────────────────────────────────────────────

    def put(self, activation: CachedActivation, persist: bool = False) -> str:
        if len(self._entries) >= self._max_entries:
            # FIFO eviction
            oldest_key = next(iter(self._entries))
            del self._entries[oldest_key]
        self._entries[activation.activation_id] = activation

        # Bridge to ArtifactStore if tensor is present or persist requested
        try:
            prov = Provenance(
                model_id=activation.metadata.get("model_id", "default_model"),
                operation=activation.metadata.get("operation", "activation_capture"),
                operation_params={"head": activation.head, "component": activation.component},
            )
            meta = ArtifactMetadata(
                name=f"{activation.component}_L{activation.layer}",
                component=activation.component,
                layer=activation.layer,
                head=activation.head,
                shape=activation.shape if activation.shape else (tuple(activation.tensor.shape) if activation.tensor is not None else ()),
                dtype=str(activation.tensor.dtype) if activation.tensor is not None else "float32",
                session_id=activation.session_id,
                prompt_id=activation.prompt_id,
                created_at=activation.timestamp,
                custom=activation.metadata,
            )
            art = ExecutionArtifact(
                artifact_id=activation.activation_id,
                metadata=meta,
                provenance=prov,
                parent_ids=activation.metadata.get("parent_ids", []),
                tensor=activation.tensor,
            )
            self._store.put(art, persist_to_disk=persist)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.debug("Failed to cache activation artifact: %s", exc)

        return activation.activation_id

    def get(self, activation_id: str) -> Optional[CachedActivation]:
        if activation_id in self._entries:
            return self._entries[activation_id]
        
        # Check artifact store
        art = self._store.get(activation_id)
        if art is not None:
            act = CachedActivation(
                activation_id=art.artifact_id,
                session_id=art.metadata.session_id,
                prompt_id=art.metadata.prompt_id,
                layer=art.metadata.layer or 0,
                head=art.metadata.head,
                component=art.metadata.component,
                shape=art.metadata.shape,
                tensor=art.get_tensor(),
                timestamp=art.metadata.created_at,
                metadata=art.metadata.custom,
            )
            self._entries[activation_id] = act
            return act

        return None

    def delete(self, activation_id: str) -> bool:
        deleted = False
        if activation_id in self._entries:
            del self._entries[activation_id]
            deleted = True
        if self._store.delete(activation_id):
            deleted = True
        return deleted

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
