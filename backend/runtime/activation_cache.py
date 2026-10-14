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

import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Optional

import torch

from .errors import CacheError


#: Sentinel for `search(head=...)` meaning "entries whose head *is* None".
#:
#: `head` is `Optional[int]` on the entry, and `None` already meant "do not
#: filter" in `search`. So the population that actually has no head -- the MLP
#: and residual captures that `run_inference` writes -- was unreachable by
#: head, and `search(head=None)` returned attention and MLP rows mixed
#: together. A caller could not ask for "the headless ones" at all.
#:
#: A distinct sentinel keeps `search(head=None)` meaning no-filter, so no
#: existing caller changes behaviour.
HEADLESS = object()


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
    """In-memory activation cache with search and eviction.

    Ownership
    ---------
    The cache owns what it stores. Both ends are copied:

    * on the way **in**, `put` clones the tensor and copies the entry, so
      neither the tensor nor the metadata can be rewritten from outside;
    * on the way **out**, `get` and `search` hand back copies, so a caller
      holding a result cannot reach in and alter what the cache believes it
      measured.

    The first half was already here and correct. The second was not: `get` and
    `search` returned the stored object by reference, so the clone on the way
    in was undone by any caller that wrote to what it was given. Measured on
    the previous code:

        after get()+mutate, cache value: [999.0, 5.0]

    and the same held for metadata -- setting `entry.layer` moved the entry in
    `search`, which is a measurement index rather than a mutable field.

    Bounding
    --------
    `max_entries` bounds how many; `max_bytes` bounds how much memory, which
    entry count cannot do -- one entry holding a 1 MiB tensor and one holding
    a byte are both "one entry". `max_bytes=None` means unbounded by size.
    """

    def __init__(self, max_entries: int = 10_000,
                 max_bytes: int | None = None):
        self._entries: dict[str, CachedActivation] = {}
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._lock = threading.RLock()
        self._bytes = 0
        self._entry_bytes: dict[str, int] = {}

    # ── memory accounting ────────────────────────────────────────

    @staticmethod
    def _tensor_bytes(tensor: Optional[torch.Tensor]) -> int:
        """Bytes of tensor payload held by an entry.

        Deliberately payload only -- `numel * element_size`. Including Python
        object overhead would make the total depend on CPython's allocator,
        so two identical workloads would report different figures and the
        number could not be checked against a budget.
        """
        if tensor is None:
            return 0
        try:
            return int(tensor.numel()) * int(tensor.element_size())
        except (AttributeError, TypeError, RuntimeError):
            # A stand-in or a meta tensor may not answer; unknown size is
            # treated as zero rather than guessed at.
            return 0

    def bytes_used(self) -> int:
        """Total tensor bytes currently retained."""
        with self._lock:
            return self._bytes

    @property
    def max_bytes(self) -> int | None:
        return self._max_bytes

    @max_bytes.setter
    def max_bytes(self, value: int | None) -> None:
        with self._lock:
            self._max_bytes = value
            self._evict_to_fit(adding_bytes=0)

    # ── eviction ─────────────────────────────────────────────────

    def _drop(self, key: str) -> None:
        """Remove one entry and its byte accounting. Caller holds the lock."""
        self._bytes -= self._entry_bytes.pop(key, 0)
        del self._entries[key]

    def _evict_to_fit(self, adding_bytes: int, incoming_key: str | None = None) -> None:
        """Drop oldest-first until both bounds hold. Caller holds the lock.

        `incoming_key` is the entry about to be inserted. An id already
        present is an *update*, not an insertion, so it must not consume an
        eviction slot -- otherwise overwriting one measurement silently
        discards an unrelated one:

            filled: [0, 1, 2]
            after overwriting an existing id: [1, 2, 99]   <- layer 0 gone
        """
        incoming_key = incoming_key or ""

        # A non-positive capacity means "hold nothing". It must not raise: the
        # previous code called `next(iter(...))` on an empty dict whenever
        # `len >= max_entries`, so capacity 0 raised StopIteration on the first
        # put, with nothing saying what had gone wrong.
        if self._max_entries <= 0:
            for key in list(self._entries):
                self._drop(key)
            return

        # An entry larger than the whole byte budget is refused rather than
        # stored: a cache whose defining property is a memory bound cannot
        # hold an entry that breaks it. Evicting everything else to make room
        # for one oversized value would evict the entire cache and still not
        # satisfy the bound.
        if self._max_bytes is not None and adding_bytes > self._max_bytes:
            raise CacheError(
                f"activation of {adding_bytes} bytes exceeds the cache budget "
                f"of {self._max_bytes} bytes; it was not stored"
            )

        # Prospective size *after* the insert, so the comparison is against
        # what the cache will actually hold. The earlier version compared
        # against the current size, which let the bound be exceeded by exactly
        # one entry on every insert.
        prospective = len(self._entries)
        if incoming_key not in self._entries:
            prospective += 1

        while prospective > self._max_entries and self._entries:
            oldest = self._oldest_excluding(incoming_key)
            if oldest is None:
                break
            self._drop(oldest)
            prospective -= 1

        if self._max_bytes is None:
            return
        while self._bytes + adding_bytes > self._max_bytes and self._entries:
            oldest = self._oldest_excluding(incoming_key)
            if oldest is None:
                break
            self._drop(oldest)

    def _oldest_excluding(self, skip: str) -> str | None:
        """Oldest key other than `skip`, or None if only `skip` remains.

        Caller holds the lock. Never evict the entry being written.
        """
        for key in self._entries:
            if key != skip:
                return key
        return None

    # ── CRUD ─────────────────────────────────────────────────────

    def put(self, activation: CachedActivation) -> str:
        """Store an activation, copying both the tensor and the entry.

        The tensor clone is what stops a caller reusing its buffer from
        rewriting a cached measurement afterwards: the entry was stored by
        reference, so any later in-place write silently changed a value the
        cache reported under an activation_id and timestamp claiming it was
        measured. `run_inference` hands the same store to a hook and keeps
        using it.

        The entry copy closes the other half. The tensor clone alone left the
        *metadata* shared, so `entry.layer = 99` on the caller's object still
        moved the stored entry in `search`.
        """
        # Outside the lock: a failing clone must not consume an eviction slot,
        # and cloning is the expensive part.
        stored = replace(activation)
        if stored.tensor is not None:
            stored.tensor = stored.tensor.detach().clone()
        if stored.metadata:
            # `metadata` is documented as a dict but a caller can put anything
            # there, and `dict("not a dict")` raises ValueError from inside
            # `put` -- turning a malformed field into a refused measurement
            # with a message about dictionary unpacking.
            try:
                stored.metadata = dict(stored.metadata)
            except (TypeError, ValueError):
                stored.metadata = {} if not isinstance(
                    stored.metadata, dict) else stored.metadata
        stored_bytes = self._tensor_bytes(stored.tensor)

        with self._lock:
            key = stored.activation_id
            is_update = key in self._entries

            # A non-positive capacity holds nothing, including this.
            if self._max_entries <= 0:
                return stored.activation_id

            if is_update:
                self._bytes -= self._entry_bytes.pop(key, 0)

            self._evict_to_fit(stored_bytes, incoming_key=key)

            self._entries[key] = stored
            self._entry_bytes[key] = stored_bytes
            self._bytes += stored_bytes
        return stored.activation_id

    def get(self, activation_id: str) -> Optional[CachedActivation]:
        """Return a copy, so the caller cannot reach the cached value."""
        with self._lock:
            entry = self._entries.get(activation_id)
        return None if entry is None else self._copy(entry)

    def delete(self, activation_id: str) -> bool:
        with self._lock:
            if activation_id not in self._entries:
                return False
            self._drop(activation_id)
            return True

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._entry_bytes.clear()
            self._bytes = 0

    def clear_session(self, session_id: str) -> int:
        with self._lock:
            keys = [aid for aid, a in self._entries.items()
                    if a.session_id == session_id]
            for k in keys:
                self._drop(k)
            return len(keys)

    @staticmethod
    def _copy(entry: CachedActivation) -> CachedActivation:
        """A detached copy of one entry, tensor included."""
        clone = replace(entry)
        if clone.tensor is not None:
            clone.tensor = clone.tensor.detach().clone()
        if clone.metadata:
            clone.metadata = dict(clone.metadata)
        return clone

    # ── Search ───────────────────────────────────────────────────

    def search(
        self,
        session_id: str | None = None,
        layer: int | None = None,
        component: str | None = None,
        head: int | None | object = None,
        prompt_id: str | None = None,
    ) -> list[CachedActivation]:
        """Filtered copies. Filtering is read access; it must not confer write."""
        with self._lock:
            results = list(self._entries.values())

        if session_id is not None:
            results = [a for a in results if a.session_id == session_id]
        if layer is not None:
            results = [a for a in results if a.layer == layer]
        if component is not None:
            results = [a for a in results if a.component == component]
        if head is HEADLESS:
            # Explicitly "no head", which is distinct from "do not filter".
            results = [a for a in results if a.head is None]
        elif head is not None:
            results = [a for a in results if a.head == head]
        if prompt_id is not None:
            results = [a for a in results if a.prompt_id == prompt_id]

        return [self._copy(a) for a in results]

    def count(self) -> int:
        with self._lock:
            return len(self._entries)

    def __repr__(self) -> str:
        return (f"ActivationCache(entries={len(self._entries)}, "
                f"bytes={self._bytes})")

    def __repr__(self) -> str:
        return f"ActivationCache(entries={len(self._entries)})"


# Global singleton
cache = ActivationCache()
