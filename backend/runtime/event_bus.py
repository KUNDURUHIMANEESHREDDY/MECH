"""Epic 4: Runtime Event Bus — pub/sub event system with replay.

Every action in the runtime emits events that frontend or other
components can subscribe to.  Events are recorded in an EventLog
for later replay — useful for debugging and the neural debugger.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Callable


# ── Event types ──────────────────────────────────────────────────


@dataclass
class Event:
    type: str
    timestamp: float = field(default_factory=time.time)
    payload: dict[str, Any] = field(default_factory=dict)


# ── Event constants ──────────────────────────────────────────────

MODEL_LOADING = "model.loading"
MODEL_LOADED = "model.loaded"
MODEL_UNLOADED = "model.unloaded"
MODEL_SWITCHED = "model.switched"

SESSION_CREATED = "session.created"
SESSION_CLOSED = "session.closed"

INFERENCE_STARTED = "inference.started"
LAYER_PROCESSED = "inference.layer_processed"
INFERENCE_FINISHED = "inference.finished"
INFERENCE_ERROR = "inference.error"

ACTIVATION_CAPTURED = "activation.captured"
ACTIVATION_CACHED = "activation.cached"

HOOK_REGISTERED = "hook.registered"
HOOK_REMOVED = "hook.removed"
HOOK_ERROR = "hook.error"

CACHE_UPDATED = "cache.updated"
CACHE_HIT = "cache.hit"
CACHE_MISS = "cache.miss"


# ── EventLog (Epic 3: Event Replay) ──────────────────────────────


class EventLog:
    """Append-only event log.  Supports filtering and replay."""

    def __init__(self, max_entries: int = 100_000):
        self._entries: list[Event] = []
        self._max = max_entries

    def record(self, event: Event) -> None:
        if len(self._entries) >= self._max:
            self._entries.pop(0)
        self._entries.append(event)

    def query(
        self,
        event_type: str | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[Event]:
        results = self._entries
        if event_type:
            results = [e for e in results if e.type == event_type]
        if since is not None:
            results = [e for e in results if e.timestamp >= since]
        return results[-limit:]

    def replay(self, fn: Callable[[Event], None], event_type: str | None = None) -> None:
        """Replay all matching events through a callback."""
        for event in self._entries:
            if event_type is None or event.type == event_type:
                try:
                    fn(event)
                except Exception:
                    pass

    def clear(self) -> None:
        self._entries.clear()

    @property
    def count(self) -> int:
        return len(self._entries)

    @property
    def events_per_sec(self) -> float:
        if len(self._entries) < 2:
            return 0.0
        span = self._entries[-1].timestamp - self._entries[0].timestamp
        return len(self._entries) / span if span > 0 else 0.0


# ── Bus ──────────────────────────────────────────────────────────


SubscriberFn = Callable[[Event], None]


class EventBus:
    """Synchronous in-process event bus with event logging."""

    def __init__(self, log: EventLog | None = None):
        self._subscribers: dict[str, list[SubscriberFn]] = {}
        self._log = log or EventLog()

    @property
    def log(self) -> EventLog:
        return self._log

    def on(self, event_type: str, fn: SubscriberFn) -> None:
        self._subscribers.setdefault(event_type, []).append(fn)

    def off(self, event_type: str, fn: SubscriberFn) -> None:
        subs = self._subscribers.get(event_type, [])
        if fn in subs:
            subs.remove(fn)

    def emit(self, event_type: str, **payload: Any) -> None:
        event = Event(type=event_type, payload=payload)
        self._log.record(event)
        for fn in self._subscribers.get(event_type, []):
            try:
                fn(event)
            except Exception:
                pass

    def clear(self) -> None:
        self._subscribers.clear()

    def reset(self) -> None:
        """Clear subscribers AND event log."""
        self._subscribers.clear()
        self._log.clear()


# Global singleton
bus = EventBus()
