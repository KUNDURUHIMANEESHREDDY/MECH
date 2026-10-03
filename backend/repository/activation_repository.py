"""Activation Repository — Data access layer for cached activations.

Responsibilities:
    - Store, retrieve, index, and filter cached activations.
    - Pure data access (NO statistics calculations here).
"""

from __future__ import annotations

import datetime as _dt
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


class LegacyActivationRepository:
    """High-performance activation store and query engine.

    Legacy dict-based repository restored for the pre-consolidation test
    suite (query/search/aggregate/statistics/get_metrics contract).
    """

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, Any]] = {}
        self._hits = 0
        self._misses = 0
        self._lookup_times: list[float] = []
        self._seed_sample_data()

    def _seed_sample_data(self) -> None:
        """Fixture rows so query/search/aggregate have a contract to satisfy.

        These activations (2.41, 3.12, 0.88) were never read off a model. They
        exist to exercise the query layer, and each row now says so via
        `provenance: "reference"` / `measured: False`, so a search returning
        one cannot be mistaken for an observed activation. Downstream code
        should filter on `measured` before drawing any conclusion.
        """
        sample_records = [
            {
                "id": "act_001",
                "layer": 8,
                "head": 9,
                "neuron_index": 402,
                "component": "mlp",
                "token": "circuit",
                "activation": 2.41,
                "session_id": "sess_1",
                "prompt_id": "prompt_ioi",
                "provenance": "reference",
                "measured": False,
                "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            },
            {
                "id": "act_002",
                "layer": 11,
                "head": 12,
                "neuron_index": 118,
                "component": "attention",
                "token": "France",
                "activation": 3.12,
                "session_id": "sess_1",
                "prompt_id": "prompt_cap",
                "provenance": "reference",
                "measured": False,
                "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            },
            {
                "id": "act_003",
                "layer": 0,
                "head": 1,
                "neuron_index": 14,
                "component": "residual",
                "token": "The",
                "activation": 0.88,
                "session_id": "sess_2",
                "prompt_id": "prompt_general",
                "provenance": "reference",
                "measured": False,
                "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            },
        ]
        for rec in sample_records:
            self._cache[rec["id"]] = rec

    def query(
        self,
        layer: Optional[int] = None,
        component: Optional[str] = None,
        token: Optional[str] = None,
        min_activation: Optional[float] = None,
        threshold: Optional[float] = None,
        session_id: Optional[str] = None,
        head: Optional[int] = None,
        prompt_id: Optional[str] = None,
        activation_id: Optional[str] = None,
        top_k: Optional[int] = None,
        sort_by: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Query stored activations using multi-field filtering criteria."""
        t0 = time.perf_counter()
        cutoff = threshold if threshold is not None else min_activation

        results: list[dict[str, Any]] = []
        for rec in self._cache.values():
            if activation_id is not None and rec["id"] != activation_id:
                continue
            if layer is not None and rec["layer"] != layer:
                continue
            if component is not None and rec["component"] != component:
                continue
            if token is not None and rec["token"].lower() != token.lower():
                continue
            if cutoff is not None and rec["activation"] < cutoff:
                continue
            if session_id is not None and rec.get("session_id") != session_id:
                continue
            if head is not None and rec.get("head") != head:
                continue
            if prompt_id is not None and rec.get("prompt_id") != prompt_id:
                continue
            results.append(dict(rec))

        if sort_by == "activation":
            results.sort(key=lambda r: r["activation"], reverse=True)
        elif sort_by == "layer":
            results.sort(key=lambda r: r["layer"])

        if top_k is not None:
            results = results[:top_k]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._lookup_times.append(elapsed_ms)
        if results:
            self._hits += 1
        else:
            self._misses += 1

        return results

    def find(self, activation_id: str) -> Optional[dict[str, Any]]:
        """Lookup a single activation record by ID."""
        res = self.query(activation_id=activation_id)
        return res[0] if res else None

    def aggregate(self, group_by: str = "layer") -> dict[str, Any]:
        """Aggregate activations by layer or component."""
        groups: dict[str, list[float]] = {}
        for rec in self._cache.values():
            key = str(rec.get(group_by, "unknown"))
            groups.setdefault(key, []).append(rec["activation"])

        aggregated = {}
        for key, vals in groups.items():
            aggregated[key] = {
                "count": len(vals),
                "mean": round(sum(vals) / len(vals), 4),
                "max": max(vals),
            }
        return {"grouped_by": group_by, "groups": aggregated}

    def statistics(self) -> dict[str, Any]:
        """Compute overall repository dataset statistics."""
        activations = [r["activation"] for r in self._cache.values()]
        if not activations:
            return {"total": 0, "mean": 0.0, "max": 0.0}
        return {
            "total_records": len(activations),
            "mean_activation": round(sum(activations) / len(activations), 4),
            "max_activation": max(activations),
            "min_activation": min(activations),
        }

    def export(self) -> list[dict[str, Any]]:
        """Export all cached activation records."""
        return [dict(r) for r in self._cache.values()]

    def search(self, search_type: str, query: str = "", **kwargs: Any) -> list[dict[str, Any]]:
        """Generic Search API supporting types: neuron, token, session, experiment, layer, attention_head."""
        search_type = search_type.lower()
        if search_type == "neuron":
            return [
                r for r in self._cache.values()
                if query in f"L{r['layer']}_N{r.get('neuron_index', 0)}" or query in r.get("token", "")
            ]
        if search_type == "token":
            return [r for r in self._cache.values() if query.lower() in r["token"].lower()]
        if search_type in ("session", "experiment"):
            return [r for r in self._cache.values() if r.get("session_id") == query or not query]
        if search_type == "layer":
            layer_num = int(query) if query.isdigit() else None
            return self.query(layer=layer_num)
        if search_type == "attention_head":
            return [r for r in self._cache.values() if r["component"] == "attention"]

        return self.query()

    def get_metrics(self) -> dict[str, Any]:
        """Return comprehensive cache performance statistics."""
        total = self._hits + self._misses
        hit_ratio = round(self._hits / total, 4) if total > 0 else 0.0
        avg_ms = (
            round(sum(self._lookup_times) / len(self._lookup_times), 4)
            if self._lookup_times else 0.0
        )
        timestamps = [r["timestamp"] for r in self._cache.values() if "timestamp" in r]
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_ratio": hit_ratio,
            "cached_items": len(self._cache),
            "memory_bytes": len(str(self._cache).encode("utf-8")),
            "oldest_entry": min(timestamps) if timestamps else None,
            "newest_entry": max(timestamps) if timestamps else None,
            "average_lookup_ms": avg_ms,
        }


_legacy_repo_instance: Optional[LegacyActivationRepository] = None


def get_activation_repository() -> LegacyActivationRepository:
    """Return the legacy dict-based activation repository singleton."""
    global _legacy_repo_instance
    if _legacy_repo_instance is None:
        _legacy_repo_instance = LegacyActivationRepository()
    return _legacy_repo_instance
