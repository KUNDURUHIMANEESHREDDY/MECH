"""Activation Repository & Query Engine.

Provides activation storage, rich multi-field query capabilities, aggregation,
statistics, export functions, and cache performance telemetry.
"""

from __future__ import annotations

import datetime as _dt
import time
from typing import Any, Dict, List, Optional


class ActivationRepository:
    """High-performance activation store and query engine."""

    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._hits = 0
        self._misses = 0
        self._lookup_times: List[float] = []
        self._seed_sample_data()

    def _seed_sample_data(self) -> None:
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
    ) -> List[Dict[str, Any]]:
        """Query stored activations using multi-field filtering criteria."""
        t0 = time.perf_counter()
        cutoff = threshold if threshold is not None else min_activation

        results: List[Dict[str, Any]] = []
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

    def find(self, activation_id: str) -> Optional[Dict[str, Any]]:
        """Lookup a single activation record by ID."""
        res = self.query(activation_id=activation_id)
        return res[0] if res else None

    def aggregate(self, group_by: str = "layer") -> Dict[str, Any]:
        """Aggregate activations by layer or component."""
        groups: Dict[str, List[float]] = {}
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

    def statistics(self) -> Dict[str, Any]:
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

    def export(self) -> List[Dict[str, Any]]:
        """Export all cached activation records."""
        return [dict(r) for r in self._cache.values()]

    def search(self, search_type: str, query: str = "", **kwargs: Any) -> List[Dict[str, Any]]:
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

    def get_metrics(self) -> Dict[str, Any]:
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


_activation_repo_instance: ActivationRepository | None = None


def get_activation_repository() -> ActivationRepository:
    global _activation_repo_instance
    if _activation_repo_instance is None:
        _activation_repo_instance = ActivationRepository()
    return _activation_repo_instance
