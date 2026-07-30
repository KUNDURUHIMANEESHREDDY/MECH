"""Epic 2 — Adaptive Cache Engine with Comprehensive Value Eviction Score.

Evaluates activation retention based on expected future value:
Value = (reuse_probability * recomputation_cost_sec * access_frequency * (1 + downstream_dependency_count)) /
        ((1 + time_since_last_access_sec) * memory_cost_mb * retrieval_latency_ms)
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class TensorCacheMetadata:
    tensor_id: str
    layer: int
    importance_score: float
    reuse_probability: float
    recomputation_cost_sec: float
    access_frequency: int
    downstream_dependency_count: int
    time_since_last_access_sec: float
    memory_cost_mb: float
    retrieval_latency_ms: float
    expected_future_value: float
    storage_tier: str  # "VRAM", "NVMe", "System_RAM"
    last_accessed: str


class AdaptiveCacheEngine:
    """Manages learned activation retention and multi-metric value-based cache eviction policies."""

    def __init__(self) -> None:
        self.cached_tensors: Dict[str, TensorCacheMetadata] = {
            "t_l8_act": TensorCacheMetadata(
                tensor_id="t_l8_act",
                layer=8,
                importance_score=0.96,
                reuse_probability=0.92,
                recomputation_cost_sec=2.5,
                access_frequency=14,
                downstream_dependency_count=3,
                time_since_last_access_sec=1.2,
                memory_cost_mb=128.0,
                retrieval_latency_ms=0.5,
                expected_future_value=round((0.92 * 2.5 * 14 * 4) / (2.2 * 128.0 * 0.5), 6),
                storage_tier="VRAM",
                last_accessed=_dt.datetime.utcnow().isoformat() + "Z",
            ),
            "t_l2_act": TensorCacheMetadata(
                tensor_id="t_l2_act",
                layer=2,
                importance_score=0.32,
                reuse_probability=0.20,
                recomputation_cost_sec=0.4,
                access_frequency=2,
                downstream_dependency_count=0,
                time_since_last_access_sec=45.0,
                memory_cost_mb=256.0,
                retrieval_latency_ms=2.0,
                expected_future_value=round((0.20 * 0.4 * 2 * 1) / (46.0 * 256.0 * 2.0), 6),
                storage_tier="NVMe",
                last_accessed=_dt.datetime.utcnow().isoformat() + "Z",
            ),
        }

    def evaluate_retention(
        self,
        tensor_id: str,
        layer: int,
        importance_score: float,
        reuse_probability: float = 0.80,
        recomputation_cost_sec: float = 1.5,
        access_frequency: int = 5,
        downstream_dependency_count: int = 2,
        time_since_last_access_sec: float = 2.0,
        memory_cost_mb: float = 128.0,
        retrieval_latency_ms: float = 1.0,
    ) -> Dict[str, Any]:
        num = reuse_probability * recomputation_cost_sec * (access_frequency or 1) * (1 + downstream_dependency_count)
        den = (1 + time_since_last_access_sec) * (memory_cost_mb or 1.0) * (retrieval_latency_ms or 1.0)
        future_val = round(num / den, 6)
        tier = "VRAM" if future_val >= 0.005 else ("NVMe" if future_val >= 0.00005 else "System_RAM")

        meta = TensorCacheMetadata(
            tensor_id=tensor_id,
            layer=layer,
            importance_score=importance_score,
            reuse_probability=reuse_probability,
            recomputation_cost_sec=recomputation_cost_sec,
            access_frequency=access_frequency,
            downstream_dependency_count=downstream_dependency_count,
            time_since_last_access_sec=time_since_last_access_sec,
            memory_cost_mb=memory_cost_mb,
            retrieval_latency_ms=retrieval_latency_ms,
            expected_future_value=future_val,
            storage_tier=tier,
            last_accessed=_dt.datetime.utcnow().isoformat() + "Z",
        )
        self.cached_tensors[tensor_id] = meta
        return asdict(meta)

    def run_eviction_policy(self, max_vram_tensors: int = 5) -> Dict[str, Any]:
        vram_tensors = [t for t in self.cached_tensors.values() if t.storage_tier == "VRAM"]
        evicted_count = 0

        if len(vram_tensors) > max_vram_tensors:
            vram_tensors.sort(key=lambda t: t.expected_future_value)
            to_evict = vram_tensors[: len(vram_tensors) - max_vram_tensors]
            for t in to_evict:
                t.storage_tier = "NVMe"
                evicted_count += 1

        return {
            "status": "ValueBasedEvictionCompleted",
            "total_cached_tensors": len(self.cached_tensors),
            "vram_tensors_count": len([t for t in self.cached_tensors.values() if t.storage_tier == "VRAM"]),
            "nvme_tensors_count": len([t for t in self.cached_tensors.values() if t.storage_tier == "NVMe"]),
            "evicted_to_nvme_count": evicted_count,
        }

    def list_cached_tensors(self) -> List[Dict[str, Any]]:
        return [asdict(t) for t in self.cached_tensors.values()]
