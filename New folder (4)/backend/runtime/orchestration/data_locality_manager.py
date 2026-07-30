"""Data Locality Manager."""

from __future__ import annotations

from typing import Any, Dict


class DataLocalityManager:
    """Decides optimal storage placements (RAM, GPU VRAM, NVMe, Cloud S3) for tensors and checkpoints."""

    def allocate_tensor_locality(self, tensor_id: str, size_mb: float = 250.0) -> Dict[str, Any]:
        tier = "GPU_VRAM" if size_mb < 1000.0 else "NVMe_CACHE"
        return {
            "tensor_id": tensor_id,
            "size_mb": size_mb,
            "allocated_tier": tier,
            "locality_node": "node_0_gpu_0",
            "eviction_policy": "LRU",
        }
