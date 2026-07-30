"""Distributed Tensor Cache Manager."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class DistributedTensorCache:
    """Manages shared multi-node activation tensor caches across workers."""

    def __init__(self) -> None:
        self.store: Dict[str, Dict[str, Any]] = {}

    def put_tensor(self, key: str, shape: List[int], dtype: str = "float32", node_id: str = "node_0") -> Dict[str, Any]:
        entry = {"key": key, "shape": shape, "dtype": dtype, "node_id": node_id, "cached": True}
        self.store[key] = entry
        return entry

    def get_tensor(self, key: str) -> Optional[Dict[str, Any]]:
        return self.store.get(key)
