"""Worker Registry for Distributed Scheduler."""

from __future__ import annotations

from typing import Any, Dict, List


class WorkerRegistry:
    """Registry tracking worker nodes, GPU memory, and status."""

    def __init__(self) -> None:
        self._workers: Dict[str, Dict[str, Any]] = {
            "worker_gpu_0": {
                "worker_id": "worker_gpu_0",
                "device": "cuda:0",
                "status": "idle",
                "vram_free_mb": 24576,
                "health": "healthy",
            },
            "worker_gpu_1": {
                "worker_id": "worker_gpu_1",
                "device": "cuda:1",
                "status": "idle",
                "vram_free_mb": 24576,
                "health": "healthy",
            },
        }

    def list_workers(self) -> List[Dict[str, Any]]:
        return list(self._workers.values())

    def get_available_worker(self) -> Dict[str, Any]:
        for w in self._workers.values():
            if w["status"] == "idle":
                return w
        return self._workers["worker_gpu_0"]
