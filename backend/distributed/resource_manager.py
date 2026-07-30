"""Distributed Compute Resource Manager.

Monitors GPU/CPU cluster workers, VRAM allocation, CUDA capability, 
model activation caches, and task queue lengths.

Performs hardware-aware task routing:
  - Large Models (>7B) ➔ 80GB GPU Workers
  - Small Models (GPT-2, Gemma-2B) ➔ 24GB GPU Workers
  - Token Data Processing ➔ CPU Worker Cluster
"""

from __future__ import annotations

import datetime as _dt
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class WorkerResourceProfile:
    """Hardware resource profile for a distributed cluster worker."""
    worker_id: str
    device_type: str  # cuda:0, cuda:1, cpu
    device_name: str  # e.g., NVIDIA A100-80GB, RTX 4090, CPU Cluster
    vram_total_gb: float
    vram_used_gb: float
    vram_free_gb: float
    gpu_utilization_pct: float
    cached_models: List[str] = field(default_factory=list)
    active_tasks_count: int = 0
    status: str = "Idle"  # Idle, Busy, Offline


class ResourceManager:
    """Monitors hardware compute resources and dispatches experiments to optimal workers."""

    def __init__(self) -> None:
        self.workers: Dict[str, WorkerResourceProfile] = {}
        self._initialize_default_cluster()

    def _initialize_default_cluster(self) -> None:
        """Initializes default cluster worker profiles."""
        self.workers = {
            "gpu_worker_0": WorkerResourceProfile(
                worker_id="gpu_worker_0",
                device_type="cuda:0",
                device_name="NVIDIA A100-80GB",
                vram_total_gb=80.0,
                vram_used_gb=12.4,
                vram_free_gb=67.6,
                gpu_utilization_pct=15.0,
                cached_models=["gpt2", "gemma"],
                active_tasks_count=0,
                status="Idle"
            ),
            "gpu_worker_1": WorkerResourceProfile(
                worker_id="gpu_worker_1",
                device_type="cuda:1",
                device_name="NVIDIA RTX 4090-24GB",
                vram_total_gb=24.0,
                vram_used_gb=4.2,
                vram_free_gb=19.8,
                gpu_utilization_pct=8.0,
                cached_models=["gpt2"],
                active_tasks_count=0,
                status="Idle"
            ),
            "cpu_worker_cluster": WorkerResourceProfile(
                worker_id="cpu_worker_cluster",
                device_type="cpu",
                device_name="AMD EPYC 64-Core Cluster",
                vram_total_gb=256.0,
                vram_used_gb=18.5,
                vram_free_gb=237.5,
                gpu_utilization_pct=0.0,
                cached_models=[],
                active_tasks_count=0,
                status="Idle"
            )
        }

    def select_optimal_worker(self, model_id: str, estimated_vram_gb: float = 4.0) -> WorkerResourceProfile:
        """Hardware-aware routing selecting optimal cluster worker for a task."""
        # 1. Prefer worker with model already cached in VRAM
        for w in self.workers.values():
            if w.status != "Offline" and model_id.lower() in [m.lower() for m in w.cached_models]:
                if w.vram_free_gb >= estimated_vram_gb:
                    return w

        # 2. Large model routing (> 16GB requirement) ➔ A100 80GB
        if estimated_vram_gb > 16.0 and "gpu_worker_0" in self.workers:
            return self.workers["gpu_worker_0"]

        # 3. Default GPU worker with most free VRAM
        gpu_workers = [w for w in self.workers.values() if w.device_type.startswith("cuda") and w.status != "Offline"]
        if gpu_workers:
            gpu_workers.sort(key=lambda x: x.vram_free_gb, reverse=True)
            return gpu_workers[0]

        # 4. Fallback to CPU Cluster
        return self.workers["cpu_worker_cluster"]

    def allocate_worker_task(self, worker_id: str, vram_needed_gb: float = 2.0) -> None:
        """Allocates a task on a worker, updating resource utilization."""
        if worker_id in self.workers:
            w = self.workers[worker_id]
            w.active_tasks_count += 1
            w.vram_used_gb = min(w.vram_total_gb, w.vram_used_gb + vram_needed_gb)
            w.vram_free_gb = max(0.0, w.vram_total_gb - w.vram_used_gb)
            w.status = "Busy"

    def release_worker_task(self, worker_id: str, vram_freed_gb: float = 2.0) -> None:
        """Releases a task from a worker, freeing VRAM."""
        if worker_id in self.workers:
            w = self.workers[worker_id]
            w.active_tasks_count = max(0, w.active_tasks_count - 1)
            w.vram_used_gb = max(0.0, w.vram_used_gb - vram_freed_gb)
            w.vram_free_gb = min(w.vram_total_gb, w.vram_total_gb - w.vram_used_gb)
            if w.active_tasks_count == 0:
                w.status = "Idle"

    def get_cluster_status(self) -> List[Dict[str, Any]]:
        """Returns hardware cluster status summary."""
        return [w.__dict__ for w in self.workers.values()]
