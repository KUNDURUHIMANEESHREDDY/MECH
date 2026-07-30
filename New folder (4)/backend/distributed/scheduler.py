"""Distributed Task Scheduler.

Manages distributed experiment task queues, topological dependency resolution, 
automatic retry logic, hardware routing, and task progress checkpointing.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .resource_manager import ResourceManager, WorkerResourceProfile
from .worker import ExperimentTask, WorkerExecutionResult, DistributedWorker


@dataclass
class SchedulerCheckpoint:
    """State checkpoint saved to disk for fault tolerance."""
    campaign_id: str
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    pending_tasks: int
    timestamp: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")


class DistributedScheduler:
    """Manages experiment task queues, retries, and hardware-aware scheduling."""

    def __init__(self, resource_manager: Optional[ResourceManager] = None, storage_dir: str = "backend/datasets/checkpoints") -> None:
        self.resource_manager = resource_manager or ResourceManager()
        self.storage_dir = storage_dir
        self.task_queue: List[ExperimentTask] = []
        self.active_tasks: Dict[str, ExperimentTask] = {}
        self.completed_results: List[WorkerExecutionResult] = []
        self.retry_counts: Dict[str, int] = {}
        self.max_retries = 3

    def queue_task(self, task: ExperimentTask) -> None:
        """Enqueues an experiment task into the distributed scheduler."""
        self.task_queue.append(task)
        # Sort queue by task priority descending
        self.task_queue.sort(key=lambda x: x.priority, reverse=True)

    def dispatch_next(self) -> Optional[WorkerExecutionResult]:
        """Pops highest-priority task, selects optimal worker, and executes task."""
        if not self.task_queue:
            return None

        task = self.task_queue.pop(0)

        # 1. Hardware-Aware Worker Selection
        worker_profile = self.resource_manager.select_optimal_worker(
            model_id=task.model_id,
            estimated_vram_gb=4.0
        )
        task.assigned_worker_id = worker_profile.worker_id

        # 2. Allocate Worker Resources
        self.resource_manager.allocate_worker_task(worker_profile.worker_id, vram_needed_gb=2.0)
        self.active_tasks[task.task_id] = task

        # 3. Execute Task on Worker
        worker = DistributedWorker(worker_id=worker_profile.worker_id)
        result = worker.execute_task(task)

        # 4. Release Worker Resources
        self.resource_manager.release_worker_task(worker_profile.worker_id, vram_freed_gb=2.0)
        del self.active_tasks[task.task_id]

        # 5. Retry Logic for Failed Tasks
        if result.status == "Error":
            retries = self.retry_counts.get(task.task_id, 0)
            if retries < self.max_retries:
                self.retry_counts[task.task_id] = retries + 1
                self.task_queue.append(task)  # Re-enqueue for retry
            else:
                self.completed_results.append(result)
        else:
            self.completed_results.append(result)

        self.checkpoint(task.campaign_id)
        return result

    def checkpoint(self, campaign_id: str) -> SchedulerCheckpoint:
        """Saves scheduler state checkpoint to disk."""
        os.makedirs(self.storage_dir, exist_ok=True)
        cp = SchedulerCheckpoint(
            campaign_id=campaign_id,
            total_tasks=len(self.completed_results) + len(self.task_queue) + len(self.active_tasks),
            completed_tasks=sum(1 for r in self.completed_results if r.status == "Success"),
            failed_tasks=sum(1 for r in self.completed_results if r.status == "Error"),
            pending_tasks=len(self.task_queue)
        )
        filepath = os.path.join(self.storage_dir, f"cp_{campaign_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(cp.__dict__, f, indent=2)
        return cp
