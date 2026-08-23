"""Distributed Task Scheduler.

Manages distributed experiment task queues, topological dependency resolution, 
automatic retry logic, hardware routing, and task progress checkpointing.
"""

from __future__ import annotations

import concurrent.futures
import datetime as _dt
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .resource_manager import ResourceManager, WorkerResourceProfile
from .worker import DistributedWorker, ExperimentTask, WorkerExecutionResult


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
    """Manages experiment task queues, retries, and multi-worker concurrent scheduling."""

    def __init__(
        self,
        resource_manager: Optional[ResourceManager] = None,
        storage_dir: str = "backend/datasets/checkpoints",
        default_workers: int = 4,
    ) -> None:
        self.resource_manager = resource_manager or ResourceManager()
        self.storage_dir = storage_dir
        self.default_workers = default_workers
        self.task_queue: List[ExperimentTask] = []
        self.active_tasks: Dict[str, ExperimentTask] = {}
        self.completed_results: List[WorkerExecutionResult] = []
        self.retry_counts: Dict[str, int] = {}
        self.max_retries = 3
        self._lock = threading.Lock()

    def queue_task(self, task: ExperimentTask) -> None:
        """Enqueues an experiment task into the distributed scheduler."""
        with self._lock:
            self.task_queue.append(task)
            # Sort queue by task priority descending
            self.task_queue.sort(key=lambda x: x.priority, reverse=True)

    def queue_tasks(self, tasks: List[ExperimentTask]) -> None:
        """Enqueues a batch of tasks."""
        with self._lock:
            self.task_queue.extend(tasks)
            self.task_queue.sort(key=lambda x: x.priority, reverse=True)

    def _execute_task_internal(self, task: ExperimentTask) -> WorkerExecutionResult:
        """Executes a single task with safe hardware resource allocation."""
        # 1. Hardware-Aware Worker Selection
        with self._lock:
            worker_profile = self.resource_manager.select_optimal_worker(
                model_id=task.model_id,
                estimated_vram_gb=4.0,
            )
            task.assigned_worker_id = worker_profile.worker_id
            self.resource_manager.allocate_worker_task(worker_profile.worker_id, vram_needed_gb=2.0)
            self.active_tasks[task.task_id] = task

        # 2. Execute Task on Worker Node
        try:
            worker = DistributedWorker(worker_id=worker_profile.worker_id)
            result = worker.execute_task(task)
        finally:
            with self._lock:
                self.resource_manager.release_worker_task(worker_profile.worker_id, vram_freed_gb=2.0)
                self.active_tasks.pop(task.task_id, None)

        # 3. Retry Logic for Failed Tasks
        with self._lock:
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

    def dispatch_next(self) -> Optional[WorkerExecutionResult]:
        """Pops highest-priority task and synchronously executes it."""
        with self._lock:
            if not self.task_queue:
                return None
            task = self.task_queue.pop(0)

        return self._execute_task_internal(task)

    def dispatch_all_parallel(self, max_workers: Optional[int] = None) -> List[WorkerExecutionResult]:
        """Dispatches all enqueued tasks across concurrent cluster workers."""
        workers = max_workers or self.default_workers
        results: List[WorkerExecutionResult] = []

        with self._lock:
            tasks_to_run = list(self.task_queue)
            self.task_queue.clear()

        if not tasks_to_run:
            return results

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self._execute_task_internal, task) for task in tasks_to_run]
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    results.append(
                        WorkerExecutionResult(
                            task_id="unknown",
                            worker_id="pool_error",
                            status="Error",
                            error_message=str(e),
                        )
                    )

        return results

    def dispatch_batch_parallel(
        self,
        tasks: List[ExperimentTask],
        max_workers: Optional[int] = None,
    ) -> List[WorkerExecutionResult]:
        """Executes a given batch of tasks in parallel across worker nodes."""
        self.queue_tasks(tasks)
        return self.dispatch_all_parallel(max_workers=max_workers)

    def checkpoint(self, campaign_id: str) -> SchedulerCheckpoint:
        """Saves scheduler state checkpoint to disk."""
        os.makedirs(self.storage_dir, exist_ok=True)
        with self._lock:
            cp = SchedulerCheckpoint(
                campaign_id=campaign_id,
                total_tasks=len(self.completed_results) + len(self.task_queue) + len(self.active_tasks),
                completed_tasks=sum(1 for r in self.completed_results if r.status == "Success"),
                failed_tasks=sum(1 for r in self.completed_results if r.status == "Error"),
                pending_tasks=len(self.task_queue),
            )
        filepath = os.path.join(self.storage_dir, f"cp_{campaign_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(cp.__dict__, f, indent=2)
        return cp

    def restore_checkpoint(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        """Restores scheduler checkpoint from disk."""
        filepath = os.path.join(self.storage_dir, f"cp_{campaign_id}.json")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def get_scheduler_status(self) -> Dict[str, Any]:
        """Returns live scheduler cluster status and task counts."""
        with self._lock:
            return {
                "pending_tasks": len(self.task_queue),
                "active_tasks": len(self.active_tasks),
                "completed_tasks": sum(1 for r in self.completed_results if r.status == "Success"),
                "failed_tasks": sum(1 for r in self.completed_results if r.status == "Error"),
                "cluster_workers": self.resource_manager.get_cluster_status(),
            }
