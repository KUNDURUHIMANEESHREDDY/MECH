"""Experiment Priority Scheduler Engine."""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List
from .workers import WorkerRegistry


class ExperimentSchedulerEngine:
    """Priority queue experiment scheduler with worker assignment."""

    def __init__(self) -> None:
        self.worker_registry = WorkerRegistry()
        self._queue: List[Dict[str, Any]] = []

    def submit_job(self, job_id: str, priority: int = 1, spec: Dict[str, Any] | None = None) -> Dict[str, Any]:
        worker = self.worker_registry.get_available_worker()
        job = {
            "job_id": job_id,
            "priority": priority,
            "spec": spec or {},
            "assigned_worker": worker["worker_id"],
            "status": "scheduled",
            "submitted_at": _dt.datetime.utcnow().isoformat() + "Z",
        }
        self._queue.append(job)
        self._queue.sort(key=lambda x: x["priority"], reverse=True)
        return job

    def list_jobs(self) -> List[Dict[str, Any]]:
        return self._queue
