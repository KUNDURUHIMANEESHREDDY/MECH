"""Asynchronous Job Worker Queue for MECH Scientific Platform.

Executes expensive model operations (forward passes, interventions, sweeps, evaluations)
asynchronously in a background worker pool with stage-based progress tracking and cancellation.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from backend.storage.database import DesktopStorage
from backend.storage.scientific_entities import ComputeJob, JobStatus

logger = logging.getLogger("MECH.science.job_queue")


class AsyncJobQueue:
    """Thread-safe background job queue with stage-based progress reporting."""

    def __init__(self, max_workers: int = 2, storage: Optional[DesktopStorage] = None) -> None:
        from pathlib import Path
        db_path = Path.home() / ".cache" / "neural-debugger" / "mech.db"
        self.storage = storage or DesktopStorage(db_path)
        self.storage.initialize()

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="mech_worker")
        self._lock = threading.Lock()
        self._cancellation_tokens: Dict[str, threading.Event] = {}
        self._futures: Dict[str, concurrent.futures.Future] = {}

    def submit_job(
        self,
        job_type: str,
        name: str,
        target_fn: Callable[[Callable[[str, float], None], threading.Event], Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ComputeJob:
        """Submits a job to the background worker pool."""
        job_id = f"job_{uuid.uuid4().hex[:10]}"
        cancel_event = threading.Event()

        job = ComputeJob(
            id=job_id,
            job_type=job_type,
            name=name,
            status=JobStatus.QUEUED,
            progress=0.0,
            current_stage="QUEUED",
            metadata=metadata or {},
        )

        with self._lock:
            self._cancellation_tokens[job_id] = cancel_event
            self.storage.save_job(job.model_dump())

        def _worker_wrapper() -> Any:
            t0 = time.time()
            try:
                # Stage update callback
                def update_stage(stage_name: str, progress: float | None = None) -> None:
                    """Update job stage and progress.

                    If progress is None, stage-based labeling is used rather than
                    fabricated percentages. This avoids implying precision that
                    isn't available.
                    """
                    if cancel_event.is_set():
                        raise RuntimeError("Job cancelled by user.")
                    with self._lock:
                        j = self.storage.get_job(job_id)
                        if j:
                            j["current_stage"] = stage_name
                            # Only set progress if a meaningful value is provided;
                            # otherwise leave for stage-based display
                            if progress is not None:
                                j["progress"] = min(1.0, max(0.0, progress))
                            j["status"] = JobStatus.RUNNING.value
                            self.storage.save_job(j)

                update_stage("PREPARING", 0.05)
                import inspect
                sig = inspect.signature(target_fn)
                if len(sig.parameters) >= 2:
                    result = target_fn(update_stage, cancel_event)
                else:
                    result = target_fn(update_stage)

                if cancel_event.is_set():
                    with self._lock:
                        self.storage.update_job_status(job_id, JobStatus.CANCELLED.value, progress=1.0)
                    return None

                with self._lock:
                    j = self.storage.get_job(job_id)
                    if j:
                        j["status"] = JobStatus.COMPLETED.value
                        j["progress"] = 1.0
                        j["current_stage"] = "COMPLETED"
                        j["completed_at"] = time.time()
                        j["result_artifact_id"] = result.get("id") if isinstance(result, dict) else None
                        self.storage.save_job(j)

                return result

            except Exception as exc:
                logger.exception("Job %s failed: %s", job_id, exc)
                with self._lock:
                    status = JobStatus.CANCELLED.value if cancel_event.is_set() else JobStatus.FAILED.value
                    j = self.storage.get_job(job_id)
                    if j:
                        j["status"] = status
                        j["error"] = str(exc)
                        j["completed_at"] = time.time()
                        self.storage.save_job(j)
                return None

            finally:
                with self._lock:
                    self._cancellation_tokens.pop(job_id, None)
                    self._futures.pop(job_id, None)

        future = self._executor.submit(_worker_wrapper)
        with self._lock:
            self._futures[job_id] = future

        return job

    def cancel_job(self, job_id: str) -> bool:
        """Signals cancellation to a running job."""
        with self._lock:
            if job_id in self._cancellation_tokens:
                self._cancellation_tokens[job_id].set()
                self.storage.update_job_status(job_id, JobStatus.CANCELLED.value)
                return True
            # If still queued
            j = self.storage.get_job(job_id)
            if j and j.get("status") in (JobStatus.QUEUED.value, JobStatus.RUNNING.value):
                self.storage.update_job_status(job_id, JobStatus.CANCELLED.value)
                return True
        return False

    def list_jobs(self, limit: int = 50, status: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.storage.list_jobs(limit=limit, status=status)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.storage.get_job(job_id)


# Global singleton job queue
job_queue = AsyncJobQueue()