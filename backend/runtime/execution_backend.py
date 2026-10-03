"""Execution Backend Abstraction.

Unifies execution behind a single interface to abstract Local CUDA, Ray, Slurm,
and Kubernetes. The planner can select an execution backend rather than
branching logic throughout.
"""

from __future__ import annotations

import itertools
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional


class ExecutionBackend(ABC):
    @abstractmethod
    def submit_job(self, job_func: Callable, *args: Any, **kwargs: Any) -> str:
        """Submit a job and return a job ID."""

    @abstractmethod
    def get_status(self, job_id: str) -> Dict[str, Any]:
        """Check the status of a submitted job."""


class LocalCUDAExecutionBackend(ExecutionBackend):
    """Executes jobs synchronously on the local machine.

    Two things were fixed here.

    ``submit_job`` discarded the job's return value and always returned the
    literal ``"local_job_001"``, so every submission on a machine reported the
    same id and the result of the work was unrecoverable.

    ``get_status`` returned ``{"status": "completed", "progress": 100}`` for
    *any* input, including a job id that was never submitted. A caller polling
    an unknown or typo'd id -- or one from a previous process -- was told the
    work finished at 100%. Progress that cannot fail is not progress.

    Status is now recorded per submission, ids are unique, results are kept,
    and an unknown id is reported as unknown.
    """

    def __init__(self) -> None:
        self._counter = itertools.count(1)
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def submit_job(self, job_func: Callable, *args: Any, **kwargs: Any) -> str:
        job_id = f"local_job_{next(self._counter):04d}"
        started = time.time()
        self._jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "progress": 0,
            "submitted_at": started,
            "result": None,
            "error": None,
        }
        try:
            result = job_func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            # A failing job must not be recorded as complete. This is the
            # specific failure the old get_status could not express.
            self._jobs[job_id].update(
                status="failed", progress=None,
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc()[-800:],
                finished_at=time.time(),
            )
            return job_id
        self._jobs[job_id].update(
            status="completed", progress=100,
            result=result, finished_at=time.time(),
        )
        return job_id

    def get_status(self, job_id: str) -> Dict[str, Any]:
        job = self._jobs.get(job_id)
        if job is None:
            # Never submitted, or from a previous process. Unknown is the only
            # honest answer; "completed" would invent a finished run.
            return {
                "status": "unknown",
                "job_id": job_id,
                "progress": None,
                "reason": (
                    f"No job '{job_id}' is known to this backend. It was never "
                    "submitted here, or belongs to a different process."
                ),
            }
        out = dict(job)
        if out.get("duration_s") is None and job.get("finished_at"):
            out["duration_s"] = round(job["finished_at"] - job["submitted_at"], 6)
        return out

    def get_result(self, job_id: str) -> Optional[Any]:
        """The job's return value, or None if it has not completed."""
        job = self._jobs.get(job_id)
        if not job or job.get("status") != "completed":
            return None
        return job.get("result")
