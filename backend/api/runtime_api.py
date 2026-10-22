"""Runtime API v2 -- a durable, bounded job queue.

`backend/main.py:322-336` probes for this module and mounts it at ``/api/v2``
when it exists. Before this file existed the mount was skipped with a warning
and every ``/api/v2`` call was a 404.

What is here, and what is deliberately not
------------------------------------------
A job queue only earns its name if a restart does not lie about the jobs it
lost. So:

* **Jobs are persisted** as one JSON file each under
  ``MECH_V2_JOB_DIR`` (default ``backend/storage/v2_jobs``), written with
  ``os.replace`` so a torn write cannot leave a half record.
* **A restart does not resume work it cannot resume.** Any job found in
  ``queued`` or ``running`` at load time is relabelled ``interrupted`` with a
  reason. It is never silently marked ``completed``.
* **Concurrency is bounded** by ``MECH_V2_MAX_CONCURRENCY`` (default 2) through
  a thread pool, and the queue length by ``MECH_V2_MAX_QUEUED`` (default 64).
* **Cancellation is cooperative.** A running job is asked to stop between
  items; it is reported ``cancelled`` only when the handler says it stopped
  early. Cancelling a job that already finished is a 409, not a silent no-op.

This module does not measure anything itself. The handlers below call the
existing benchmark pipelines, which fail closed with ``unavailable`` plus a
reason when no model is loaded -- and that verdict is propagated, never
converted into a success. Nothing here may report ``provenance: "live"`` for a
number the underlying pipeline did not measure.

Auth: every route lives under ``/api/``, which ``backend/core/auth.py:207-208``
treats as protected. Nothing here is added to ``is_public_path``.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from backend.core.observability import get_settings
from backend.core.recovery import (
    EngineTimeout,
    call_with_timeout,
    cleanup_partial_files,
)

router = APIRouter()

#: Terminal states. A job in one of these is never worked on again.
TERMINAL = frozenset({"completed", "failed", "cancelled", "unavailable",
                      "interrupted"})

JOB_ID_RE_LEN = 12


def _int_env(name: str, default: int, low: int, high: int) -> int:
    raw = os.environ.get(name, "")
    try:
        value = int(str(raw).strip()) if str(raw).strip() else default
    except (TypeError, ValueError):
        return default
    return max(low, min(value, high))


def _max_concurrency() -> int:
    return _int_env("MECH_V2_MAX_CONCURRENCY", 2, 1, 8)


def _max_queued() -> int:
    return _int_env("MECH_V2_MAX_QUEUED", 64, 1, 512)


def _job_dir() -> Path:
    configured = os.environ.get("MECH_V2_JOB_DIR", "").strip()
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parent.parent / "storage" / "v2_jobs"


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _new_job_id() -> str:
    return "j" + uuid.uuid4().hex[:JOB_ID_RE_LEN]


# --------------------------------------------------------------------------- #
# handlers: the actual work a job can do
# --------------------------------------------------------------------------- #


def _clamp_prompts(raw: Any, default: int = 100) -> int:
    """Validate n_prompts the way the HTTP layer expects: raise ValueError.

    The caller turns that into a 400 -- the dispatcher's existing convention --
    rather than letting it surface as a 500.
    """
    if raw is None:
        return default
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError("n_prompts must be an integer")
    if raw < 1 or raw > 1000:
        raise ValueError("n_prompts must be between 1 and 1000")
    return raw


def _handle_benchmark_sweep(payload: Dict[str, Any],
                            should_stop: "threading.Event") -> Dict[str, Any]:
    """Run the named benchmarks one at a time through the live runner.

    Each item carries its own provenance. A benchmark with no live executor
    contributes ``unavailable`` with the runner's reason; the sweep as a whole
    is never reported as ``completed`` on the strength of the ones that did
    run, and never as ``live`` unless every item measured.
    """
    from backend.validation.benchmark_runner import MechanisticBenchmarkRunner

    raw_names = payload.get("benchmarks") or ["IOI"]
    if not isinstance(raw_names, list) or not raw_names:
        raise ValueError("benchmarks must be a non-empty list of names")
    names = [str(name) for name in raw_names][:16]
    n_prompts = _clamp_prompts(payload.get("n_prompts"))

    runner = MechanisticBenchmarkRunner()
    items: List[Dict[str, Any]] = []
    for name in names:
        if should_stop.is_set():
            return {
                "status": "cancelled",
                "reason": "cancelled between benchmark items",
                "n_prompts": n_prompts,
                "items": items,
            }
        try:
            # The runner takes an optional panel size where it supports one.
            # A single benchmark is one engine call, so it gets the engine
            # timeout: a call that does not finish in time is a failure of
            # that item, never a silent success.
            result = call_with_timeout(
                lambda: runner.run_benchmark(f"bench_{name}", n_prompts=n_prompts),
                get_settings().engine_timeout_s,
                label="benchmark %s" % name,
            )
        except TypeError:
            # Older runner signature: run it at its own fixed panel size and
            # say so, rather than pretending the requested size was honoured.
            result = call_with_timeout(
                lambda: runner.run_benchmark(f"bench_{name}"),
                get_settings().engine_timeout_s,
                label="benchmark %s" % name,
            )
            if isinstance(result, dict):
                result = dict(result)
                result["requested_n_prompts"] = n_prompts
                result["n_prompts_honoured"] = False
        except Exception as exc:  # noqa: BLE001 - reported per item
            result = {
                "status": "error",
                "provenance": "unavailable",
                "measured": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        entry = {
            "benchmark_name": name,
            "status": str((result or {}).get("status") or "unavailable"),
            "provenance": str((result or {}).get("provenance") or "unavailable"),
        }
        for key in ("score", "pass_rate", "eval_samples", "n_prompts",
                    "reason", "error", "validation_eligible",
                    "publication_eligible", "ineligible_because"):
            if isinstance(result, dict) and key in result:
                entry[key] = result[key]
        items.append(entry)

    measured = [i for i in items if i.get("provenance") == "live"]
    if measured and len(measured) == len(items):
        status = "completed"
    elif measured:
        status = "completed_with_unavailable"
    else:
        status = "unavailable"
    return {
        "status": status,
        "n_prompts": n_prompts,
        "items": items,
        "measured": len(measured),
        "unavailable": len(items) - len(measured),
    }


def _handle_ioi_stability(payload: Dict[str, Any],
                          should_stop: "threading.Event") -> Dict[str, Any]:
    """Run the IOI reproduction pipeline's stability audit.

    The pipeline raises ``LiveUnavailable`` when the model is not loaded; that
    is reported as an ``unavailable`` job with the pipeline's own reason.
    """
    from backend.science.reproducibility.ioi_pipeline import IOIReproductionPipeline

    n_prompts = _clamp_prompts(payload.get("n_prompts"))
    n_seeds = payload.get("n_seeds", 5)
    if isinstance(n_seeds, bool) or not isinstance(n_seeds, int) or not 1 <= n_seeds <= 10:
        raise ValueError("n_seeds must be an integer between 1 and 10")
    if should_stop.is_set():
        return {"status": "cancelled", "reason": "cancelled before start"}

    pipeline = IOIReproductionPipeline(mock_mode=False)
    result = pipeline.run_stability_audit(n_seeds=n_seeds, n_prompts=n_prompts)
    if not isinstance(result, dict):
        return {"status": "failed",
                "error": "the stability audit returned a non-dict result"}
    result = dict(result)
    result.setdefault("n_prompts", n_prompts)
    result.setdefault("n_seeds", n_seeds)
    return result


#: Job kind -> handler. Unknown kinds are a 400 at submission time, so a caller
#: cannot queue work that has no implementation.
HANDLERS = {
    "benchmark_sweep": _handle_benchmark_sweep,
    "ioi_stability": _handle_ioi_stability,
}


# --------------------------------------------------------------------------- #
# the queue
# --------------------------------------------------------------------------- #


class _JobQueue:
    """Bounded in-process worker with per-job JSON persistence."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._stops: Dict[str, threading.Event] = {}
        self._pool: Optional[ThreadPoolExecutor] = None
        self._pool_workers = 0
        self._loaded = False

    # -- persistence ----------------------------------------------------
    def _path(self, job_id: str) -> Path:
        return _job_dir() / f"{job_id}.json"

    def _persist(self, job: Dict[str, Any]) -> None:
        directory = _job_dir()
        tmp = None
        try:
            directory.mkdir(parents=True, exist_ok=True)
            target = self._path(str(job["job_id"]))
            tmp = target.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(job, handle, indent=2, default=str)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, target)
        except Exception as exc:  # noqa: BLE001 - a persistence failure is recorded
            job["persist_error"] = f"{type(exc).__name__}: {exc}"
            # A failed write leaves a half-written temp file behind. It is
            # not a job record, nothing will ever read it, and it would sit
            # in the job directory looking like work in progress.
            if tmp is not None:
                cleanup_partial_files([tmp])

    def _load(self) -> None:
        """Load persisted jobs. Queued/running jobs become `interrupted`."""
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            directory = _job_dir()
            try:
                names = sorted(p for p in directory.iterdir()
                               if p.suffix == ".json")
            except Exception:
                return
            for path in names:
                try:
                    with open(path, "r", encoding="utf-8") as handle:
                        job = json.load(handle)
                except Exception:
                    continue
                if not isinstance(job, dict) or not job.get("job_id"):
                    continue
                if job.get("status") not in TERMINAL:
                    job["status"] = "interrupted"
                    job["error"] = (
                        "the API process restarted while this job was "
                        f"{job.get('status') or 'queued'}; it was not resumed "
                        "and did not complete")
                    job["finished_at"] = _now()
                    self._persist(job)
                self._jobs[str(job["job_id"])] = job

    # -- pool -----------------------------------------------------------
    def _executor(self) -> ThreadPoolExecutor:
        with self._lock:
            workers = _max_concurrency()
            if self._pool is None or self._pool_workers != workers:
                if self._pool is not None:
                    self._pool.shutdown(wait=False)
                self._pool = ThreadPoolExecutor(
                    max_workers=workers, thread_name_prefix="mech-v2-job")
                self._pool_workers = workers
            return self._pool

    # -- public ---------------------------------------------------------
    def submit(self, kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._load()
        with self._lock:
            active = [j for j in self._jobs.values()
                      if j.get("status") not in TERMINAL]
            if len(active) >= _max_queued():
                raise RuntimeError(
                    f"queue is full: {len(active)} non-terminal job(s), "
                    f"limit {_max_queued()}")
            job = {
                "job_id": _new_job_id(),
                "kind": kind,
                "payload": payload,
                "status": "queued",
                "created_at": _now(),
                "started_at": None,
                "finished_at": None,
                "result": None,
                "error": None,
                "cancel_requested": False,
                "max_concurrency": _max_concurrency(),
            }
            self._jobs[job["job_id"]] = job
            self._stops[job["job_id"]] = threading.Event()
            self._persist(job)
        try:
            self._executor().submit(self._run, job["job_id"])
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                job["status"] = "failed"
                job["error"] = f"could not be scheduled: {exc}"
                job["finished_at"] = _now()
                self._persist(job)
        return dict(job)

    def _run(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            stop = self._stops.get(job_id)
            if job is None or stop is None:
                return
            if job.get("status") in TERMINAL:
                return
            job["status"] = "running"
            job["started_at"] = _now()
            self._persist(job)
            kind = str(job.get("kind"))
            payload = dict(job.get("payload") or {})
        handler = HANDLERS.get(kind)
        if handler is None:
            with self._lock:
                job["status"] = "failed"
                job["error"] = f"no handler for job kind {kind!r}"
                job["finished_at"] = _now()
                self._persist(job)
            return
        try:
            # The whole job gets a budget. A handler that overruns it is
            # reported as failed -- a job that never finishes must not be
            # left showing "running" for ever, and it must never be
            # reported as completed.
            result = call_with_timeout(
                lambda: handler(payload, stop),
                get_settings().job_timeout_s,
                label="job %s" % kind,
            )
            error = None
        except EngineTimeout as exc:
            result = None
            error = f"timeout: {exc}"
        except Exception as exc:  # noqa: BLE001 - a failed job is reported
            result = None
            error = f"{type(exc).__name__}: {exc}"
        with self._lock:
            job["finished_at"] = _now()
            if error is not None:
                job["status"] = "failed"
                job["error"] = error
            else:
                job["result"] = result
                status = str((result or {}).get("status") or "").lower()
                reason = str((result or {}).get("reason")
                             or (result or {}).get("error") or "")
                if status == "cancelled":
                    job["status"] = "cancelled"
                    job["error"] = reason or "cancelled"
                elif status in ("unavailable", "blocked", "not_run"):
                    # A pipeline that could not measure did not succeed.
                    job["status"] = "unavailable"
                    job["error"] = reason or "the handler reported unavailable"
                elif status in ("error", "failed"):
                    job["status"] = "failed"
                    job["error"] = reason or "the handler reported an error"
                else:
                    job["status"] = "completed"
            self._persist(job)

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        self._load()
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def listing(self, status: Optional[str] = None,
                kind: Optional[str] = None,
                limit: int = 50) -> List[Dict[str, Any]]:
        self._load()
        with self._lock:
            jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if str(j.get("status")) == status]
        if kind:
            jobs = [j for j in jobs if str(j.get("kind")) == kind]
        jobs.sort(key=lambda j: str(j.get("created_at") or ""), reverse=True)
        return [dict(j) for j in jobs[:limit]]

    def cancel(self, job_id: str) -> Dict[str, Any]:
        self._load()
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.get("status") in TERMINAL:
                raise RuntimeError(
                    f"job is already {job.get('status')}; it cannot be cancelled")
            job["cancel_requested"] = True
            stop = self._stops.get(job_id)
            if stop is not None:
                stop.set()
            if job.get("status") == "queued":
                # Never started: the terminal state is certain, so record it.
                job["status"] = "cancelled"
                job["finished_at"] = _now()
                job["error"] = "cancelled before it started"
            self._persist(job)
            return dict(job)


QUEUE = _JobQueue()


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #


@router.get("/jobs")
def list_jobs(status: Optional[str] = Query(default=None),
              kind: Optional[str] = Query(default=None),
              limit: int = Query(default=50, ge=1, le=200)) -> Dict[str, Any]:
    """List jobs, newest first. Configured bounds are reported with the list."""
    jobs = QUEUE.listing(status=status, kind=kind, limit=limit)
    return {
        "jobs": jobs,
        "count": len(jobs),
        "bounds": {
            "max_concurrency": _max_concurrency(),
            "max_queued": _max_queued(),
            "job_dir": str(_job_dir()),
        },
        "kinds": sorted(HANDLERS),
        "provenance": "live",
    }


@router.post("/jobs", status_code=202)
def create_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Submit a job. Bad input is a 400, a full queue a 503."""
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="request body must be an object")
    kind = str(payload.get("kind") or "").strip()
    if not kind:
        raise HTTPException(
            status_code=400,
            detail=f"kind is required; one of {sorted(HANDLERS)}")
    if kind not in HANDLERS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown kind {kind!r}; one of {sorted(HANDLERS)}")
    params = payload.get("params") or {}
    if not isinstance(params, dict):
        raise HTTPException(status_code=400, detail="params must be an object")
    try:
        # Validate now rather than inside the worker, so the caller gets a 400
        # instead of a job that fails a second later.
        if "n_prompts" in params:
            _clamp_prompts(params.get("n_prompts"))
        if kind == "benchmark_sweep":
            names = params.get("benchmarks", ["IOI"])
            if not isinstance(names, list) or not names:
                raise ValueError("benchmarks must be a non-empty list of names")
        if kind == "ioi_stability" and "n_seeds" in params:
            seeds = params.get("n_seeds")
            if isinstance(seeds, bool) or not isinstance(seeds, int) \
                    or not 1 <= seeds <= 10:
                raise ValueError("n_seeds must be an integer between 1 and 10")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        job = QUEUE.submit(kind, params)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"job": job, "provenance": "live"}


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    job = QUEUE.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id!r}")
    return {"job": job, "provenance": "live"}


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> Dict[str, Any]:
    """Request cancellation. A finished job is a 409, not a silent success."""
    try:
        job = QUEUE.cancel(job_id)
    except KeyError:
        raise HTTPException(status_code=404,
                            detail=f"unknown job {job_id!r}") from None
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"job": job, "provenance": "live"}
