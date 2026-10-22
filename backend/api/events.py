"""Structured job event stream (spec §21.2).

The events here are *observations* of ``backend/api/runtime_api.py``'s queue,
not a parallel source of truth. Every event is built by comparing a job's
current persisted state with the last state this process observed for it. Two
consequences worth stating outright:

* A job that never ran produces no ``job.succeeded`` event, ever. There is no
  code path that can emit one -- the type is selected from the observed
  ``status`` field and nothing else.
* ``progress`` is ``None`` unless the queue itself reports a real fraction in
  ``[0, 1]``. ``runtime_api`` does not report one today, so every event from a
  real job carries ``progress: None``. Zero is never used to mean "unknown".

Transport
---------
``GET /events/stream`` is Server-Sent Events in the same shape as
``backend/api/dispatcher.py:/society/stream``: ``id:``, ``event:`` and one
``data:`` JSON line per frame, ``: keepalive`` comments when idle.
``Last-Event-ID`` (or ``?since=``) resumes from a sequence number; the retained
ring buffer is bounded, and a client that has fallen off it recovers by reading
current state from ``GET /api/v2/jobs/{job_id}``. An unknown ``job_id`` is a
404 *before* the stream opens rather than an error frame, so a client never has
to distinguish "no such job" from a mid-stream failure.

Nothing here measures anything. Under ``/api/`` the app-level bearer auth in
``backend/core/auth.py`` applies; this module adds no public path.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Iterator, List, Optional, Tuple

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import StreamingResponse

from backend.api.runtime_api import QUEUE, TERMINAL

router = APIRouter()

#: Bumped only when the envelope changes shape. Clients pin this.
EVENT_VERSION = 1

#: The envelope, exactly as promised in the spec: these ten fields plus
#: ``sequence``. Exposed as a constant and as ``GET /events/schema`` so a client
#: can validate without reading this file.
ENVELOPE_FIELDS: Tuple[str, ...] = (
    "event_id",
    "event_version",
    "job_id",
    "experiment_id",
    "timestamp",
    "type",
    "stage",
    "message",
    "progress",
    "data",
    "sequence",
)

#: ``status`` -> event ``type``. A status with no entry emits ``job.status``
#: and says so in ``stage``; it is never mapped onto a success.
STATUS_EVENT_TYPES: Dict[str, str] = {
    "queued": "job.queued",
    "running": "job.started",
    "completed": "job.succeeded",
    "failed": "job.failed",
    "cancelled": "job.cancelled",
    "unavailable": "job.unavailable",
    "interrupted": "job.interrupted",
}

#: Types that end a single-job stream.
TERMINAL_EVENT_TYPES = frozenset(
    STATUS_EVENT_TYPES[s] for s in TERMINAL if s in STATUS_EVENT_TYPES
)

EVENT_SCHEMA: Dict[str, Any] = {
    "event_version": EVENT_VERSION,
    "envelope": {
        "type": "object",
        "additionalProperties": False,
        "required": list(ENVELOPE_FIELDS),
        "properties": {
            "event_id": {"type": "string"},
            "event_version": {"type": "integer", "const": EVENT_VERSION},
            "job_id": {"type": "string"},
            "experiment_id": {"type": ["string", "null"]},
            "timestamp": {"type": "string", "format": "date-time"},
            "type": {"type": "string"},
            "stage": {"type": "string"},
            "message": {"type": "string"},
            "progress": {
                "type": ["number", "null"],
                "minimum": 0,
                "maximum": 1,
                "description": (
                    "Fraction of work observed from the queue, or null when no "
                    "fraction exists. Never 0 to mean unknown."
                ),
            },
            "data": {"type": "object"},
            "sequence": {"type": "integer", "minimum": 0},
        },
    },
    "types": sorted(set(STATUS_EVENT_TYPES.values())) + ["job.progress", "job.status"],
    "resume": {
        "header": "Last-Event-ID",
        "query": "since",
        "note": (
            "Both carry the `sequence` of the last event seen. Sequences are "
            "process-wide and strictly increasing; a client that has dropped off "
            "the retained buffer should fall back to GET /api/v2/jobs/{job_id}."
        ),
    },
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _float_env(name: str, default: float, low: float, high: float) -> float:
    raw = os.environ.get(name, "")
    try:
        value = float(str(raw).strip()) if str(raw).strip() else default
    except (TypeError, ValueError):
        return default
    return max(low, min(value, high))


def _poll_interval() -> float:
    return _float_env("MECH_EVENT_POLL_SECONDS", 0.25, 0.05, 5.0)


def _keepalive_seconds() -> float:
    return _float_env("MECH_EVENT_KEEPALIVE_SECONDS", 15.0, 0.5, 120.0)


def _max_stream_seconds() -> float:
    return _float_env("MECH_EVENT_STREAM_MAX_SECONDS", 300.0, 1.0, 3600.0)


def _utc_now() -> str:
    """ISO-8601 UTC with a literal ``Z``; second precision is enough here."""
    return (
        _dt.datetime.now(_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _observed_progress(job: Dict[str, Any]) -> Optional[float]:
    """A real fraction from the queue, or ``None``. Never synthesised."""
    result = job.get("result")
    for source in (job, result if isinstance(result, dict) else None):
        if not isinstance(source, dict):
            continue
        value = source.get("progress")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        fraction = float(value)
        if 0.0 <= fraction <= 1.0:
            return fraction
    return None


_RESULT_KEYS = (
    "status",
    "provenance",
    "reason",
    "error",
    "measured",
    "unavailable",
    "n_prompts",
)


def _data(job: Dict[str, Any]) -> Dict[str, Any]:
    """The part of the job record an event is allowed to repeat.

    Deliberately excludes ``payload``: an event stream is not a place to
    echo whatever a caller sent, and the queue's record is the durable copy.
    """
    data: Dict[str, Any] = {
        "kind": job.get("kind"),
        "status": job.get("status"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "cancelled": bool(job.get("cancel_requested")),
    }
    error = job.get("error")
    if error:
        data["error"] = str(error)[:500]
    persist_error = job.get("persist_error")
    if persist_error:
        data["persist_error"] = str(persist_error)[:300]
    result = job.get("result")
    if isinstance(result, dict):
        data["result"] = {k: result[k] for k in _RESULT_KEYS if k in result}
    return data


def _experiment_id(job: Dict[str, Any]) -> Optional[str]:
    for source in (job, job.get("payload"), job.get("result")):
        if isinstance(source, dict):
            value = source.get("experiment_id")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _partial_failure(job: Dict[str, Any]) -> str:
    """Describe a run that finished with part of its work missing.

    A sweep that measured some items and could not measure others is not a
    clean success, and the event must not read like one. The status itself
    is not rewritten here -- the vocabulary is a contract -- but the message
    carries the reason, so a client that shows the message shows the truth.
    """
    result = job.get("result")
    if not isinstance(result, dict):
        return ""
    status = str(result.get("status") or "")
    if status == "completed_with_unavailable":
        return ("part of the work was unavailable: %s measured, %s unavailable"
                % (result.get("measured", "?"), result.get("unavailable", "?")))
    return ""


def _signature(job: Dict[str, Any]) -> Tuple[Any, ...]:
    """What "the state changed" means. Nothing here is a guess."""
    return (
        str(job.get("status") or ""),
        job.get("started_at"),
        job.get("finished_at"),
        job.get("error"),
        _observed_progress(job),
    )


# --------------------------------------------------------------------------- #
# the hub: one observer, one sequence, one bounded buffer
# --------------------------------------------------------------------------- #


class EventHub:
    """Serialises queue observations into a single monotonic event sequence.

    Every streaming client reads the same buffer, so a transition is recorded
    once no matter how many clients are watching, and ``sequence`` never goes
    backwards across reconnects.
    """

    def __init__(self, retention: int = 1000, max_tracked_jobs: int = 5000) -> None:
        self._lock = threading.Lock()
        self._history: Deque[Dict[str, Any]] = deque(maxlen=retention)
        self._seen: Dict[str, Tuple[Any, ...]] = {}
        self._max_tracked = max_tracked_jobs
        self._next_sequence = 1
        self._open_streams = 0

    # -- recording ------------------------------------------------------ #

    def observe(self, job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Record one transition if ``job`` differs from the last observation."""
        job_id = str(job.get("job_id") or "")
        if not job_id:
            return None
        signature = _signature(job)
        with self._lock:
            if self._seen.get(job_id) == signature:
                return None
            previous = self._seen.get(job_id)
            self._seen[job_id] = signature
            if len(self._seen) > self._max_tracked:
                # Drop the oldest tracked job; its record stays on disk, so a
                # reconnect re-observes it as a fresh state rather than a gap.
                self._seen.pop(next(iter(self._seen)))
            status = str(job.get("status") or "")
            event_type = STATUS_EVENT_TYPES.get(status, "job.status")
            if (
                previous is not None
                and previous[:4] == signature[:4]
                and signature[4] is not None
            ):
                event_type = "job.progress"
            sequence = self._next_sequence
            self._next_sequence += 1
            error = str(job.get("error") or "").strip()
            message = f"job {status or 'unknown'}"
            if error:
                message = f"{message}: {error[:400]}"
            else:
                partial = _partial_failure(job)
                if partial:
                    message = f"{message}: {partial}"
            event = {
                "event_id": f"evt_{sequence:012d}_{job_id}",
                "event_version": EVENT_VERSION,
                "job_id": job_id,
                "experiment_id": _experiment_id(job),
                "timestamp": _utc_now(),
                "type": event_type,
                "stage": str(job.get("stage") or status or "unknown"),
                "message": message,
                "progress": _observed_progress(job),
                "data": _data(job),
                "sequence": sequence,
            }
            self._history.append(event)
            return event

    # -- reading -------------------------------------------------------- #

    def sync(self, job_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Observe current queue state; return the events that produced."""
        if job_id:
            job = QUEUE.get(job_id)
            jobs = [job] if job is not None else []
        else:
            jobs = QUEUE.listing(limit=200)
        events: List[Dict[str, Any]] = []
        for job in jobs:
            event = self.observe(job)
            if event is not None:
                events.append(event)
        return events

    def snapshot(
        self, job_id: Optional[str] = None, after: int = 0, limit: int = 500
    ) -> List[Dict[str, Any]]:
        """Retained events with ``sequence > after``, oldest first."""
        with self._lock:
            events = [
                dict(event)
                for event in self._history
                if event["sequence"] > after
                and (job_id is None or event["job_id"] == job_id)
            ]
        return events[-limit:]

    def latest_sequence(self) -> int:
        with self._lock:
            return self._next_sequence - 1

    @property
    def retained_capacity(self) -> int:
        return self._history.maxlen or 0

    # -- stream bookkeeping (observable, so teardown can be asserted) --- #

    def open_stream(self) -> int:
        with self._lock:
            self._open_streams += 1
            return self._open_streams

    def close_stream(self) -> int:
        with self._lock:
            self._open_streams = max(0, self._open_streams - 1)
            return self._open_streams

    def open_stream_count(self) -> int:
        with self._lock:
            return self._open_streams


HUB = EventHub()


# --------------------------------------------------------------------------- #
# frames
# --------------------------------------------------------------------------- #


def sse_frame(event: Dict[str, Any]) -> str:
    payload = json.dumps(event, ensure_ascii=False, default=str)
    return (
        f"id: {event['sequence']}\n"
        f"event: {event['type']}\n"
        f"data: {payload}\n\n"
    )


#: Written when nothing has happened. A comment is not an event, so an idle
#: stream cannot be mistaken for a fabricated progress report.
KEEPALIVE_FRAME = ": keepalive\n\n"


def _resume_point(since: Optional[int], last_event_id: Optional[str]) -> int:
    """The sequence the client already has; streaming continues after it."""
    if since is not None:
        return max(0, int(since))
    if last_event_id:
        try:
            return max(0, int(str(last_event_id).strip()))
        except (TypeError, ValueError):
            # A non-numeric id cannot be a sequence this server issued. Treat it
            # as "start from the beginning" rather than silently skipping.
            return 0
    return 0


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #


@router.get("/events/schema")
def events_schema() -> Dict[str, Any]:
    """The envelope a client will receive, and how to resume it."""
    return {
        "event_version": EVENT_VERSION,
        "envelope_fields": list(ENVELOPE_FIELDS),
        "schema": EVENT_SCHEMA,
        "retained_history": HUB.retained_capacity,
        "latest_sequence": HUB.latest_sequence(),
        "provenance": "live",
    }


@router.get("/events")
def events(
    job_id: Optional[str] = Query(default=None),
    after: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
) -> Dict[str, Any]:
    """Retained events, oldest first. Observes the queue first, so a caller
    asking right after a transition does not miss it."""
    if job_id and QUEUE.get(job_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id!r}")
    HUB.sync(job_id)
    events_list = HUB.snapshot(job_id, after=after, limit=limit)
    return {
        "events": events_list,
        "count": len(events_list),
        "event_version": EVENT_VERSION,
        "latest_sequence": HUB.latest_sequence(),
        "bounded": {"retained": HUB.retained_capacity, "returned": len(events_list)},
        "provenance": "live",
    }


@router.get("/events/stream")
def stream_events(
    job_id: Optional[str] = Query(default=None),
    since: Optional[int] = Query(default=None, ge=0),
    last_event_id: Optional[str] = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    """Tail observed job transitions as Server-Sent Events.

    Rejects an unknown ``job_id`` before the stream opens: a 404 with a reason
    is something a client can act on, an error frame in a 200 body is not.
    """
    if job_id and QUEUE.get(job_id) is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"unknown job {job_id!r}: no events exist for it and it cannot "
                "be observed"
            ),
        )
    start = _resume_point(since, last_event_id)

    def generate() -> Iterator[str]:
        HUB.open_stream()
        try:
            last_sent = start
            for event in HUB.snapshot(job_id, after=start):
                last_sent = event["sequence"]
                yield sse_frame(event)
            deadline = time.monotonic() + _max_stream_seconds()
            idle_since = time.monotonic()
            while True:
                produced = [
                    event
                    for event in HUB.sync(job_id)
                    if event["sequence"] > last_sent
                ]
                now = time.monotonic()
                for event in produced:
                    last_sent = event["sequence"]
                    idle_since = now
                    yield sse_frame(event)
                if job_id:
                    # Every terminal signature has an event by now: `observe`
                    # records one the first time it sees it. Draining is done,
                    # so close instead of holding the connection open.
                    job = QUEUE.get(job_id)
                    if job is None or job.get("status") in TERMINAL:
                        # One last observation before closing. The status can
                        # flip between the sync above and this check, and a
                        # client told "this stream ended for a finished job"
                        # without ever receiving the terminal event has no way
                        # to recover it from the stream -- the contract is that
                        # every terminal signature arrives as an event.
                        for event in HUB.sync(job_id):
                            if event["sequence"] > last_sent:
                                last_sent = event["sequence"]
                                yield sse_frame(event)
                        return
                if now > deadline:
                    return
                if not produced and now - idle_since >= _keepalive_seconds():
                    idle_since = now
                    yield KEEPALIVE_FRAME
                time.sleep(_poll_interval())
        finally:
            # Runs on normal completion, generator close and client drop. A
            # dropped client surfaces as GeneratorExit on the next write, which
            # the keepalive bounds to one interval, so there is no orphan poll.
            HUB.close_stream()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
