"""Job event stream tests (§21.2).

These tests run against a *stubbed* handler registered into the real
``runtime_api.HANDLERS`` map, so the queue really transitions a job through
queued -> running -> terminal and the events are derived from that state. No
model weights and no downloads are involved.

The negative cases are the point: a job that has not reached a terminal state
has no success event, a failed job has no success event, an ``unavailable``
handler result is ``job.unavailable`` rather than ``job.failed``, and an
unknown job id is a 404 before any stream is opened.
"""

from __future__ import annotations

import asyncio
import atexit
import datetime as _dt
import json
import os
import shutil
import tempfile
import threading
import time
from typing import Any, Dict, Iterator, List

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Isolate the queue's on-disk records and pin it before anything else touches
# QUEUE._load(); otherwise the tests would read whatever the repository's own
# job directory happens to contain.
_JOB_DIR = tempfile.mkdtemp(prefix="mech-events-")
atexit.register(shutil.rmtree, _JOB_DIR, True)
os.environ["MECH_V2_JOB_DIR"] = _JOB_DIR

from backend.api import events as events_mod  # noqa: E402
from backend.api import runtime_api  # noqa: E402
from backend.api.events import ENVELOPE_FIELDS, EVENT_SCHEMA, EVENT_VERSION, HUB  # noqa: E402

AUTH_HEADERS = {"Authorization": "Bearer pytest-shared-token"}
TERMINAL = runtime_api.TERMINAL

#: Released by the tests to let the blocking handler finish.
_RELEASE = threading.Event()


def _noop(payload: Dict[str, Any], stop: threading.Event) -> Dict[str, Any]:
    """A handler that does no work. It is a test stub, not a measurement."""
    return {"status": "completed"}


def _blocks(payload: Dict[str, Any], stop: threading.Event) -> Dict[str, Any]:
    """Stays ``running`` until the test releases it."""
    _RELEASE.wait(30)
    return {"status": "cancelled" if stop.is_set() else "completed"}


def _raises(payload: Dict[str, Any], stop: threading.Event) -> Dict[str, Any]:
    raise ValueError("deliberate failure: handler raised")


def _unavailable(payload: Dict[str, Any], stop: threading.Event) -> Dict[str, Any]:
    return {"status": "unavailable", "reason": "no model loaded for this handler"}


_STUBS = {
    "_test_noop": _noop,
    "_test_blocks": _blocks,
    "_test_raises": _raises,
    "_test_unavailable": _unavailable,
}


@pytest.fixture(scope="module", autouse=True)
def stub_handlers() -> Iterator[None]:
    runtime_api.HANDLERS.update(_STUBS)
    try:
        yield
    finally:
        for name in _STUBS:
            runtime_api.HANDLERS.pop(name, None)
        _RELEASE.set()


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """The router on its own app: the mount prefix is the Lead's decision, and
    these tests must not depend on that edit having landed."""
    api = FastAPI()
    api.include_router(events_mod.router, prefix="/api/v1")
    with TestClient(api) as test_client:
        yield test_client


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _wait_for_terminal(job_id: str, timeout: float = 30.0) -> Dict[str, Any]:
    deadline = time.time() + timeout
    last: Dict[str, Any] = {}
    while time.time() < deadline:
        last = runtime_api.QUEUE.get(job_id) or {}
        if last.get("status") in TERMINAL:
            return last
        time.sleep(0.05)
    raise AssertionError(f"job {job_id!r} did not finish; last state {last!r}")


def _events_for(client: TestClient, job_id: str) -> List[Dict[str, Any]]:
    response = client.get("/api/v1/events", params={"job_id": job_id})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["event_version"] == EVENT_VERSION
    return body["events"]


def _frames(text: str) -> List[Dict[str, Any]]:
    """Parse an SSE body into ``{sse_event, id, data}`` frames."""
    frames: List[Dict[str, Any]] = []
    for block in text.split("\n\n"):
        block = block.strip("\n")
        if not block or block.startswith(":"):
            continue
        frame: Dict[str, Any] = {"sse_event": None, "id": None, "data": None}
        for line in block.splitlines():
            field, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if field == "event":
                frame["sse_event"] = value
            elif field == "id":
                frame["id"] = value
            elif field == "data":
                frame["data"] = json.loads(value)
        if frame["data"] is not None:
            frames.append(frame)
    return frames


def _wait_for_stream_teardown(timeout: float = 5.0) -> int:
    deadline = time.time() + timeout
    count = HUB.open_stream_count()
    while time.time() < deadline and count:
        time.sleep(0.02)
        count = HUB.open_stream_count()
    return count


# --------------------------------------------------------------------------- #
# envelope + schema
# --------------------------------------------------------------------------- #


def test_envelope_matches_declared_schema(client: TestClient) -> None:
    job = runtime_api.QUEUE.submit("_test_noop", {})
    _wait_for_terminal(job["job_id"])

    events = _events_for(client, job["job_id"])
    assert events, "a completed job must have at least one observed event"
    assert set(EVENT_SCHEMA["envelope"]["required"]) == set(ENVELOPE_FIELDS)

    seen_ids = set()
    for event in events:
        assert set(event) == set(ENVELOPE_FIELDS), sorted(set(event) ^ set(ENVELOPE_FIELDS))
        assert event["event_version"] == EVENT_VERSION == 1
        assert event["job_id"] == job["job_id"]
        assert event["type"] in EVENT_SCHEMA["types"]
        assert isinstance(event["sequence"], int) and event["sequence"] >= 1
        assert event["event_id"] not in seen_ids
        seen_ids.add(event["event_id"])
        assert event["stage"]
        assert event["message"]
        assert isinstance(event["data"], dict)
        # timestamp must be real ISO-8601 UTC, not a placeholder
        assert event["timestamp"].endswith("Z")
        parsed = _dt.datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
        assert parsed.tzinfo is not None
        assert abs((_dt.datetime.now(_dt.timezone.utc) - parsed).total_seconds()) < 300

    sequences = [event["sequence"] for event in events]
    assert sequences == sorted(sequences)
    assert len(set(sequences)) == len(sequences), "sequence must be strictly increasing"


def test_progress_is_null_when_unknown_and_never_zero(client: TestClient) -> None:
    job = runtime_api.QUEUE.submit("_test_noop", {})
    final = _wait_for_terminal(job["job_id"])
    assert final["status"] == "completed"

    events = _events_for(client, job["job_id"])
    assert events
    for event in events:
        # runtime_api reports no fraction, so every event must say so.
        assert event["progress"] is None, event
        assert event["progress"] != 0, "0 must never stand in for an unknown fraction"
    schema_progress = EVENT_SCHEMA["envelope"]["properties"]["progress"]
    assert "null" in schema_progress["type"]


def test_schema_endpoint_documents_the_envelope(client: TestClient) -> None:
    response = client.get("/api/v1/events/schema")
    assert response.status_code == 200
    body = response.json()
    assert body["event_version"] == EVENT_VERSION
    assert body["envelope_fields"] == list(ENVELOPE_FIELDS)
    assert body["schema"]["envelope"]["required"] == list(ENVELOPE_FIELDS)
    assert "job.succeeded" in body["schema"]["types"]
    assert body["schema"]["resume"]["header"] == "Last-Event-ID"
    assert body["provenance"] == "live"


# --------------------------------------------------------------------------- #
# events derive from observed state
# --------------------------------------------------------------------------- #


def test_events_are_derived_from_real_job_state(client: TestClient) -> None:
    job = runtime_api.QUEUE.submit("_test_noop", {"experiment_id": "exp-42"})
    job_id = job["job_id"]
    assert job["status"] == "queued"  # the queue's own admission state
    final = _wait_for_terminal(job_id)

    events = _events_for(client, job_id)
    last = events[-1]
    assert last["type"] == "job.succeeded"
    # Every field below is read from the queue record, not invented here.
    assert last["data"]["status"] == final["status"] == "completed"
    assert last["data"]["kind"] == final["kind"] == "_test_noop"
    assert last["data"]["created_at"] == final["created_at"]
    assert last["data"]["finished_at"] == final["finished_at"]
    assert last["experiment_id"] == "exp-42"


def test_no_success_event_before_the_job_finishes(client: TestClient) -> None:
    _RELEASE.clear()
    job = runtime_api.QUEUE.submit("_test_blocks", {})
    job_id = job["job_id"]
    try:
        # Wait until the queue is actually running the job, then observe.
        deadline = time.time() + 20
        events: List[Dict[str, Any]] = []
        while time.time() < deadline:
            state = runtime_api.QUEUE.get(job_id) or {}
            events = _events_for(client, job_id)
            if state.get("status") == "running" and events:
                break
            time.sleep(0.05)
        state = runtime_api.QUEUE.get(job_id) or {}
        assert state.get("status") not in TERMINAL, state
        events = _events_for(client, job_id)
        assert events, "an observed running job must have an event"
        types = {event["type"] for event in events}
        assert "job.succeeded" not in types
        assert types <= {"job.queued", "job.started", "job.progress"}
        assert all(event["data"]["status"] != "completed" for event in events)
    finally:
        _RELEASE.set()
    final = _wait_for_terminal(job_id)
    assert final["status"] == "completed"
    # Only now may a success event appear.
    assert "job.succeeded" in {event["type"] for event in _events_for(client, job_id)}


def test_failed_job_emits_failure_and_never_success(client: TestClient) -> None:
    job = runtime_api.QUEUE.submit("_test_raises", {})
    final = _wait_for_terminal(job["job_id"])
    assert final["status"] == "failed"
    assert "deliberate failure" in str(final["error"])

    events = _events_for(client, job["job_id"])
    types = [event["type"] for event in events]
    assert "job.failed" in types
    assert "job.succeeded" not in types
    assert "deliberate failure" in events[-1]["message"]
    assert events[-1]["data"]["status"] == "failed"
    assert events[-1]["progress"] is None


def test_unavailable_is_not_reported_as_failed(client: TestClient) -> None:
    job = runtime_api.QUEUE.submit("_test_unavailable", {})
    final = _wait_for_terminal(job["job_id"])
    assert final["status"] == "unavailable"

    events = _events_for(client, job["job_id"])
    types = [event["type"] for event in events]
    assert "job.unavailable" in types
    assert "job.failed" not in types, "NOT_EXECUTED must stay distinct from FAILED"
    assert "job.succeeded" not in types
    assert "no model loaded" in events[-1]["message"]
    assert events[-1]["data"]["status"] == "unavailable"


# --------------------------------------------------------------------------- #
# transport: streaming, resume, rejection, teardown
# --------------------------------------------------------------------------- #


def test_stream_is_sse_and_terminates_for_a_finished_job(client: TestClient) -> None:
    job = runtime_api.QUEUE.submit("_test_noop", {})
    _wait_for_terminal(job["job_id"])

    # A single-job stream closes once the job is terminal, so this is a plain
    # request rather than an open connection.
    response = client.get("/api/v1/events/stream", params={"job_id": job["job_id"]})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"

    frames = _frames(response.text)
    assert frames, response.text
    assert frames[-1]["data"]["type"] == "job.succeeded"
    for frame in frames:
        # SSE id is the envelope sequence; the event name is the type.
        assert frame["id"] == str(frame["data"]["sequence"])
        assert frame["sse_event"] == frame["data"]["type"]
        assert frame["data"]["progress"] is None


def _drain(iterator, max_frames: int = 10) -> List[Dict[str, Any]]:
    """Read SSE frames from a StreamingResponse body iterator, releasing the
    blocking handler as soon as the job is observed running."""

    async def read() -> List[Dict[str, Any]]:
        frames: List[Dict[str, Any]] = []
        try:
            async for chunk in iterator:
                parsed = _frames(chunk)
                if not parsed:
                    continue  # a keepalive comment, not an event
                frames.extend(parsed)
                if parsed[-1]["data"]["type"] == "job.started":
                    _RELEASE.set()
                if parsed[-1]["data"]["type"] == "job.succeeded":
                    break
                if len(frames) >= max_frames:
                    break
        finally:
            await iterator.aclose()
        return frames

    return asyncio.run(read())


def test_stream_reports_transitions_then_closes() -> None:
    """Drive the generator directly so the transition order is deterministic."""
    _RELEASE.clear()
    job = runtime_api.QUEUE.submit("_test_blocks", {})
    job_id = job["job_id"]
    watchdog = threading.Timer(25.0, _RELEASE.set)
    watchdog.start()
    try:
        response = events_mod.stream_events(job_id=job_id, since=None, last_event_id=None)
        frames = _drain(response.body_iterator)
    finally:
        _RELEASE.set()
        watchdog.cancel()

    types = [frame["data"]["type"] for frame in frames]
    sequences = [frame["data"]["sequence"] for frame in frames]
    assert "job.started" in types, types
    assert types[-1] == "job.succeeded", types
    assert types.index("job.started") < types.index("job.succeeded")
    assert sequences == sorted(sequences)
    assert len(set(sequences)) == len(sequences)
    for frame in frames:
        assert frame["id"] == str(frame["data"]["sequence"])
        assert frame["sse_event"] == frame["data"]["type"]
    assert _wait_for_stream_teardown() == 0, "generator must tear down on close"
    assert _wait_for_terminal(job_id)["status"] == "completed"


def test_resume_from_last_event_id_skips_delivered_events(client: TestClient) -> None:
    job = runtime_api.QUEUE.submit("_test_noop", {})
    _wait_for_terminal(job["job_id"])
    events = _events_for(client, job["job_id"])
    assert len(events) >= 1
    last_had = events[0]["sequence"]

    response = client.get(
        "/api/v1/events/stream",
        params={"job_id": job["job_id"]},
        headers={"Last-Event-ID": str(last_had)},
    )
    assert response.status_code == 200
    resumed = [frame["data"]["sequence"] for frame in _frames(response.text)]
    assert all(sequence > last_had for sequence in resumed), resumed
    assert resumed == [
        event["sequence"] for event in events if event["sequence"] > last_had
    ]

    # ?since= is the same cursor, for clients that cannot set a header.
    same = client.get(
        "/api/v1/events/stream",
        params={"job_id": job["job_id"], "since": last_had},
    )
    assert [frame["data"]["sequence"] for frame in _frames(same.text)] == resumed


def test_idle_stream_emits_keepalive_comment_not_an_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An idle stream must stay silent as events: only ``: keepalive`` comments."""
    monkeypatch.setenv("MECH_EVENT_KEEPALIVE_SECONDS", "0.5")
    monkeypatch.setenv("MECH_EVENT_POLL_SECONDS", "0.05")
    _RELEASE.clear()
    job = runtime_api.QUEUE.submit("_test_blocks", {})
    job_id = job["job_id"]
    watchdog = threading.Timer(10.0, _RELEASE.set)
    watchdog.start()

    async def read_until_keepalive() -> List[str]:
        response = events_mod.stream_events(
            job_id=job_id, since=None, last_event_id=None
        )
        chunks: List[str] = []
        try:
            async for chunk in response.body_iterator:
                chunks.append(chunk)
                if chunk.startswith(":"):
                    break
                if len(chunks) > 40:
                    break
        finally:
            await response.body_iterator.aclose()
        return chunks

    try:
        chunks = asyncio.run(read_until_keepalive())
    finally:
        _RELEASE.set()
        watchdog.cancel()

    keepalives = [chunk for chunk in chunks if chunk.startswith(":")]
    assert keepalives, "an idle stream must emit keepalive comments"
    assert all(chunk == ": keepalive\n\n" for chunk in keepalives)
    assert "progress" not in "".join(keepalives)
    for frame in _frames("".join(chunks)):
        # Whatever arrived as an event is still an observation, never a guess.
        assert frame["data"]["type"] in {"job.queued", "job.started", "job.progress"}
        assert frame["data"]["progress"] is None
    assert _wait_for_stream_teardown() == 0


def test_unknown_job_is_rejected_before_the_stream_opens(client: TestClient) -> None:
    response = client.get("/api/v1/events/stream", params={"job_id": "j-not-a-real-job"})
    assert response.status_code == 404
    assert not response.headers["content-type"].startswith("text/event-stream")
    detail = response.json()["detail"]
    assert "unknown job" in detail and "j-not-a-real-job" in detail
    # Nothing streamed means nothing to misread as event data.
    assert "job." not in response.text

    history = client.get("/api/v1/events", params={"job_id": "j-not-a-real-job"})
    assert history.status_code == 404
    assert "unknown job" in history.json()["detail"]


def test_retained_history_is_bounded(client: TestClient) -> None:
    body = client.get("/api/v1/events").json()
    assert body["bounded"]["retained"] > 0
    assert len(body["events"]) <= body["bounded"]["retained"]
    assert body["count"] == len(body["events"])


def _mounted_prefix(app: FastAPI) -> str:
    """Find the prefix this router is actually mounted under.

    Routes are probed over HTTP rather than by scanning ``app.routes``: the
    installed Starlette wraps ``include_router`` results in ``_IncludedRouter``
    objects that carry no ``path``, so scanning silently finds nothing -- which
    is exactly how this check used to turn into a skip.
    """
    probe = TestClient(app, raise_server_exceptions=False)
    for prefix in ("/api/v1", "/api"):
        if probe.get(f"{prefix}/events/schema").status_code != 404:
            return prefix
    raise AssertionError(
        "the events router is not reachable at /api/v1 or /api; "
        "backend/main.py must mount backend.api.events.router"
    )


def test_mounted_app_serves_events_and_rejects_unknown_jobs() -> None:
    """Exercise the router through the real app: bearer auth, unknown job 404s,
    and a schema-conformant envelope for a real submitted job."""
    from backend.main import app

    prefix = _mounted_prefix(app)
    real = TestClient(app, raise_server_exceptions=False)

    schema_path = f"{prefix}/events/schema"
    assert real.get(schema_path).status_code == 401, "events must need a bearer token"
    schema = real.get(schema_path, headers=AUTH_HEADERS)
    assert schema.status_code == 200
    assert schema.json()["envelope_fields"] == list(ENVELOPE_FIELDS)

    unknown = "j-not-a-real-job"
    history = real.get(f"{prefix}/events", params={"job_id": unknown}, headers=AUTH_HEADERS)
    assert history.status_code == 404
    assert unknown in history.json()["detail"] and "unknown job" in history.json()["detail"]

    stream = real.get(
        f"{prefix}/events/stream", params={"job_id": unknown}, headers=AUTH_HEADERS
    )
    assert stream.status_code == 404
    assert not stream.headers["content-type"].startswith("text/event-stream")
    assert "unknown job" in stream.json()["detail"]
    assert "job." not in stream.text

    job = runtime_api.QUEUE.submit("_test_noop", {})
    _wait_for_terminal(job["job_id"])
    body = real.get(
        f"{prefix}/events", params={"job_id": job["job_id"]}, headers=AUTH_HEADERS
    ).json()
    last = body["events"][-1]
    assert set(last) == set(ENVELOPE_FIELDS)
    assert last["event_version"] == EVENT_VERSION
    assert last["type"] == "job.succeeded"
    assert last["progress"] is None
    assert last["data"]["status"] == "completed"
