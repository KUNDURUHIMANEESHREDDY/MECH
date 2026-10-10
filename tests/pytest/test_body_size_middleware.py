"""F-02: Body-size middleware must enforce actual bytes and preserve readability.

The old middleware had two defects:

1. **Content-Length trust**: When `Content-Length` header was present and under
   the limit, the middleware trusted it and never counted actual bytes. A
   malicious client could send `Content-Length: 100` but stream 100 MB.

2. **Stream consumption**: When `Content-Length` was absent, the middleware
   consumed `request.stream()` to count bytes but did not preserve the body.
   Downstream (FastAPI, Pydantic) received an exhausted stream and failed to
   parse JSON.

This test verifies both are fixed:
- Actual bytes are always counted, regardless of headers
- Body is buffered and replayable by downstream
"""
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.main import app  # noqa: E402
from backend.core import auth as auth_mod  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(autouse=True)
def _set_env(monkeypatch):
    monkeypatch.setenv("MECH_API_TOKEN", "test-token")
    monkeypatch.setenv("MECH_MAX_BODY_BYTES", "1024")  # 1 KB cap for tests
    auth_mod.reset_cache()


def _post(client, path, body, headers=None, content_length=None):
    h = {"Authorization": "Bearer test-token", "Content-Type": "application/json"}
    if headers:
        h.update(headers)
    if content_length is not None:
        h["Content-Length"] = str(content_length)
    return client.post(path, content=body, headers=h)


# ── Honest client within limit ────────────────────────────────────────── #

def test_honest_client_within_limit():
    """Small body with correct Content-Length passes."""
    with TestClient(app) as client:
        body = b'{"key": "value"}'
        r = _post(client, "/api/v1/runtime/status", body, content_length=len(body))
    assert r.status_code == 200, r.text


def test_honest_client_exceeds_limit_via_header():
    """Large declared Content-Length rejected immediately."""
    with TestClient(app) as client:
        r = _post(client, "/api/v1/runtime/status", b"x", content_length=2000)
    assert r.status_code == 413


# ── Lying client: declared length small, actual stream large ──────────── #

def test_lying_content_length_rejected_by_actual_bytes():
    """Client says Content-Length: 10 but streams 2 KB -> rejected on actual bytes."""
    with TestClient(app) as client:
        # Declare 10 bytes but actually send 2048 bytes (over 1 KB cap)
        body = b"x" * 2048
        r = _post(client, "/api/v1/runtime/status", body, content_length=10)
    assert r.status_code == 413, f"expected 413, got {r.status_code}: {r.text}"
    assert "exceeds" in r.json()["detail"]


def test_chunked_transfer_no_content_length_still_checked():
    """No Content-Length header (chunked) -> actual bytes still enforced."""
    with TestClient(app) as client:
        body = b"x" * 2048
        r = _post(client, "/api/v1/runtime/status", body)  # no Content-Length
    assert r.status_code == 413


# ── Downstream readability: body must be re-readable by FastAPI/Pydantic ─ #

def test_body_buffered_and_replayed_for_downstream():
    """After middleware, downstream can still read and parse JSON."""
    with TestClient(app) as client:
        body = b'{"prompt": "hello world"}'
        # Use a real endpoint that parses JSON
        r = _post(client, "/api/v1/runtime/analyze_tokens", body)
    assert r.status_code == 200, f"downstream failed to parse body: {r.text}"
    assert r.json()["count"] == 2  # "hello", "world"


def test_body_buffered_works_for_both_small_and_medium():
    """Multiple requests in sequence all have readable bodies."""
    with TestClient(app) as client:
        for i in range(5):
            body = f'{{"prompt": "test {i}"}}'.encode()
            r = _post(client, "/api/v1/runtime/analyze_tokens", body)
            assert r.status_code == 200, f"request {i} failed: {r.text}"
            assert r.json()["count"] == 2


# ── Edge cases ────────────────────────────────────────────────────────── #

def test_malformed_content_length_rejected():
    with TestClient(app) as client:
        r = _post(client, "/api/v1/runtime/status", b"x", content_length="not-a-number")
    assert r.status_code == 400


def test_empty_body_allowed():
    with TestClient(app) as client:
        r = _post(client, "/api/v1/runtime/status", b"", content_length=0)
    assert r.status_code == 200


def test_exactly_at_cap_allowed():
    with TestClient(app) as client:
        body = b"x" * 1024  # exactly at 1 KB cap
        r = _post(client, "/api/v1/runtime/status", body)
    assert r.status_code == 200


def test_one_byte_over_cap_rejected():
    with TestClient(app) as client:
        body = b"x" * 1025  # 1 byte over
        r = _post(client, "/api/v1/runtime/status", body)
    assert r.status_code == 413


def test_understated_content_length_still_rejected():
    """Client declares Content-Length: 10 but actual is 2048.
    
    The middleware should read actual bytes and reject, not trust the header.
    """
    with TestClient(app) as client:
        body = b"x" * 2048
        r = _post(client, "/api/v1/runtime/status", body, content_length=10)
    assert r.status_code == 413


def test_overstated_content_length_declared_over_but_actual_under():
    """Client declares Content-Length: 2000 (over cap) but actual is small.
    
    Fast path rejects on declared length; actual body size doesn't matter
    because we reject before reading.
    """
    with TestClient(app) as client:
        body = b"small"
        r = _post(client, "/api/v1/runtime/status", body, content_length=2000)
    assert r.status_code == 413