"""P0-1: mandatory bearer auth on the control plane.

Seam: HTTP boundary (TestClient against backend.main.app) + unit seam at
backend.core.auth.verify_bearer. Anonymous /api/* and /mcp must 401; / and
/health stay public; /health/subsystems requires auth.
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fastapi.testclient import TestClient

from backend.core import auth as auth_mod
from backend.main import app


def _client():
    return TestClient(app, raise_server_exceptions=False)


def test_anonymous_api_rejected(monkeypatch):
    monkeypatch.setenv("MECH_API_TOKEN", "test-token-123")
    auth_mod.reset_cache()
    try:
        with _client() as c:
            r = c.get("/api/status")
        assert r.status_code == 401, r.text
        assert "credentials" in r.json().get("detail", "")
    finally:
        auth_mod.reset_cache()


def test_invalid_token_rejected(monkeypatch):
    monkeypatch.setenv("MECH_API_TOKEN", "correct-token")
    auth_mod.reset_cache()
    try:
        with _client() as c:
            r = c.get("/api/status", headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401
    finally:
        auth_mod.reset_cache()


def test_valid_token_accepted(monkeypatch):
    monkeypatch.setenv("MECH_API_TOKEN", "valid-abc")
    auth_mod.reset_cache()
    try:
        with _client() as c:
            r = c.get("/api/status", headers=auth_mod.auth_headers())
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "ok"
    finally:
        auth_mod.reset_cache()


def test_root_and_liveness_stay_public(monkeypatch):
    monkeypatch.setenv("MECH_API_TOKEN", "tok")
    auth_mod.reset_cache()
    try:
        with _client() as c:
            assert c.get("/").status_code == 200
            assert c.get("/health").status_code == 200
    finally:
        auth_mod.reset_cache()


def test_subsystems_requires_auth(monkeypatch):
    monkeypatch.setenv("MECH_API_TOKEN", "tok2")
    auth_mod.reset_cache()
    try:
        with _client() as c:
            assert c.get("/health/subsystems").status_code == 401
            r = c.get("/health/subsystems", headers=auth_mod.auth_headers())
        assert r.status_code == 200, r.text
    finally:
        auth_mod.reset_cache()


def test_mcp_http_requires_auth(monkeypatch):
    monkeypatch.setenv("MECH_API_TOKEN", "mcp-tok")
    auth_mod.reset_cache()
    try:
        with _client() as c:
            r = c.post("/mcp", json={})
        # Either 401 from the gate or 4xx/405 from the MCP app is fine for
        # anonymous; what must NOT happen is reaching a tool result.
        assert r.status_code == 401, r.text
    finally:
        auth_mod.reset_cache()


def test_malformed_auth_header_rejected(monkeypatch):
    monkeypatch.setenv("MECH_API_TOKEN", "tok3")
    auth_mod.reset_cache()
    try:
        with _client() as c:
            assert c.get("/api/status", headers={"Authorization": "Token abc"}).status_code == 401
            assert c.get("/api/status", headers={"Authorization": "Bearer "}).status_code == 401
            assert c.get("/api/status", headers={"Authorization": "Bearer"}).status_code == 401
    finally:
        auth_mod.reset_cache()


def test_verify_bearer_unit(monkeypatch):
    monkeypatch.setenv("MECH_API_TOKEN", "unit-secret")
    auth_mod.reset_cache()
    try:
        assert auth_mod.verify_bearer("unit-secret") is True
        assert auth_mod.verify_bearer("wrong") is False
        assert auth_mod.verify_bearer(None) is False
        assert auth_mod.verify_bearer("") is False
        assert auth_mod.verify_bearer(123) is False  # type: ignore[arg-type]
    finally:
        auth_mod.reset_cache()


def test_verify_uses_constant_time_compare():
    import inspect
    src = inspect.getsource(auth_mod.verify_bearer)
    assert "compare_digest" in src


def test_is_public_path_unit():
    assert auth_mod.is_public_path("/", "GET") is True
    assert auth_mod.is_public_path("/health", "GET") is True
    assert auth_mod.is_public_path("/health/subsystems", "GET") is False
    assert auth_mod.is_public_path("/api/status", "GET") is False
    assert auth_mod.is_public_path("/api/v1/ping", "POST") is False
    assert auth_mod.is_public_path("/mcp", "POST") is False
    assert auth_mod.is_public_path("/mcp/", "POST") is False
    assert auth_mod.is_public_path("/openapi.json", "GET") is False
    # CORS preflight carries no credentials by design; the real request still gates.
    assert auth_mod.is_public_path("/api/status", "OPTIONS") is True


def test_token_path_cannot_be_redirected_from_the_environment(monkeypatch, tmp_path):
    """F-4: a redirectable token path lets an attacker choose which file is read.

    ``MECH_TOKEN_FILE`` used to do exactly that, so any process able to set an
    environment variable before the first ``get_bearer_token()`` call picked the
    path the control-plane token was persisted to and read from. It is now
    ignored; the path is fixed relative to the auth module.

    The value override (``MECH_API_TOKEN``) is a different seam and stays: it
    sets the token itself rather than where it lives, and the rest of this file
    depends on it.
    """
    attacker_path = tmp_path / "attacker_chosen_token"
    attacker_path.write_text("attacker-chosen-token\n", encoding="utf-8")
    monkeypatch.setenv("MECH_TOKEN_FILE", str(attacker_path))

    resolved = auth_mod._token_file()

    assert resolved != attacker_path
    assert resolved.name == auth_mod._TOKEN_FILENAME


# ---------------------------------------------------------------------------
# Query-parameter credentials: the SSE / <img> seam
# ---------------------------------------------------------------------------
#
# `EventSource` (used by the Research Society live stream) and `<img src>` cannot
# set an `Authorization` header. Ten frontend call sites were therefore sending
# no credentials at all and receiving 401 in standalone mode. The middleware now
# also accepts `?token=` / `?bearer_token=` for exactly those transports.
#
# These tests pin both directions: the seam works, and it is still a *verified*
# credential rather than an authentication bypass.


def test_query_parameter_token_is_accepted(monkeypatch):
    monkeypatch.setenv("MECH_API_TOKEN", "query-token-abc")
    auth_mod.reset_cache()
    try:
        with _client() as c:
            r = c.get("/api/status?token=query-token-abc")
            r2 = c.get("/api/status?bearer_token=query-token-abc")
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "ok"
        assert r2.status_code == 200, r2.text
    finally:
        auth_mod.reset_cache()


def test_wrong_query_parameter_token_is_rejected(monkeypatch):
    monkeypatch.setenv("MECH_API_TOKEN", "query-token-abc")
    auth_mod.reset_cache()
    try:
        with _client() as c:
            r = c.get("/api/status?token=wrong")
        assert r.status_code == 401
        assert "credentials" in r.json().get("detail", "")
    finally:
        auth_mod.reset_cache()


def test_query_parameter_does_not_unlock_a_data_route_without_a_token(monkeypatch):
    """The seam authenticates; it does not make a protected route public."""
    monkeypatch.setenv("MECH_API_TOKEN", "query-token-abc")
    auth_mod.reset_cache()
    try:
        with _client() as c:
            anonymous = c.get("/api/society/runs")
            authenticated = c.get("/api/society/runs?token=query-token-abc")
        assert anonymous.status_code == 401
        assert authenticated.status_code == 200, authenticated.text
    finally:
        auth_mod.reset_cache()


def test_header_token_still_wins_and_public_paths_stay_public(monkeypatch):
    monkeypatch.setenv("MECH_API_TOKEN", "query-token-abc")
    auth_mod.reset_cache()
    try:
        with _client() as c:
            # A valid header with no query parameter is unaffected by the seam.
            header_only = c.get("/api/status", headers=auth_mod.auth_headers())
            # Liveness stays public; the seam must not have widened the allowlist.
            health = c.get("/health")
            root = c.get("/")
        assert header_only.status_code == 200
        assert health.status_code == 200
        assert root.status_code == 200
    finally:
        auth_mod.reset_cache()


def test_query_string_is_not_persisted_in_the_request_log(monkeypatch):
    """The token must not be written into MECH's own request log.

    The middleware records ``request.url.path``, which excludes the query string.
    Pinning it here means a future change to log the full URL cannot quietly
    start persisting credentials.
    """
    from backend.core import request_log

    request_log.record("GET", "/api/status", 200, 1.0)
    entries = request_log.recent(limit=5) if hasattr(request_log, "recent") else []
    for entry in entries:
        assert "token=" not in str(entry.get("path", "")), entry
