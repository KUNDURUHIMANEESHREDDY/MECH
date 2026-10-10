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
