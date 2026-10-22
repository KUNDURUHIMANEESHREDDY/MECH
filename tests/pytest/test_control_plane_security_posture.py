"""The control plane's credential and CORS posture.

Four defects, each pinned here so it cannot come back silently:

* **M1 -- the session cookie is an ambient credential.** A "site" is
  (scheme, registrable domain); the PORT IS NOT PART OF IT (RFC 6265bis
  5.2.3). A page served from ``http://localhost:3000`` is therefore
  same-site with ``http://localhost:8000`` and the browser attaches the
  ``SameSite=Strict`` cookie to a credentialed request from it. CORS
  decides whether the response is readable; it does not decide whether the
  cookie is sent. The server checks the Origin of a cookie-authenticated
  request itself now.

* **M2 -- a bearer token in the query string reaches the access log.**
  uvicorn writes the full query string, so ``?token=`` put the credential
  in stdout in cleartext. The seam is opt-in now.

* **L1 -- ``MECH_CORS_ORIGINS`` widened the allowlist to any remote
  origin, with credentials.** Entries are validated, and an added origin
  no longer gets credentialed cross-origin access.

* **L4 -- the body-size ceiling sat outside the auth gate**, so an
  anonymous caller could make the server buffer megabytes, and a 401 was
  produced before the CORS middleware ran, so a browser client saw an
  opaque CORS failure instead of a readable refusal.

The negative tests here pass because the system correctly REFUSES the
request. A test that "the attack still works" would be a failure.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

TOKEN = "posture-token-4c1a"


@pytest.fixture()
def client(monkeypatch):
    """A client with no credentials of its own."""
    monkeypatch.setenv("MECH_API_TOKEN", TOKEN)
    from backend.core import auth as auth_mod
    auth_mod.reset_cache()
    from backend.main import app
    yield TestClient(app)
    auth_mod.reset_cache()


def _mint_cookie(client) -> str:
    """Exchange the bearer token for a session cookie."""
    response = client.post("/session", headers={"Authorization": f"Bearer {TOKEN}"})
    assert response.status_code == 200, response.text
    cookie = response.cookies.get("mech_session")
    assert cookie, f"no session cookie in {dict(response.cookies)}"
    return cookie


def _cookie_client(client) -> TestClient:
    """The same client, now holding a session cookie."""
    client.cookies.set("mech_session", _mint_cookie(client))
    return client


# --------------------------------------------------------------------------- #
# M1: the server checks the Origin of a cookie-authenticated request
# --------------------------------------------------------------------------- #


def test_cookie_request_from_another_loopback_port_is_refused(client):
    """The loopback port is not an identity (M1).

    Before the fix this returned 200: the browser attaches the
    SameSite=Strict cookie because the request is same-site, and the CORS
    allowlist accepted every loopback port.
    """
    _cookie_client(client)
    response = client.get("/api/status", headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 403, response.text
    assert "origin" in response.json().get("detail", "").lower()


def test_cookie_request_from_a_remote_origin_is_refused(client):
    _cookie_client(client)
    response = client.get("/api/status",
                          headers={"Origin": "https://evil.example.com"})
    assert response.status_code == 403, response.text


def test_cookie_request_from_the_opaque_null_origin_is_refused(client):
    """"null" is what a file:// page, a sandboxed iframe and a data: URL
    all report, so it identifies no page the server can trust."""
    _cookie_client(client)
    response = client.get("/api/status", headers={"Origin": "null"})
    assert response.status_code == 403, response.text


def test_cookie_request_from_the_servers_own_origin_is_allowed(client):
    """The standalone build serves its own frontend: same host and port."""
    _cookie_client(client)
    response = client.get("/api/status", headers={
        "Host": "127.0.0.1:8000", "Origin": "http://127.0.0.1:8000"})
    assert response.status_code == 200, response.text


def test_cookie_request_without_an_origin_is_allowed(client):
    """A browser omits Origin on a same-origin GET; that flow must keep working."""
    _cookie_client(client)
    response = client.get("/api/status")
    assert response.status_code == 200, response.text


def test_trusted_origin_opt_in_allows_an_explicit_origin(client, monkeypatch):
    """MECH_TRUSTED_ORIGINS is the deliberate way to name an origin."""
    monkeypatch.setenv("MECH_TRUSTED_ORIGINS", "http://localhost:3000")
    _cookie_client(client)
    response = client.get("/api/status", headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 200, response.text


def test_loopback_wildcard_is_opt_in(client, monkeypatch):
    """Any loopback port is an opt-in, never the default (M1)."""
    _cookie_client(client)
    refused = client.get("/api/status", headers={"Origin": "http://localhost:3000"})
    assert refused.status_code == 403, refused.text

    monkeypatch.setenv("MECH_TRUST_LOOPBACK_ORIGINS", "1")
    allowed = client.get("/api/status", headers={"Origin": "http://localhost:3000"})
    assert allowed.status_code == 200, allowed.text


def test_the_bearer_token_is_not_subject_to_the_origin_check(client):
    """An explicit credential is not ambient: the desktop app sends one."""
    response = client.get("/api/status", headers={
        "Authorization": f"Bearer {TOKEN}", "Origin": "http://localhost:3000"})
    assert response.status_code == 200, response.text


# --------------------------------------------------------------------------- #
# M2: the query-string credential is opt-in
# --------------------------------------------------------------------------- #


def test_query_parameter_token_is_refused_by_default(client):
    """The credential must not reach the access log (M2)."""
    response = client.get("/api/status?token=" + TOKEN)
    assert response.status_code == 401, response.text
    assert "credentials" in response.json().get("detail", "")


def test_bearer_query_parameter_is_refused_by_default(client):
    response = client.get("/api/status?bearer_token=" + TOKEN)
    assert response.status_code == 401, response.text


def test_session_route_refuses_a_query_parameter_token(client):
    response = client.post("/session?token=" + TOKEN)
    assert response.status_code == 401, response.text


def test_query_parameter_token_is_allowed_under_the_explicit_opt_in(client, monkeypatch):
    """EventSource and <img> cannot set a header; that seam stays reachable,
    but only when the operator says so."""
    monkeypatch.setenv("MECH_ALLOW_QUERY_TOKEN", "1")
    response = client.get("/api/status?token=" + TOKEN)
    assert response.status_code == 200, response.text


def test_a_wrong_query_parameter_token_is_still_refused(client, monkeypatch):
    monkeypatch.setenv("MECH_ALLOW_QUERY_TOKEN", "1")
    response = client.get("/api/status?token=wrong")
    assert response.status_code == 401, response.text


# --------------------------------------------------------------------------- #
# L1: the CORS allowlist and the credential policy
# --------------------------------------------------------------------------- #


def test_cors_allowlist_entries_must_be_well_formed_origins(monkeypatch):
    """An entry that is not an origin is refused, not allowed by accident."""
    import backend.main as backend_main
    monkeypatch.setenv("MECH_CORS_ORIGINS",
                        "not-an-origin, https://ok.example.com/path, *, "
                        "http://user:pw@evil.example.com, ftp://ok.example.com")
    try:
        origins = backend_main._cors_origins()
    finally:
        monkeypatch.delenv("MECH_CORS_ORIGINS", raising=False)
    assert origins == ["null"], origins


def test_cors_allowlist_entry_is_normalized(monkeypatch):
    import backend.main as backend_main
    monkeypatch.setenv("MECH_CORS_ORIGINS", "http://dev.local:9999/")
    try:
        origins = backend_main._cors_origins()
    finally:
        monkeypatch.delenv("MECH_CORS_ORIGINS", raising=False)
    assert origins == ["null", "http://dev.local:9999"], origins


def test_an_added_remote_origin_does_not_carry_credentials(monkeypatch):
    """Adding an origin must not switch on credentialed cross-origin access."""
    import backend.main as backend_main
    monkeypatch.setenv("MECH_CORS_ORIGINS", "https://evil.example.com")
    try:
        reloaded = importlib.reload(backend_main)
        with TestClient(reloaded.app) as client:
            response = client.get("/health",
                                  headers={"Origin": "https://evil.example.com"})
            assert response.headers.get("access-control-allow-origin") == \
                "https://evil.example.com"
            assert response.headers.get("access-control-allow-credentials") is None, (
                "a remote origin added to the allowlist must not receive "
                "Access-Control-Allow-Credentials"
            )
    finally:
        monkeypatch.delenv("MECH_CORS_ORIGINS", raising=False)
        importlib.reload(backend_main)


def test_the_opaque_null_origin_does_not_carry_credentials(client):
    """"null" stays in the allowlist (the packaged desktop app loads from
    file://) but it never carries the operator's credentials -- exactly like
    the session gate, which refuses it."""
    response = client.get("/health", headers={"Origin": "null"})
    assert response.headers.get("access-control-allow-origin") == "null"
    assert response.headers.get("access-control-allow-credentials") is None


def test_a_loopback_origin_keeps_its_credentials(client):
    """The loopback dev flow is unchanged: ACAO and ACAC both present."""
    response = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert response.headers.get("access-control-allow-credentials") == "true"


def test_a_trusted_remote_origin_may_carry_credentials(monkeypatch):
    """The explicit opt-in: name the origin in MECH_TRUSTED_ORIGINS."""
    import backend.main as backend_main
    monkeypatch.setenv("MECH_CORS_ORIGINS", "https://tools.example.com")
    monkeypatch.setenv("MECH_TRUSTED_ORIGINS", "https://tools.example.com")
    try:
        reloaded = importlib.reload(backend_main)
        with TestClient(reloaded.app) as client:
            response = client.get("/health",
                                  headers={"Origin": "https://tools.example.com"})
            assert response.headers.get("access-control-allow-origin") == \
                "https://tools.example.com"
            assert response.headers.get("access-control-allow-credentials") == "true"
    finally:
        monkeypatch.delenv("MECH_CORS_ORIGINS", raising=False)
        monkeypatch.delenv("MECH_TRUSTED_ORIGINS", raising=False)
        importlib.reload(backend_main)


# --------------------------------------------------------------------------- #
# L4: the middleware stack
# --------------------------------------------------------------------------- #


def _middleware_names(app) -> list:
    names = []
    for entry in app.user_middleware:
        dispatch = entry.kwargs.get("dispatch")
        names.append(getattr(dispatch, "__name__", None)
                     or getattr(entry.cls, "__name__", ""))
    return names


def test_the_middleware_stack_matches_the_declared_order():
    """The effective order is asserted, not inferred from registration.

    Starlette's add_middleware inserts at position 0, so the LAST
    registration is the OUTERMOST middleware: adding a middleware above
    would silently move the auth gate behind the body ceiling again.
    """
    from backend.main import _MIDDLEWARE_ORDER, app
    assert _middleware_names(app) == list(_MIDDLEWARE_ORDER), _middleware_names(app)


def test_cors_is_outside_the_auth_gate_and_the_body_ceiling_is_inside_it():
    from backend.main import app
    names = _middleware_names(app)
    assert names.index("CORSMiddleware") < names.index("_auth_middleware")
    assert names.index("_auth_middleware") < names.index("_body_size_middleware")


def test_an_anonymous_oversized_post_is_refused_before_the_body_is_read(client, monkeypatch):
    """The ceiling must not cost an anonymous caller a buffered megabyte."""
    monkeypatch.setenv("MECH_MAX_BODY_BYTES", "1024")
    response = client.post("/api/ping", content=b"x" * 4096,
                           headers={"Content-Type": "application/json"})
    assert response.status_code == 401, response.text
    assert response.json().get("detail") == "missing credentials"


def test_an_authenticated_oversized_post_is_still_refused(client, monkeypatch):
    """The ceiling itself is unchanged: it still bites an authenticated caller."""
    monkeypatch.setenv("MECH_MAX_BODY_BYTES", "1024")
    response = client.post("/api/ping", content=b"x" * 4096,
                           headers={"Content-Type": "application/json",
                                    "Authorization": f"Bearer {TOKEN}"})
    assert response.status_code == 413, response.text


def test_a_401_carries_cors_headers(client):
    """A browser client must be able to READ the refusal."""
    response = client.get("/api/status", headers={"Origin": "http://localhost:5173"})
    assert response.status_code == 401
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert response.headers.get("access-control-allow-credentials") == "true"


def test_a_403_carries_cors_headers(client):
    _cookie_client(client)
    response = client.get("/api/status", headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 403
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
