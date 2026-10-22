"""The browser session: a cookie JavaScript cannot read replaces the token.

The web build has no privileged process, so it must not hold the bearer token.
`POST /session` exchanges the token for an HttpOnly, SameSite=Strict cookie
signed with the token; the auth middleware accepts that cookie as a credential.

Tested against the real ASGI app. `TestClient` is used *without* its context
manager on purpose: entering it runs the lifespan, which pre-loads model
weights, and these tests are about auth, not inference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TOKEN = "session-test-token-4f1c"


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("MECH_API_TOKEN", TOKEN)

    from backend.core import auth as auth_mod

    auth_mod.reset_cache()

    from fastapi.testclient import TestClient

    from backend.main import app

    return TestClient(app)


def _mint(client) -> str:
    response = client.post("/session", headers={"Authorization": f"Bearer {TOKEN}"})
    assert response.status_code == 200, response.text
    cookie = response.cookies.get("mech_session")
    assert cookie, f"no session cookie in {dict(response.cookies)}"
    return cookie


def test_session_requires_a_valid_token(client) -> None:
    assert client.post("/session").status_code == 401
    assert client.post("/session", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/session", params={"token": "wrong"}).status_code == 401


def test_session_cookie_is_httponly_and_strict(client) -> None:
    response = client.post("/session", headers={"Authorization": f"Bearer {TOKEN}"})
    header = response.headers.get("set-cookie", "")
    lowered = header.lower()
    assert "mech_session=" in lowered
    assert "httponly" in lowered, f"the session cookie is readable by page script: {header}"
    assert "samesite=strict" in lowered, header
    assert "path=/" in lowered, header


def test_session_body_does_not_echo_the_token(client) -> None:
    response = client.post("/session", headers={"Authorization": f"Bearer {TOKEN}"})
    assert TOKEN not in response.text
    assert TOKEN not in response.headers.get("set-cookie", "")


def test_cookie_authenticates_the_control_plane(client) -> None:
    """The point of the exchange: no header, still authenticated."""
    from fastapi.testclient import TestClient

    from backend.main import app

    cookie = _mint(client)

    # A client that never minted a session is still rejected.
    assert TestClient(app).get("/api/status").status_code == 401, (
        "an anonymous request was accepted"
    )

    # The minting client stores the HttpOnly cookie and is authenticated by it.
    assert client.get("/api/status").status_code == 200

    # The cookie also works on a client that never saw the Set-Cookie header.
    fresh = TestClient(app)
    fresh.cookies.set("mech_session", cookie)
    assert fresh.get("/api/status").status_code == 200


def test_a_forged_cookie_is_rejected(client) -> None:
    assert client.get("/api/status", cookies={"mech_session": "nonce.deadbeef"}).status_code == 401
    assert client.get("/api/status", cookies={"mech_session": ""}).status_code == 401


def test_rotating_the_token_invalidates_issued_sessions(client, monkeypatch) -> None:
    """A session must not outlive the credential it was minted from."""
    cookie = _mint(client)
    assert client.get("/api/status", cookies={"mech_session": cookie}).status_code == 200

    monkeypatch.setenv("MECH_API_TOKEN", "rotated-token-91ab")
    from backend.core import auth as auth_mod

    auth_mod.reset_cache()

    assert client.get("/api/status", cookies={"mech_session": cookie}).status_code == 401


def test_logout_clears_the_cookie(client) -> None:
    response = client.delete("/session")
    assert response.status_code == 200
    header = response.headers.get("set-cookie", "").lower()
    assert "mech_session=" in header
    # An expired cookie is how a deletion is expressed.
    assert "max-age=0" in header or "expires=" in header, header


def test_session_path_stays_reachable_without_a_credential(client) -> None:
    """`/session` must be public or nothing could ever obtain a cookie."""
    from backend.core import auth as auth_mod

    assert auth_mod.is_public_path("/session", "POST") is True
    assert auth_mod.is_public_path("/api/status", "GET") is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
