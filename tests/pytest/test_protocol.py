"""End-to-end tests of the FastAPI backend via an in-process TestClient.

The original stdio JSON-lines sidecar protocol was retired when the backend
consolidated onto the FastAPI HTTP surface (``backend/main.py``); these tests
exercise the real running app instead.
"""
from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.main import app
from backend.core import auth as auth_mod


@pytest.fixture(scope="module", autouse=True)
def _test_token():
    """Use a known token for this module, then put the environment back.

    This used `os.environ[...] = ...` with no restore, so the override
    outlived the module and any later module that captured its headers at
    import time started failing with 401 -- the pollution only showed up in a
    full-suite run, not when this file ran alone.
    """
    previous = os.environ.get("MECH_API_TOKEN")
    os.environ["MECH_API_TOKEN"] = "protocol-test-token"
    auth_mod.reset_cache()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("MECH_API_TOKEN", None)
        else:
            os.environ["MECH_API_TOKEN"] = previous
        auth_mod.reset_cache()


def _h():
    return auth_mod.auth_headers("protocol-test-token")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_root_roundtrip(client):
    res = client.get("/")
    assert res.status_code == 200
    body = res.json()
    assert body["name"] == "MECH Platform"
    assert body["status"] == "running"


def test_health_roundtrip(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "healthy"}


def test_ping_roundtrip(client):
    res = client.post("/api/v1/ping", headers=_h())
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_unknown_route_returns_404(client):
    res = client.get("/api/v1/does_not_exist", headers=_h())
    assert res.status_code == 404


def test_anonymous_control_plane_rejected(client):
    assert client.post("/api/v1/ping").status_code == 401
    assert client.get("/api/v1/does_not_exist").status_code == 401
