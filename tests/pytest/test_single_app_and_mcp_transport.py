"""One control plane: one FastAPI app, one gate, one working MCP transport.

* **L2** -- ``backend/interpretability/api.py`` built a SECOND FastAPI
  application with no bearer gate, no CORS policy and no body-size
  ceiling. Nothing mounted it, and nothing outside its own package
  imported it, so it was deleted rather than re-mounted: a tree with two
  security postures is the shape the root ``main.py`` shim exists to end.

* **L3** -- the MCP HTTP transport was mounted but every authenticated
  request 500'd with "Task group is not initialized": FastMCP's
  streamable-HTTP app only serves requests from inside its session
  manager's task group, and a mounted sub-application never receives a
  lifespan of its own. The transport is now built and started by the
  application's own lifespan.

* **L7** -- ``PluginManager.load_active_plugins`` executed plugin code in
  the backend process and had zero callers; the production path runs
  plugins in a separate worker process.
"""

from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

TOKEN = "single-app-token-9b2e"


# --------------------------------------------------------------------------- #
# L2: exactly one FastAPI application in the tree
# --------------------------------------------------------------------------- #


def _backend_files():
    return sorted((REPO_ROOT / "backend").rglob("*.py"))


def test_only_backend_main_builds_a_fastapi_application():
    """Walk backend/ and find every FastAPI() construction.

    A second FastAPI instance is a second security posture: it has its own
    middleware stack, and unless it is mounted under backend/main.py's gate
    it serves the control plane with no credential at all.
    """
    offenders = []
    for path in _backend_files():
        source = path.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name == "FastAPI":
                offenders.append(path.relative_to(REPO_ROOT).as_posix())
    assert offenders == ["backend/main.py"], (
        "more than one FastAPI application in the tree: " + ", ".join(offenders))


def test_the_interpretability_api_module_is_gone():
    """It was deleted, not stubbed out."""
    assert not (REPO_ROOT / "backend" / "interpretability" / "api.py").exists()


def test_backend_interpretability_still_imports_and_exposes_no_app():
    import backend.interpretability as interp
    assert not hasattr(interp, "app"), (
        "backend.interpretability must not expose a second application")
    assert interp.__version__
    assert "StatisticsComputer" in interp.__all__


def test_the_gate_app_is_the_only_application():
    from backend.main import app
    from fastapi import FastAPI
    assert isinstance(app, FastAPI)


# --------------------------------------------------------------------------- #
# L3: the MCP HTTP transport actually works
# --------------------------------------------------------------------------- #


INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "mech-test", "version": "0.0.1"},
    },
}


def test_an_authenticated_mcp_http_request_succeeds(monkeypatch):
    """The mounted transport must serve, not 500.

    Before the fix this raised "RuntimeError: Task group is not
    initialized. Make sure to use run()." because the MCP session
    manager's lifespan never ran.
    """
    monkeypatch.setenv("MECH_API_TOKEN", TOKEN)
    from backend.core import auth as auth_mod
    auth_mod.reset_cache()
    from backend.main import app

    try:
        with TestClient(app) as client:
            response = client.post(
                "/mcp/mcp",
                headers={
                    # The MCP transport's DNS-rebinding protection only
                    # answers on a host it is allowed on.
                    "Host": "127.0.0.1:8000",
                    "Authorization": f"Bearer {TOKEN}",
                    "Accept": "application/json, text/event-stream",
                    "Content-Type": "application/json",
                },
                content=json.dumps(INITIALIZE),
            )
            assert response.status_code == 200, response.text
            assert "mech-control" in response.text, response.text
            assert response.headers.get("mcp-session-id"), response.headers
    finally:
        auth_mod.reset_cache()


def test_an_anonymous_mcp_http_request_is_refused(monkeypatch):
    """The gate holds for the MCP transport too."""
    monkeypatch.setenv("MECH_API_TOKEN", TOKEN)
    from backend.core import auth as auth_mod
    auth_mod.reset_cache()
    from backend.main import app

    try:
        with TestClient(app) as client:
            response = client.post("/mcp/mcp", json={})
            assert response.status_code == 401, response.text
    finally:
        auth_mod.reset_cache()


def test_the_route_table_does_not_depend_on_startup(monkeypatch):
    """The application's routes are the same before and after it has run.

    The MCP mount used to be installed by the lifespan, so the route table
    gained ``/mcp`` the first time the application started -- and kept it
    afterwards. Everything that reads the route table therefore read a
    different application depending on startup history: the published
    OpenAPI schema, the route-authorization matrix, and any introspection
    of the app. Two snapshots of the SAME object taken moments apart
    disagreed, which is what `test_adversarial_challenge.py` caught.

    The mount is now registered at import time, and startup swaps the
    transport behind it.
    """
    monkeypatch.setenv("MECH_API_TOKEN", TOKEN)
    from backend.core import auth as auth_mod
    auth_mod.reset_cache()
    from backend.main import app

    def _table():
        return sorted(
            (getattr(route, "path", ""),
             tuple(sorted(getattr(route, "methods", []) or [])))
            for route in app.routes
        )

    try:
        before = _table()
        assert ("/mcp", ()) in before, (
            "the MCP endpoint must be in the route table before the "
            "application has ever started")
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
            during = _table()
        after = _table()
        assert before == during == after, (
            "the route table changed across startup: "
            f"before={len(before)} during={len(during)} after={len(after)}")
    finally:
        auth_mod.reset_cache()


def test_the_mcp_endpoint_answers_before_startup(monkeypatch):
    """A mounted endpoint that is not running says so; it does not 404.

    The route exists from import time, so a request to it before startup
    has to be answered by the application rather than falling through to
    the static handler. What it must not do is look like a path that does
    not exist.

    ``TestClient`` only runs the lifespan when it is used as a context
    manager, so a bare client here really does arrive before startup.
    """
    monkeypatch.setenv("MECH_API_TOKEN", TOKEN)
    from backend.core import auth as auth_mod
    auth_mod.reset_cache()
    from backend.main import app

    try:
        client = TestClient(app)
        response = client.post(
            "/mcp/mcp",
            json={},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert response.status_code == 503, response.text
        assert "not running" in response.text
    finally:
        auth_mod.reset_cache()


# --------------------------------------------------------------------------- #
# L7: no in-process plugin loader
# --------------------------------------------------------------------------- #


def test_the_in_process_plugin_loader_is_gone():
    from backend.plugins.plugin_manager import PluginManager
    assert not hasattr(PluginManager, "load_active_plugins"), (
        "executing plugin code inside the backend process is dead code; the "
        "production path is PluginService.enable -> start_remote_plugin")


def test_nothing_calls_the_in_process_plugin_loader():
    for path in _backend_files():
        source = path.read_text(encoding="utf-8", errors="ignore")
        assert "load_active_plugins" not in source, (
            f"{path} still refers to the in-process plugin loader")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
