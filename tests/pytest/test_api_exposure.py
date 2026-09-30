"""Exposure tests for the unauthenticated FastAPI surface.

The API has no authentication and now exposes plugin install/enable plus
per-runner model weights, so what it binds to and which origins it trusts are
security-relevant, not preferences.

The server used to bind 0.0.0.0 (reachable from the LAN) and allow the "null"
and "file://" CORS origins (any local HTML file or sandboxed iframe could make
credentialed calls). Both are asserted here so they cannot quietly come back.
"""
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_PY = REPO_ROOT / "main.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _source() -> str:
    return MAIN_PY.read_text(encoding="utf-8")


def test_binds_loopback_by_default():
    """0.0.0.0 put an unauthenticated API on the network."""
    source = _source()
    run = re.search(r"uvicorn\.run\((.*?)\n\s*\)", source, re.S)
    assert run, "could not find the uvicorn.run call"
    block = run.group(1)

    host_arg = re.search(r"host\s*=\s*([^,\n]+)", block)
    assert host_arg, f"no host argument in uvicorn.run: {block[:200]}"
    host_expr = host_arg.group(1).strip()
    assert "0.0.0.0" not in host_expr, (
        f"uvicorn.run binds {host_expr}; this API has no authentication"
    )


def test_loopback_is_the_default_host_value():
    """The env override must default to loopback, not to 0.0.0.0."""
    source = _source()
    default = re.search(
        r'os\.environ\.get\(\s*"MECH_BIND_HOST"\s*,\s*"([^"]+)"', source)
    assert default, "MECH_BIND_HOST must supply an explicit default"
    assert default.group(1) in ("127.0.0.1", "localhost"), (
        f"default bind host is {default.group(1)}, expected loopback"
    )


def test_bind_host_is_overridable():
    """Containers and shared dev machines still need a way to opt in."""
    assert "MECH_BIND_HOST" in _source(), (
        "provide MECH_BIND_HOST so deliberate exposure stays possible"
    )


def test_cors_does_not_allow_null_or_file_origins():
    """Neither origin is used by this repo, and both allow local files to call."""
    source = _source()
    origins = re.search(r"_CORS_ORIGINS\s*=\s*\[([^\]]*)\]", source)
    assert origins, "could not find the _CORS_ORIGINS list"
    listed = origins.group(1)
    assert '"null"' not in listed, (
        'the "null" origin lets any sandboxed iframe make credentialed requests'
    )
    assert "file://" not in listed, (
        "the file:// origin lets any local HTML file make credentialed requests"
    )


def test_cors_still_allows_the_vite_dev_server():
    """The dev server is the actual browser client; it must keep working."""
    source = _source()
    assert "http://localhost:5173" in source
    assert "http://127.0.0.1:5173" in source


def test_cors_override_env_still_supported():
    assert "MECH_CORS_ORIGINS" in _source()


def test_no_route_is_registered_on_a_wildcard_host():
    """Sanity: the app itself is importable and exposes no host-independent route."""
    import os
    os.environ.setdefault("MECH_STORAGE_DB", ":memory:")
    import main as backend_main
    paths = backend_main.app.openapi()["paths"]
    assert paths, "the app exposes no routes"
    # Every route is path-based; none encodes a hostname.
    for path in paths:
        assert "://" not in path, f"route encodes a host: {path}"
