"""Exposure tests for the authenticated FastAPI surface.

The API requires bearer auth (backend/core/auth.py) and exposes plugin
install/enable plus per-runner model weights, so what it binds to and which
origins it trusts are security-relevant, not preferences.

Three of these assertions are unchanged from security/plugin-and-path-hardening:
loopback binding, MECH_BIND_HOST as a deliberate opt-in, and MECH_CORS_ORIGINS
as an additive override.

Two changed, and the reason matters.

**CORS now lives in backend/main.py, not the root shim.** The root main.py used
to build a second, divergent FastAPI instance with its own allowlist, so the
active policy depended on which file was launched. Root main.py is now a shim
that re-exports backend.main.app, so the allowlist is asserted where it is
actually applied.

**The "null" origin stays allowed, and this is load-bearing.** The originating
branch removed it on the grounds that "nothing in this repo serves the frontend
from a file:// origin". That is true of the Vite dev server and false of the
packaged desktop app: frontend/electron/main.js calls `win.loadFile(indexPath)`
on the production path and lists 'file://' in its own origin allowlist. A
browser reports a file:// page as the opaque origin "null", so removing it does
not harden anything -- it breaks the shipped desktop app's API calls. It is
retained deliberately and the reasoning is asserted below, because the next
person to read "null is insecure" will otherwise remove it again.
"""
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_PY = REPO_ROOT / "main.py"
BACKEND_MAIN_PY = REPO_ROOT / "backend" / "main.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _source() -> str:
    return MAIN_PY.read_text(encoding="utf-8")


def _backend_source() -> str:
    return BACKEND_MAIN_PY.read_text(encoding="utf-8")


def test_binds_loopback_by_default():
    """0.0.0.0 would put a privileged API on the network."""
    source = _source()
    run = re.search(r"uvicorn\.run\((.*?)\n\s*\)", source, re.S)
    assert run, "could not find the uvicorn.run call"
    block = run.group(1)

    host_arg = re.search(r"host\s*=\s*([^,\n]+)", block)
    assert host_arg, f"no host argument in uvicorn.run: {block[:200]}"
    host_expr = host_arg.group(1).strip()
    # The host is now an expression (MECH_BIND_HOST with a loopback default),
    # so assert on the default rather than on a literal.
    assert "0.0.0.0" not in host_expr, (
        f"uvicorn.run binds {host_expr}; this API is privileged"
    )


def test_loopback_is_the_default_host_value():
    """The env override must default to loopback, not to 0.0.0.0."""
    default = re.search(
        r'os\.environ\.get\(\s*"MECH_BIND_HOST"\s*,\s*"([^"]+)"', _source())
    assert default, "MECH_BIND_HOST must supply an explicit default"
    assert default.group(1) in ("127.0.0.1", "localhost"), (
        f"default bind host is {default.group(1)}, expected loopback"
    )


def test_bind_host_is_overridable():
    """Containers and shared dev machines still need a way to opt in."""
    assert "MECH_BIND_HOST" in _source(), (
        "provide MECH_BIND_HOST so deliberate exposure stays possible"
    )


def test_root_main_is_only_a_shim():
    """The allowlist must not exist in two places with different contents."""
    source = _source()
    assert "CORSMiddleware" not in source, (
        "root main.py builds its own CORS policy; it must re-export "
        "backend.main.app so there is exactly one"
    )
    assert "FastAPI(" not in source, (
        "root main.py constructs a second FastAPI instance"
    )


def test_cors_allows_the_vite_dev_server():
    """The dev server is the actual browser client; it must keep working.

    Vite binds a random port when 5173 is taken, so loopback origins are matched
    by an anchored regex rather than enumerated. The original assertion looked
    for "http://localhost:5173" inside a literal list, which is neither how the
    allowlist is written nor sufficient: the property that matters is that a
    loopback origin is permitted, on whatever port it lands.
    """
    assert _cors_allows("http://localhost:5173")
    assert _cors_allows("http://127.0.0.1:5173")
    assert _cors_allows("http://localhost:5174")
    assert _cors_allows("http://127.0.0.1:4173")


def test_cors_loopback_regex_is_anchored():
    """A subdomain must not inherit loopback trust.

    This is the reason the allowlist is a regex and not a list. A pattern
    matching "localhost" unanchored would also match
    "http://localhost.evil.com", letting any site on that hostname make
    credentialed calls to a privileged API.
    """
    assert not _cors_allows("http://localhost.evil.com")
    assert not _cors_allows("http://127.0.0.1.attacker.org")
    assert not _cors_allows("http://notlocalhost")
    assert not _cors_allows("https://evil.com")


def _cors_allows(origin: str) -> bool:
    """Mirror the middleware's decision: list membership OR full regex match.

    starlette allows an origin if it is in `allow_origins` or fully matches
    `allow_origin_regex`. Testing the real predicate keeps this from decaying
    into assertions about source text.
    """
    from backend.main import _LOOPBACK_ORIGIN_REGEX, _cors_origins

    if origin in _cors_origins():
        return True
    return re.fullmatch(_LOOPBACK_ORIGIN_REGEX, origin) is not None


def _resolved_origins() -> list:
    from backend.main import _cors_origins
    return list(_cors_origins())


def test_cors_allows_the_opaque_null_origin_for_the_packaged_app():
    """frontend/electron/main.js calls win.loadFile() in production.

    A file:// page is reported by the browser as the opaque origin "null", so
    the packaged desktop app can only reach the API if "null" is allowed.
    Removing it breaks the shipped app rather than hardening it.
    """
    assert "null" in _resolved_origins(), (
        "the packaged Electron app loads from file:// and needs the opaque "
        "'null' origin"
    )

    # Assert the dependency rather than trusting the comment.
    electron_main = REPO_ROOT / "frontend" / "electron" / "main.js"
    if electron_main.exists():
        text = electron_main.read_text(encoding="utf-8", errors="ignore")
        assert "loadFile(" in text, (
            "frontend/electron/main.js no longer loads via file://; if the "
            "packaged app moved to a real origin, 'null' can be dropped"
        )


def test_cors_does_not_allow_a_bare_wildcard():
    """Wildcard plus credentials is rejected by browsers and means nothing."""
    origins = _resolved_origins()
    assert "*" not in origins


def test_cors_override_env_still_supported():
    assert "MECH_CORS_ORIGINS" in _backend_source()


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
