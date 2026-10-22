"""The frontend bundle must contain no privileged bearer token.

Superseded contract
-------------------
This file used to require ``api.ts`` to read ``runtime.__MECH_API_TOKEN__`` and
``localStorage['mech_api_token']``. Both were still renderer-readable secrets,
so "not bundled at build time" was the wrong bar: the token was delivered to
the renderer at runtime instead. Renderer code now holds no credential at all --
the desktop proxies through the Electron main process and the browser build uses
an HttpOnly session cookie.

What is pinned now
------------------
* No build-time env token (Vite inlines ``VITE_*`` into the shipped bundle).
* No ``window.__MECH_API_TOKEN__`` global.
* No persistent browser storage for the control-plane token.
* No ``apiToken()`` accessor and no ``Authorization`` header built in the
  renderer.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

API_FILE = REPO_ROOT / "frontend" / "src" / "services" / "api.ts"


def _read_api_file() -> str:
    return API_FILE.read_text(encoding="utf-8")


def _code_only(source: str) -> str:
    """Strip comments: they are allowed to *name* the removed mechanisms."""
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", source)


def test_no_vite_mech_api_token_in_source() -> None:
    """A VITE_-prefixed token would be compiled into the shipped bundle."""
    code = _code_only(_read_api_file())
    assert "VITE_MECH_API_TOKEN" not in code, (
        "VITE_MECH_API_TOKEN is referenced in executable code; Vite inlines "
        "VITE_* into the bundle, which ships the secret to every client"
    )


def test_renderer_global_token_is_gone() -> None:
    """`window.__MECH_API_TOKEN__` was renderer-readable. It must not return."""
    code = _code_only(_read_api_file())
    assert "__MECH_API_TOKEN__" not in code, (
        "api.ts still reads the renderer-global token; the desktop token must "
        "stay in the Electron main process and travel only over `api:request`"
    )


def test_no_persistent_browser_storage_for_the_token() -> None:
    """Persisting the control-plane token in localStorage is not isolation."""
    code = _code_only(_read_api_file())
    assert "localStorage" not in code, (
        "api.ts still touches localStorage for the control-plane token"
    )
    assert "mech_api_token" not in code, (
        "api.ts still references the persisted token key"
    )


def test_no_token_accessor_is_exported() -> None:
    """`apiToken()` was the seam a future caller would use to read the token."""
    code = _code_only(_read_api_file())
    assert "apiToken" not in code, (
        "api.ts still exports a token accessor; renderer code has no credential "
        "to return"
    )


def test_auth_headers_builds_no_authorization_header() -> None:
    """The renderer must not construct a bearer header from local state."""
    code = _code_only(_read_api_file())
    headers_fn = re.search(
        r"export function authHeaders\([^)]*\)[^{]*\{(?P<body>.*?)\n\}",
        code,
        flags=re.DOTALL,
    )
    assert headers_fn, "authHeaders() is missing from api.ts"
    body = headers_fn.group("body")
    assert "Authorization" not in body
    assert "Bearer" not in body


def test_both_credential_paths_are_present() -> None:
    """Desktop proxies; the browser sends the session cookie."""
    code = _code_only(_read_api_file())
    assert "apiProxy()" in code
    assert "apiRequest" in code, (
        "api.ts does not call the preload's apiRequest proxy"
    )
    assert "credentials: 'same-origin'" in code, (
        "the browser build does not send the session cookie"
    )


def test_scan_is_not_vacuous() -> None:
    """Guard the checks above: the file must still contain what we look for."""
    source = _read_api_file()
    assert "export const api = {" in source
    assert "export function authHeaders" in source


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
