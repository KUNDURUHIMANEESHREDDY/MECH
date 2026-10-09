"""Inbound bearer authentication for the MECH control plane.

The rule this module exists to enforce
--------------------------------------
**Loopback reachability is not authorization, and neither is Origin.**
Every caller that reaches ``/api/*`` (including ``/api/v1/*``), ``/mcp``,
``/health/subsystems``, ``/openapi.json``, ``/docs`` or ``/redoc`` must
present a bearer token the backend generated. ``/`` and ``/health`` stay
public; the built frontend (static files) stays public so the app can load
before it knows the token.

How it is used
--------------
Backend (``backend/main.py``)::

    @app.middleware("http")
    async def _auth_middleware(request, call_next):
        from backend.core import auth as auth_mod
        if auth_mod.is_public_path(request.url.path, request.method):
            return await call_next(request)
        ... verify Authorization: Bearer ...

Tests::

    from backend.core import auth as auth_mod
    headers = {"Authorization": f"Bearer {auth_mod.get_bearer_token()}"}

Token source of truth (in order):

1. ``MECH_API_TOKEN`` env var when set and non-empty (operator / test override).
2. Otherwise a random token persisted to
   ``backend/storage/.mech_api_token`` with ``0o600`` so the trusted
   Electron renderer can read it across restarts without churning every boot.
   The file is runtime state, never committed.

Comparison is constant-time via :func:`hmac.compare_digest`.
"""

from __future__ import annotations

import hmac
import os
import secrets
from pathlib import Path
from typing import Dict, Optional

_TOKEN_ENV = "MECH_API_TOKEN"
_TOKEN_FILENAME = ".mech_api_token"

# In-memory cache for the file-backed token so every request does not hit disk.
# The env override is checked first on every call so tests can monkeypatch the
# env without reloading this module.
_cached_file_token: Optional[str] = None


def _storage_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "storage"


def _token_file() -> Path:
    configured = os.environ.get("MECH_TOKEN_FILE", "").strip()
    if configured:
        return Path(configured)
    return _storage_dir() / _TOKEN_FILENAME


def _read_file_token(path: Path) -> Optional[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    token = text.strip()
    return token or None


def _write_file_token(path: Path, token: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write then restrict: the token must not be world-readable.
    path.write_text(token + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        # Windows ACLs do not honor POSIX mode bits; the file still holds a
        # high-entropy secret readable only by this user's profile dir.
        pass


def get_bearer_token() -> str:
    """Return the bearer token this process accepts.

    Env override wins so operators and tests have a deterministic seam.
    Otherwise load-or-create the persisted file token.
    """
    global _cached_file_token
    env_token = os.environ.get(_TOKEN_ENV, "").strip()
    if env_token:
        return env_token
    if _cached_file_token:
        return _cached_file_token
    path = _token_file()
    existing = _read_file_token(path)
    if existing:
        _cached_file_token = existing
        return existing
    fresh = secrets.token_urlsafe(32)
    _write_file_token(path, fresh)
    _cached_file_token = fresh
    return fresh


def verify_bearer(provided: Optional[str]) -> bool:
    """True only when ``provided`` equals the active bearer token.

    Never raises on bad input: ``None``, empty, and non-string values are
    simply not authorized. Comparison is constant-time via
    :func:`hmac.compare_digest`.
    """
    if not isinstance(provided, str):
        return False
    candidate = provided.strip()
    if not candidate:
        return False
    expected = get_bearer_token()
    if not isinstance(expected, str) or not expected:
        return False
    return hmac.compare_digest(candidate.encode("utf-8"),
                               expected.encode("utf-8"))


def extract_bearer(authorization_header: Optional[str]) -> Optional[str]:
    """Pull the credential out of an ``Authorization`` header value."""
    if not isinstance(authorization_header, str):
        return None
    scheme, sep, credential = authorization_header.partition(" ")
    if not sep:
        return None
    if scheme.strip().lower() != "bearer":
        return None
    credential = credential.strip()
    return credential or None


def is_public_path(path: str, method: str = "GET") -> bool:
    """True when a request bypasses bearer auth.

    Public: ``/``, ``/health``, CORS preflight (``OPTIONS``), and the built
    frontend's static files (any ``GET``/``HEAD`` path that is not part of the
    control plane). Everything under ``/api``, ``/mcp``, ``/health/``,
    ``/openapi.json``, ``/docs`` and ``/redoc`` requires auth.
    """
    m = (method or "GET").upper()
    # CORS preflight carries no Authorization header by design. Let the CORS
    # middleware answer it; the follow-up real request still requires auth.
    # Anonymous OPTIONS never implies trust.
    if m == "OPTIONS":
        return True
    p = path or "/"
    if not p.startswith("/"):
        p = "/" + p
    if p == "/" or p == "/health":
        return True
    if p == "/openapi.json" or p == "/openapi.yaml":
        return False
    if p == "/docs" or p.startswith("/docs/"):
        return False
    if p == "/redoc" or p.startswith("/redoc/"):
        return False
    if p == "/health/subsystems" or p.startswith("/health/"):
        return False
    if p == "/api" or p.startswith("/api/"):
        return False
    if p == "/mcp" or p.startswith("/mcp/"):
        return False
    return True


def auth_headers(token: Optional[str] = None) -> Dict[str, str]:
    """Authorization header dict for tests and same-process callers."""
    return {"Authorization": f"Bearer {token if token else get_bearer_token()}"}


def reset_cache() -> None:
    """Clear the in-memory file-token cache (tests only)."""
    global _cached_file_token
    _cached_file_token = None
