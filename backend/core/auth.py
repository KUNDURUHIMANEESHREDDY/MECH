"""Inbound bearer authentication for the MECH control plane.

The rule this module exists to enforce
--------------------------------------
**Loopback reachability is not authorization, and neither is Origin.**
Every caller that reaches ``/api/*`` (including ``/api/v1/*``), ``/mcp``,
``/health/subsystems``, ``/openapi.json``, ``/docs`` or ``/redoc`` must
present a bearer token the backend generated -- or a session cookie minted
from one. ``/`` and ``/health`` stay public; the built frontend (static
files) stays public so the app can load before it knows the token, and
``/session`` is reachable so a browser can exchange a token for a cookie.

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

import hashlib
import hmac
import os
import re
import secrets
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlsplit

_TOKEN_ENV = "MECH_API_TOKEN"
_TOKEN_FILENAME = ".mech_api_token"

# In-memory cache for the file-backed token so every request does not hit disk.
# The env override is checked first on every call so tests can monkeypatch the
# env without reloading this module.
_cached_file_token: Optional[str] = None


def _storage_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "storage"


def _token_file() -> Path:
    """The one place the token is read from and written to.

    Deliberately not configurable from the environment. ``MECH_TOKEN_FILE``
    used to redirect this path, which meant any process able to set an
    environment variable before the first ``get_bearer_token()`` call chose
    where the control-plane token was persisted and read -- an
    attacker-controlled path pointing at a file it had written. The seam did
    not earn its keep: nothing in this repository sets the variable, and the
    path below is already fixed relative to this module.
    """
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


#: Cookie that carries a browser session.
SESSION_COOKIE = "mech_session"


def _session_key() -> bytes:
    """The signing key: the active bearer token.

    Rotating the token invalidates every issued session, which is the behaviour
    an operator expects from "rotate the credential".
    """
    return get_bearer_token().encode("utf-8")


def issue_session() -> str:
    """Mint a session value: ``<nonce>.<hmac>``, stateless.

    No server-side session store is needed, so a restart does not sign every
    browser out, and nothing has to be cleaned up when a session is abandoned.
    """
    nonce = secrets.token_urlsafe(24)
    mac = hmac.new(_session_key(), nonce.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{nonce}.{mac}"


def verify_session(value: Optional[str]) -> bool:
    """True only for a session this process minted with the current token."""
    if not isinstance(value, str):
        return False
    nonce, sep, mac = value.partition(".")
    if not sep or not nonce or not mac:
        return False
    expected = hmac.new(_session_key(), nonce.encode("utf-8"), hashlib.sha256).hexdigest()
    return hmac.compare_digest(mac, expected)


def extract_session_cookie(cookie_header: Optional[str]) -> Optional[str]:
    """Pull :data:`SESSION_COOKIE` out of a ``Cookie`` header value."""
    if not isinstance(cookie_header, str) or not cookie_header.strip():
        return None
    try:
        jar = SimpleCookie()
        jar.load(cookie_header)
    except Exception:
        return None
    morsel = jar.get(SESSION_COOKIE)
    if morsel is None:
        return None
    value = morsel.value.strip()
    return value or None


# --------------------------------------------------------------------------- #
# Origins: who may use the session cookie, and who may carry credentials
# --------------------------------------------------------------------------- #
#
# A "site" is (scheme, registrable domain). The PORT IS NOT PART OF IT
# (RFC 6265bis 5.2.3), so a page served from http://localhost:3000 is
# same-site with http://localhost:8000 and the browser attaches the
# SameSite=Strict session cookie to a credentialed request from it. CORS
# decides whether the response is *readable*; it does not decide whether
# the cookie is *sent*. The server therefore has to check the Origin of a
# cookie-authenticated request itself.

#: Loopback hosts on any port. Anchored, so "http://localhost.evil.com"
#: and "http://127.0.0.1.attacker.org" do not inherit loopback trust.
LOOPBACK_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"

#: Exact origins the operator trusts with the operator's credentials: they
#: may authenticate with the session cookie and may carry credentials
#: cross-origin.
TRUSTED_ORIGINS_ENV = "MECH_TRUSTED_ORIGINS"

#: Opt-in: trust EVERY loopback origin (any port) for cookie
#: authentication. Off by default -- a loopback port is not an identity.
TRUST_LOOPBACK_ORIGINS_ENV = "MECH_TRUST_LOOPBACK_ORIGINS"

#: Opt-in: accept the bearer token from the query string (``?token=``).
#: Off by default -- uvicorn's access log writes the full query string, so
#: the credential would land in the log in cleartext.
ALLOW_QUERY_TOKEN_ENV = "MECH_ALLOW_QUERY_TOKEN"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _env_flag(name: str) -> bool:
    """True when an environment variable is set to a truthy value."""
    return (os.environ.get(name, "") or "").strip().lower() in _TRUTHY


def is_loopback_origin(origin: Optional[str]) -> bool:
    """True for a loopback origin (localhost / 127.0.0.1 / [::1])."""
    if not isinstance(origin, str):
        return False
    candidate = origin.strip()
    if not candidate:
        return False
    return re.fullmatch(LOOPBACK_ORIGIN_REGEX, candidate) is not None


def normalize_origin(value: str) -> Optional[str]:
    """Canonical form of an origin, or None when the value is not one.

    An origin is scheme://host[:port] and nothing else. A path, a query
    string, a fragment, credentials in the URL or a wildcard host make the
    value something other than an origin, and an allowlist entry that is
    not an origin is a mistake worth refusing rather than matching by
    accident.
    """
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate or "*" in candidate:
        return None
    try:
        parts = urlsplit(candidate)
        host = parts.hostname
        port = parts.port
    except ValueError:
        return None
    if parts.scheme not in ("http", "https"):
        return None
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        return None
    if parts.username or parts.password:
        return None
    if not host:
        return None
    if ":" in host:
        host = f"[{host}]"
    netloc = host if port is None else f"{host}:{port}"
    return f"{parts.scheme}://{netloc}"


def trusted_origins() -> frozenset[str]:
    """The exact origins the operator trusts (MECH_TRUSTED_ORIGINS)."""
    raw = os.environ.get(TRUSTED_ORIGINS_ENV, "") or ""
    out = set()
    for entry in raw.split(","):
        candidate = entry.strip().rstrip("/")
        if not candidate:
            continue
        if candidate == "null":
            out.add("null")
            continue
        normalized = normalize_origin(candidate)
        if normalized:
            out.add(normalized)
    return frozenset(out)


def trust_loopback_origins() -> bool:
    """True when the operator opted into trusting every loopback origin."""
    return _env_flag(TRUST_LOOPBACK_ORIGINS_ENV)


def query_token_allowed() -> bool:
    """True when the operator opted into accepting ``?token=``."""
    return _env_flag(ALLOW_QUERY_TOKEN_ENV)


def extract_query_token(params) -> Optional[str]:
    """The bearer token from the query string, when that seam is enabled.

    Returns None unless the operator opted in (MECH_ALLOW_QUERY_TOKEN).
    The default is off: the query string is written to the access log, so
    the credential would be readable there in cleartext.
    """
    if not query_token_allowed():
        return None
    try:
        token = params.get("token") or params.get("bearer_token")
    except Exception:
        return None
    if isinstance(token, str) and token.strip():
        return token.strip()
    return None


def origin_may_carry_credentials(origin: Optional[str]) -> bool:
    """True when ``origin`` may be sent Access-Control-Allow-Credentials.

    Loopback origins may. The opaque "null" origin may NOT: it is what a
    file:// page, a sandboxed iframe and a data: URL all report, so it
    identifies no page the server can trust, and the session gate refuses
    it for exactly that reason. "null" stays in the CORS *allowlist* (the
    packaged desktop app loads from file:// and needs its responses to be
    readable), but it never carries the operator's credentials.

    An origin the operator merely added to the CORS allowlist may not
    either, unless it is also listed in MECH_TRUSTED_ORIGINS: adding an
    origin to the allowlist must never be enough to switch on credentialed
    cross-origin access to the control plane.
    """
    if not isinstance(origin, str) or not origin.strip():
        return False
    candidate = origin.strip()
    if is_loopback_origin(candidate):
        return True
    key = normalize_origin(candidate) or candidate
    return key in trusted_origins()


def origin_allowed_for_session(origin: Optional[str], request_netloc: Optional[str],
                                method: str = "GET") -> bool:
    """True when a cookie-authenticated request may proceed.

    The session cookie is an *ambient* credential: the browser attaches it
    to any same-site request, and same-site ignores the port. A page on
    any loopback port is therefore able to present the operator's cookie,
    so the server checks the Origin itself.

    * No Origin header: allowed for safe methods only. A browser omits
      Origin on a same-origin GET/HEAD, and a state-changing request
      without one is not a request this server can vouch for.
    * The request's own origin (same host and port): always allowed. This
      is the standalone build serving its own frontend.
    * An origin the operator listed in MECH_TRUSTED_ORIGINS: allowed.
    * Any loopback origin: allowed only when the operator opted in with
      MECH_TRUST_LOOPBACK_ORIGINS (the cross-port browser dev flow).
    * The opaque origin "null": never allowed by default. It is what a
      file:// page, a sandboxed iframe and a data: URL all report, so it
      identifies no page the server can trust.
    """
    if not isinstance(origin, str) or not origin.strip():
        return (method or "GET").upper() in {"GET", "HEAD", "OPTIONS"}
    candidate = origin.strip()
    key = "null" if candidate == "null" else (normalize_origin(candidate) or candidate)
    if key in trusted_origins():
        return True
    if candidate == "null":
        return False
    try:
        netloc = urlsplit(candidate).netloc
    except ValueError:
        return False
    if request_netloc and netloc and netloc.lower() == request_netloc.lower():
        return True
    if trust_loopback_origins() and is_loopback_origin(candidate):
        return True
    return False


#: File extensions that identify static frontend assets.
#: Vite outputs these; they must be public so the browser can load the
#: frontend before it knows the bearer token.
_STATIC_ASSET_EXTENSIONS = frozenset({
    # JavaScript / CSS / Source maps
    ".js", ".css", ".map",
    # Fonts
    ".woff", ".woff2", ".ttf", ".eot",
    # Images
    ".ico", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".avif",
    # Manifest / manifest-like
    ".json", ".webmanifest",
    # Other common static types
    ".txt", ".xml", ".ico",
    # Images (additional)
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".avif",
})


def is_public_path(path: str, method: str = "GET") -> bool:
    """True when a request bypasses bearer auth.

    Defaults to PROTECTED (False). Explicitly allows:
      - OPTIONS (CORS preflight)
      - / and /health (liveness)
      - Static asset paths (by file extension, matching Vite output)

    Everything else requires bearer authentication.
    """
    m = (method or "GET").upper()
    # CORS preflight carries no Authorization header by design. Let the CORS
    # middleware answer it; the follow-up real request still requires auth.
    # Anonymous OPTIONS never implies trust.
    if m == "OPTIONS":
        return True

    p = (path or "/").strip()
    if not p:
        p = "/"
    if not p.startswith("/"):
        p = "/" + p

    # Exact public paths (liveness)
    if p == "/" or p == "/health":
        return True

    # Explicitly protected paths (control plane) - checked BEFORE static assets
    # so that control plane endpoints like /openapi.json are not caught by
    # static asset extension matching.
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

    # Session minting/clearing. Public so a browser can reach it before it has
    # a credential; the handler itself verifies the token it is given, and the
    # cookie it sets is HttpOnly and SameSite=Strict.
    if p == "/session":
        return True

    # Static asset paths: check file extension (with or without query string)
    # Paths like "/assets/main.js?v=123" should have extension recognized.
    path_no_query = p.split("?", 1)[0]
    _, ext = os.path.splitext(path_no_query)
    if ext.lower() in _STATIC_ASSET_EXTENSIONS:
        return True

    # Default: PROTECTED (require auth)
    return False


def auth_headers(token: Optional[str] = None) -> Dict[str, str]:
    """Authorization header dict for tests and same-process callers."""
    return {"Authorization": f"Bearer {token if token else get_bearer_token()}"}


def reset_cache() -> None:
    """Clear the in-memory file-token cache (tests only)."""
    global _cached_file_token
    _cached_file_token = None
