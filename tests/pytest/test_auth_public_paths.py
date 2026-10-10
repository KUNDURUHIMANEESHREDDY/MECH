"""F-04: Auth defaults to protected, explicit allowlist for static assets.

The old `is_public_path` defaulted to `True` (public) for any path not
explicitly listed as protected. This meant a typo in a protected prefix,
or a new endpoint added without updating the list, would be publicly
accessible.

The fix:
1. Default to `False` (protected)
2. Explicitly allow only:
   - `OPTIONS` (CORS preflight)
   - `/` and `/health` (liveness)
   - Static asset paths (file extensions typical of Vite-built frontend)

Static asset extensions are based on what Vite outputs: `.js`, `.css`, `.ico`,
`.png`, `.svg`, `.woff`, `.woff2`, `.ttf`, `.eot`, `.map`, `.json` (manifest),
`.webmanifest`, `.txt` (robots.txt), `.xml` (sitemap), `.ico`, `.png`, `.jpg`,
`.jpeg`, `.gif`, `.webp`, `.avif`.
"""
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.core import auth as auth_mod  # noqa: E402


def test_is_public_path_defaults_to_protected():
    """Unknown paths should be protected (401), not public (200).

    This is the core fix: default to protected, not public.
    """
    # These are NOT in the explicit allowlist, so they must be protected
    assert auth_mod.is_public_path("/unknown/path", "GET") is False
    assert auth_mod.is_public_path("/some/new/endpoint", "POST") is False
    assert auth_mod.is_public_path("/api/v2/newthing", "GET") is False


def test_is_public_path_explicitly_allows_known_public():
    """Explicitly allowed paths remain public."""
    assert auth_mod.is_public_path("/", "GET") is True
    assert auth_mod.is_public_path("/health", "GET") is True
    assert auth_mod.is_public_path("/health", "HEAD") is True


def test_is_public_path_options_always_allowed():
    """CORS preflight (OPTIONS) is always allowed, even on protected paths."""
    assert auth_mod.is_public_path("/api/status", "OPTIONS") is True
    assert auth_mod.is_public_path("/mcp", "OPTIONS") is True
    assert auth_mod.is_public_path("/api/v1/anything", "OPTIONS") is True


def test_is_public_path_static_asset_extensions():
    """Static asset file extensions are explicitly allowed.

    Vite outputs assets with these extensions. They must be public so the
    frontend can load without auth before it knows the token.
    """
    # JavaScript / CSS / Source maps
    assert auth_mod.is_public_path("/assets/main.js", "GET") is True
    assert auth_mod.is_public_path("/assets/main.css", "GET") is True
    assert auth_mod.is_public_path("/assets/main.js.map", "GET") is True

    # Fonts
    assert auth_mod.is_public_path("/fonts/inter.woff2", "GET") is True
    assert auth_mod.is_public_path("/fonts/inter.woff", "GET") is True
    assert auth_mod.is_public_path("/fonts/inter.ttf", "GET") is True
    assert auth_mod.is_public_path("/fonts/inter.eot", "GET") is True

    # Images
    assert auth_mod.is_public_path("/logo.png", "GET") is True
    assert auth_mod.is_public_path("/icon.svg", "GET") is True
    assert auth_mod.is_public_path("/favicon.ico", "GET") is True
    assert auth_mod.is_public_path("/og-image.jpg", "GET") is True
    assert auth_mod.is_public_path("/image.webp", "GET") is True
    assert auth_mod.is_public_path("/image.avif", "GET") is True

    # Manifest / manifest-like
    assert auth_mod.is_public_path("/manifest.json", "GET") is True
    assert auth_mod.is_public_path("/site.webmanifest", "GET") is True

    # Other common static types
    assert auth_mod.is_public_path("/robots.txt", "GET") is True
    assert auth_mod.is_public_path("/sitemap.xml", "GET") is True

    # Query strings on static assets should still be allowed
    assert auth_mod.is_public_path("/assets/main.js?v=123", "GET") is True
    assert auth_mod.is_public_path("/style.css?version=1", "GET") is True


def test_is_public_path_protected_paths_still_protected():
    """Explicitly protected paths remain protected."""
    assert auth_mod.is_public_path("/health/subsystems", "GET") is False
    assert auth_mod.is_public_path("/api/status", "GET") is False
    assert auth_mod.is_public_path("/api/v1/ping", "GET") is False
    assert auth_mod.is_public_path("/api/v1/ping", "POST") is False
    assert auth_mod.is_public_path("/mcp", "POST") is False
    assert auth_mod.is_public_path("/mcp/", "POST") is False
    assert auth_mod.is_public_path("/openapi.json", "GET") is False
    assert auth_mod.is_public_path("/docs", "GET") is False
    assert auth_mod.is_public_path("/redoc", "GET") is False


def test_is_public_path_without_leading_slash():
    """Paths without leading slash are normalized."""
    assert auth_mod.is_public_path("health", "GET") is True
    assert auth_mod.is_public_path("api/status", "GET") is False


def test_is_public_path_method_variations():
    """Method variations on protected paths stay protected."""
    for method in ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"]:
        assert auth_mod.is_public_path("/api/status", method) is False
        assert auth_mod.is_public_path("/health/subsystems", method) is False


def test_is_public_path_case_insensitive_method():
    """HTTP method matching is case-insensitive."""
    assert auth_mod.is_public_path("/api/status", "get") is False
    assert auth_mod.is_public_path("/api/status", "Get") is False
    assert auth_mod.is_public_path("/api/status", "options") is True
    assert auth_mod.is_public_path("/api/status", "Options") is True


def test_is_public_path_empty_path():
    """Empty path defaults to root."""
    assert auth_mod.is_public_path("", "GET") is True
    assert auth_mod.is_public_path("   ", "GET") is True


# ── Integration test: static assets work in the real app ───────────────── #

def test_static_asset_served_without_auth(monkeypatch):
    """Integration: a static asset path returns 200 without auth."""
    monkeypatch.setenv("MECH_API_TOKEN", "test-token")
    from backend.main import app
    from backend.core import auth as auth_mod
    from fastapi.testclient import TestClient

    auth_mod.reset_cache()
    try:
        with TestClient(app) as client:
            # This would be a static asset path - should be public
            r = client.get("/assets/main.js")
            # We don't have the file, so it's 404 from StaticFiles, not 401
            assert r.status_code != 401, f"Static asset blocked by auth: {r.status_code}"
    finally:
        auth_mod.reset_cache()


def test_api_still_requires_auth(monkeypatch):
    """Integration: API endpoints still require auth."""
    monkeypatch.setenv("MECH_API_TOKEN", "test-token")
    from backend.main import app
    from backend.core import auth as auth_mod
    from fastapi.testclient import TestClient

    auth_mod.reset_cache()
    try:
        with TestClient(app) as client:
            r = client.get("/api/status")
            assert r.status_code == 401
    finally:
        auth_mod.reset_cache()