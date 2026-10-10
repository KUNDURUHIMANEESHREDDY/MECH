"""Route Authorization Matrix Test.

Enumerates every registered route and verifies authentication/authorization behavior
for anonymous, invalid-token, and valid-token requests. This ensures no endpoint
is accidentally left unprotected.
"""

import pytest
from fastapi.testclient import TestClient

from backend.core import auth as auth_mod


AUTH_HEADERS = auth_mod.auth_headers()
INVALID_AUTH_HEADERS = {"Authorization": "Bearer invalid-token-12345"}


def _client():
    from backend.main import app
    return TestClient(app, raise_server_exceptions=False)


def test_openapi_spec_enumerates_all_routes():
    """Fetch OpenAPI spec and verify it documents all routes."""
    client = _client()
    res = client.get("/openapi.json", headers=AUTH_HEADERS)
    assert res.status_code == 200
    spec = res.json()

    paths = spec.get("paths", {})
    assert len(paths) > 50, "Expected substantial endpoint coverage"

    # Build route list (just verify structure, not returned)
    for path, methods in paths.items():
        for method, detail in methods.items():
            assert "summary" in detail or "operationId" in detail


# Known public paths that should NOT require auth
PUBLIC_PATHS = {
    ("GET", "/"),
    ("GET", "/health"),
    ("GET", "/openapi.json"),
    ("OPTIONS", "/"),  # CORS preflight
    ("OPTIONS", "/api/ping"),
    ("OPTIONS", "/api/v1/ping"),
}

# Paths that are known to be under /mcp or /docs and require auth
AUTH_REQUIRED_PREFIXES = (
    "/api/",
    "/api/v1/",
    "/mcp",
    "/mcp/",
    "/health/subsystems",
    "/docs",
    "/docs/",
    "/redoc",
    "/redoc/",
)


def is_public_path(path: str, method: str) -> bool:
    """Check if a path/method combination is expected to be public."""
    return (method.upper(), path) in PUBLIC_PATHS


def test_route_authorization_matrix():
    """
    Verify authentication behavior for every registered route.

    For each route:
    1. Anonymous request -> 401 (unless public)
    2. Invalid token -> 401
    3. Valid token -> 2xx, 4xx (validation), or 404 (not found) but NOT 500
    """
    client = _client()
    res = client.get("/openapi.json", headers=AUTH_HEADERS)
    assert res.status_code == 200
    spec = res.json()

    paths = spec.get("paths", {})

    failures = []

    for path, methods in paths.items():
        for method, detail in methods.items():
            method = method.upper()
            if method not in ("GET", "POST", "PUT", "PATCH", "DELETE"):
                continue

            is_public = is_public_path(path, method)

            # 1. Anonymous request
            try:
                anon_res = client.request(method, path)
            except Exception as e:
                failures.append(f"{method} {path}: anonymous request raised {e}")
                continue

            if is_public:
                assert anon_res.status_code < 400, (
                    f"{method} {path}: public path returned {anon_res.status_code}"
                )
            else:
                assert anon_res.status_code == 401, (
                    f"{method} {path}: expected 401 for anonymous, got {anon_res.status_code}"
                )

            # 2. Invalid token
            try:
                invalid_res = client.request(method, path, headers=INVALID_AUTH_HEADERS)
            except Exception as e:
                failures.append(f"{method} {path}: invalid token request raised {e}")
                continue

            if is_public:
                assert invalid_res.status_code < 400, (
                    f"{method} {path}: public path with invalid token returned {invalid_res.status_code}"
                )
            else:
                assert invalid_res.status_code == 401, (
                    f"{method} {path}: expected 401 for invalid token, got {invalid_res.status_code}"
                )

            # 3. Valid token - must not 500
            try:
                valid_res = client.request(method, path, headers=AUTH_HEADERS)
            except Exception as e:
                failures.append(f"{method} {path}: valid token request raised {e}")
                continue

            if is_public:
                assert valid_res.status_code < 400, (
                    f"{method} {path}: public path with valid token returned {valid_res.status_code}"
                )
            else:
                # Valid token: 2xx (success), 4xx (validation/not found), but NOT 500
                assert valid_res.status_code < 500, (
                    f"{method} {path}: valid token returned server error {valid_res.status_code}: {valid_res.text[:200]}"
                )

    assert not failures, "Authorization matrix failures:\n" + "\n".join(failures)


def test_critical_endpoints_have_correct_auth():
    """Critical endpoints that must have specific auth behavior."""
    client = _client()

    # These MUST require auth
    auth_required = [
        ("GET", "/api/status"),
        ("POST", "/api/ping"),
        ("POST", "/api/gpt2/load"),
        ("POST", "/api/gpt2/run_prompt"),
        ("POST", "/api/gpt2/attention_head"),
        ("POST", "/api/gpt2/activations"),
        ("POST", "/api/gpt2/neuron"),
        ("POST", "/api/gpt2/ioi"),
        ("POST", "/api/gpt2/patch_head"),
        ("POST", "/api/gpt2/patch_neuron"),
        ("POST", "/api/gpt2/logit_lens_all"),
        ("GET", "/api/models"),
        ("GET", "/api/circuits"),
        ("GET", "/api/experiments"),
        ("GET", "/api/sessions"),
        ("GET", "/api/knowledge-graph"),
        ("GET", "/api/agents"),
        ("POST", "/api/society/run"),
        ("GET", "/api/society/runs"),
        ("POST", "/api/benchmarks/run"),
        ("GET", "/api/benchmarks"),
        ("POST", "/api/runtime/analyze_tokens"),
        ("GET", "/api/runtime/engines"),
        ("POST", "/api/runtime/status"),
    ]

    for method, path in auth_required:
        # Anonymous -> 401
        anon = getattr(client, method.lower())(path)
        assert anon.status_code == 401, f"{method} {path}: anonymous should 401, got {anon.status_code}"

        # Invalid token -> 401
        invalid = getattr(client, method.lower())(path, headers=INVALID_AUTH_HEADERS)
        assert invalid.status_code == 401, f"{method} {path}: invalid token should 401, got {invalid.status_code}"

        # Valid token -> not 500
        valid = getattr(client, method.lower())(path, headers=AUTH_HEADERS)
        assert valid.status_code < 500, f"{method} {path}: valid token should not 500, got {valid.status_code}: {valid.text[:200]}"


def test_public_endpoints_remain_public():
    """Endpoints that must remain accessible without auth."""
    client = _client()

    # /openapi.json requires auth per auth.py is_public_path() - it's not in the public list
    public = [
        ("GET", "/"),
        ("GET", "/health"),
    ]

    for method, path in public:
        for headers in [None, INVALID_AUTH_HEADERS, AUTH_HEADERS]:
            res = getattr(client, method.lower())(path, headers=headers)
            assert res.status_code < 400, (
                f"{method} {path}: public endpoint failed with {headers}: {res.status_code}"
            )


def test_cors_preflight_always_allows_options():
    """OPTIONS requests must be allowed (CORS preflight carries no auth)."""
    client = _client()

    # Test a few protected endpoints
    protected = ["/api/status", "/api/gpt2/load", "/api/ping"]
    for path in protected:
        res = client.options(path, headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        })
        assert res.status_code == 200, f"OPTIONS {path}: should 200, got {res.status_code}"
        assert res.headers.get("access-control-allow-origin") == "http://localhost:5173"