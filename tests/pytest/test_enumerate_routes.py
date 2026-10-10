"""Enumerate and validate all registered OpenAPI endpoints — pytest version."""

import pytest
from fastapi.testclient import TestClient

from backend.core import auth as auth_mod


AUTH_HEADERS = auth_mod.auth_headers()


def _client():
    from backend.main import app
    return TestClient(app, raise_server_exceptions=False)


def test_openapi_endpoint_enumeration():
    """Every registered OpenAPI endpoint must be reachable and documented."""
    client = _client()

    # Fetch the OpenAPI spec (requires auth)
    res = client.get("/openapi.json", headers=AUTH_HEADERS)
    assert res.status_code == 200, "OpenAPI spec must be accessible"
    spec = res.json()

    paths = spec.get("paths", {})
    assert paths, "OpenAPI spec must contain paths"

    routes = []
    for path, methods in paths.items():
        for method, detail in methods.items():
            routes.append({
                "method": method.upper(),
                "path": path,
                "summary": detail.get("summary", ""),
                "operationId": detail.get("operationId", ""),
            })

    # Must have substantial endpoint coverage
    assert len(routes) > 50, f"Expected >50 endpoints, found {len(routes)}"

    # Every route should have a summary or operationId
    undocumented = [r for r in routes if not r["summary"] and not r["operationId"]]
    assert not undocumented, f"Undocumented routes: {undocumented}"

    # Print for visibility (pytest -s to see output)
    print(f"\nTotal Registered OpenAPI Endpoints: {len(routes)}")
    for r in sorted(routes, key=lambda x: (x["path"], x["method"])):
        print(f"{r['method']:<6} {r['path']:<50} {r['summary'] or r['operationId']}")


def test_critical_endpoints_present():
    """Critical endpoints that the platform depends on must be registered."""
    client = _client()
    res = client.get("/openapi.json", headers=AUTH_HEADERS)
    spec = res.json()
    paths = spec.get("paths", {})

    critical = [
        ("GET", "/health"),  # public
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

    missing = []
    for method, path in critical:
        if path not in paths or method.lower() not in paths[path]:
            missing.append(f"{method} {path}")

    assert not missing, f"Missing critical endpoints: {missing}"