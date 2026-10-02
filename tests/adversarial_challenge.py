"""Adversarial Challenge Test Suite for Milestone 1.

Tests:
1. Entry point parity: backend/main.py vs root main.py.
2. Routing parity: both /api and /api/v1 routes functional and identical.
3. CORS configuration: exhaustive matrix of origins (allowed, regex, disallowed, malformed, preflight).
4. Dynamic CORS env variable configuration.
"""
import importlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Ensure repository root is strictly at index 0 of sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path = [p for p in sys.path if p not in (str(REPO_ROOT), str(REPO_ROOT / "backend"))]
sys.path.insert(0, str(REPO_ROOT))

import main
import backend.main
from fastapi.testclient import TestClient

results: List[Dict[str, str]] = []


def record_result(category: str, test_name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    results.append({
        "category": category,
        "test": test_name,
        "status": status,
        "detail": detail,
    })
    print(f"[{status}] {category} :: {test_name} {('- ' + detail) if detail else ''}")


def extract_all_routes(fastapi_app) -> List[Tuple[str, Tuple[str, ...]]]:
    routes = []
    for r in fastapi_app.routes:
        if hasattr(r, "path"):
            methods = tuple(sorted(getattr(r, "methods", []) or []))
            routes.append((r.path, methods))
        elif hasattr(r, "include_context") and hasattr(r, "original_router"):
            prefix = getattr(r.include_context, "prefix", "")
            for sub in r.original_router.routes:
                sub_path = prefix.rstrip("/") + "/" + getattr(sub, "path", "").lstrip("/")
                sub_methods = tuple(sorted(getattr(sub, "methods", []) or []))
                routes.append((sub_path, sub_methods))
    return sorted(routes)


def run_entry_point_parity_tests():
    print("\n=== Challenge 1: Entry Point Parity ===")
    
    # 1. Object Identity
    is_identical = (main.app is backend.main.app)
    record_result(
        "Parity",
        "app_instance_identity",
        is_identical,
        f"main.app is backend.main.app: {is_identical}",
    )
    
    # 2. Metadata Parity
    meta_equal = (
        main.app.title == backend.main.app.title == "MECH Research Platform"
        and main.app.version == backend.main.app.version == "2.0.0"
        and main.app.description == backend.main.app.description
    )
    record_result("Parity", "app_metadata_parity", meta_equal, f"Title: {main.app.title}, Version: {main.app.version}")
    
    # 3. Route Table Comparison
    routes_main = extract_all_routes(main.app)
    routes_backend = extract_all_routes(backend.main.app)
    routes_identical = (routes_main == routes_backend)
    record_result(
        "Parity",
        "route_table_equality",
        routes_identical,
        f"main routes: {len(routes_main)}, backend routes: {len(routes_backend)}",
    )
    
    # 4. Lifespan Handler Parity
    lifespan_attached = (main.app.router.lifespan_context is backend.main.app.router.lifespan_context)
    record_result(
        "Parity",
        "lifespan_context_parity",
        lifespan_attached,
        f"Context manager: {main.app.router.lifespan_context}",
    )
    
    # 5. Root and Health Endpoints
    with TestClient(main.app) as client:
        res_root = client.get("/")
        res_health = client.get("/health")
        root_ok = (res_root.status_code == 200 and res_root.json().get("name") == "MECH Platform")
        health_ok = (res_health.status_code == 200 and res_health.json().get("status") == "healthy")
        record_result("Parity", "get_root_via_main", root_ok, f"Status: {res_root.status_code}, Body: {res_root.json()}")
        record_result("Parity", "get_health_via_main", health_ok, f"Status: {res_health.status_code}, Body: {res_health.json()}")


def run_route_functionality_tests():
    print("\n=== Challenge 2: Route Functionality & Parity (/api vs /api/v1) ===")
    
    # Analyze registered routes
    all_routes = extract_all_routes(backend.main.app)
    all_paths = [r[0] for r in all_routes]
    
    api_paths = {p[len("/api"):] for p in all_paths if p.startswith("/api/") and not p.startswith("/api/v1/")}
    api_v1_paths = {p[len("/api/v1"):] for p in all_paths if p.startswith("/api/v1/")}
    
    # Check if route subpaths match 1:1
    subpaths_match = (api_paths == api_v1_paths and len(api_paths) > 0)
    record_result(
        "Routing",
        "route_path_symmetry",
        subpaths_match,
        f"Paths under /api: {len(api_paths)}, Paths under /api/v1: {len(api_v1_paths)}",
    )

    # Dynamic empirical probing of all static GET endpoints under /api and /api/v1
    static_get_subpaths = [
        p for p in sorted(api_paths)
        if "{" not in p and p in [
            "/status", "/models", "/research_catalog", "/experiments",
            "/circuits", "/benchmarks", "/knowledge-graph", "/portal/summary",
            "/discoveries", "/sessions", "/runtime/status", "/runtime/engines"
        ]
    ]

    with TestClient(backend.main.app) as client:
        for subpath in static_get_subpaths:
            p_api = f"/api{subpath}"
            p_v1 = f"/api/v1{subpath}"
            res_api = client.get(p_api)
            res_v1 = client.get(p_v1)
            
            equal = (
                res_api.status_code == res_v1.status_code
                and res_api.json() == res_v1.json()
            )
            record_result(
                "Routing",
                f"get_{subpath.strip('/').replace('/', '_')}_parity",
                equal,
                f"/api: {res_api.status_code}, /api/v1: {res_v1.status_code}",
            )

        # POST /ping vs /v1/ping
        r_api_ping = client.post("/api/ping")
        r_v1_ping = client.post("/api/v1/ping")
        ping_equal = (
            r_api_ping.status_code == 200
            and r_v1_ping.status_code == 200
            and r_api_ping.json() == r_v1_ping.json() == {"status": "ok"}
        )
        record_result("Routing", "post_ping_parity", ping_equal, f"/api: {r_api_ping.json()}, /api/v1: {r_v1_ping.json()}")

        # 404 Parity on unknown routes
        r_api_404 = client.get("/api/nonexistent_route_xyz")
        r_v1_404 = client.get("/api/v1/nonexistent_route_xyz")
        not_found_equal = (r_api_404.status_code == 404 and r_v1_404.status_code == 404)
        record_result("Routing", "unknown_route_404_parity", not_found_equal, f"/api: {r_api_404.status_code}, /api/v1: {r_v1_404.status_code}")


def run_cors_stress_tests():
    print("\n=== Challenge 3: CORS Configuration Stress Testing ===")

    test_cases = [
        # (Category, Origin, ExpectedAllowed, Description)
        ("Allowed-Base", "http://localhost:5173", True, "Vite dev server default"),
        ("Allowed-Base", "http://127.0.0.1:5173", True, "Vite dev server IP"),
        ("Allowed-Base", "http://localhost:3000", True, "Desktop shell default"),
        ("Allowed-Base", "http://127.0.0.1:3000", True, "Desktop shell IP"),
        ("Allowed-Base", "null", True, "Electron sandboxed / file null origin"),
        ("Allowed-Base", "file://", True, "Electron file protocol"),
        
        # Regex allowed (arbitrary ports on localhost / 127.0.0.1)
        ("Allowed-Regex", "http://localhost:8080", True, "Localhost alternative port"),
        ("Allowed-Regex", "http://127.0.0.1:9000", True, "127.0.0.1 alternative port"),
        ("Allowed-Regex", "http://localhost:4173", True, "Vite preview port"),
        ("Allowed-Regex", "https://localhost:443", True, "Localhost HTTPS port"),
        ("Allowed-Regex", "https://127.0.0.1:8443", True, "127.0.0.1 HTTPS port"),
        ("Allowed-Regex", "http://localhost", True, "Localhost no port"),
        ("Allowed-Regex", "http://127.0.0.1", True, "127.0.0.1 no port"),

        # Disallowed external / attack vectors
        ("Disallowed", "http://evil.com", False, "External untrusted domain"),
        ("Disallowed", "https://attacker.org", False, "External HTTPS domain"),
        ("Disallowed", "http://localhost.evil.com", False, "Subdomain spoofing localhost"),
        ("Disallowed", "http://127.0.0.1.attacker.com", False, "Subdomain spoofing 127.0.0.1"),
        ("Disallowed", "http://evil-localhost", False, "Host prefix spoofing"),
        ("Disallowed", "http://evil-127.0.0.1", False, "Host prefix spoofing"),
        ("Disallowed", "http://localhost.attacker:5173", False, "Host spoofing with port"),
        ("Disallowed", "file://evil.com", False, "Fake file origin with hostname"),

        # Malformed origins
        ("Malformed", "", False, "Empty origin header"),
        ("Malformed", "   ", False, "Whitespace origin"),
        ("Malformed", "http://", False, "Incomplete scheme"),
        ("Malformed", "javascript:alert(1)", False, "Javascript pseudo-protocol"),
        ("Malformed", "data:text/html,test", False, "Data URI"),
        ("Malformed", "http://localhost:abc", False, "Non-numeric port"),
    ]

    with TestClient(backend.main.app) as client:
        for cat, origin, expected_allowed, desc in test_cases:
            headers = {"Origin": origin} if origin else {}
            # Probe GET /health
            res = client.get("/health", headers=headers)
            acao = res.headers.get("access-control-allow-origin")
            
            if expected_allowed:
                passed = (acao == origin)
                detail = f"Origin: '{origin}' -> ACAO: '{acao}' (Expected '{origin}')"
            else:
                passed = (acao is None or acao != origin)
                detail = f"Origin: '{origin}' -> ACAO: '{acao}' (Expected None/Disallowed)"
            
            record_result("CORS", f"{cat}::{desc}", passed, detail)

        # Preflight OPTIONS Tests
        print("\n  -- Preflight OPTIONS Tests --")
        preflight_headers = {
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type, Authorization",
        }
        res_opt = client.options("/api/ping", headers=preflight_headers)
        opt_acao = res_opt.headers.get("access-control-allow-origin")
        opt_acac = res_opt.headers.get("access-control-allow-credentials")
        opt_acam = res_opt.headers.get("access-control-allow-methods")
        opt_acah = res_opt.headers.get("access-control-allow-headers")
        
        preflight_ok = (
            res_opt.status_code == 200
            and opt_acao == "http://localhost:5173"
            and opt_acac == "true"
            and (opt_acam is not None and ("POST" in opt_acam.upper() or opt_acam == "*"))
            and (opt_acah is not None)
        )
        record_result(
            "CORS",
            "preflight_allowed_origin",
            preflight_ok,
            f"Status: {res_opt.status_code}, ACAO: {opt_acao}, ACAC: {opt_acac}, ACAM: {opt_acam}",
        )

        # Forbidden Preflight
        preflight_forbidden = {
            "Origin": "http://evil.com",
            "Access-Control-Request-Method": "POST",
        }
        res_opt_bad = client.options("/api/ping", headers=preflight_forbidden)
        bad_acao = res_opt_bad.headers.get("access-control-allow-origin")
        record_result(
            "CORS",
            "preflight_disallowed_origin",
            bad_acao is None,
            f"Status: {res_opt_bad.status_code}, ACAO: {bad_acao} (Expected None)",
        )


def run_dynamic_cors_env_test():
    print("\n=== Challenge 3b: Dynamic CORS MECH_CORS_ORIGINS Test ===")
    
    test_env_origins = "https://custom.mech.internal, http://dev.local:9999/, http://test-researcher.org"
    os.environ["MECH_CORS_ORIGINS"] = test_env_origins
    
    try:
        # Reload backend.main to test dynamic origin registration
        reloaded_backend = importlib.reload(backend.main)
        
        with TestClient(reloaded_backend.app) as client:
            res1 = client.get("/health", headers={"Origin": "https://custom.mech.internal"})
            res2 = client.get("/health", headers={"Origin": "http://dev.local:9999"})
            res3 = client.get("/health", headers={"Origin": "http://test-researcher.org"})
            res_evil = client.get("/health", headers={"Origin": "http://evil.com"})

            ok1 = (res1.headers.get("access-control-allow-origin") == "https://custom.mech.internal")
            ok2 = (res2.headers.get("access-control-allow-origin") == "http://dev.local:9999")
            ok3 = (res3.headers.get("access-control-allow-origin") == "http://test-researcher.org")
            ok_evil = (res_evil.headers.get("access-control-allow-origin") is None)
            
            all_env_ok = (ok1 and ok2 and ok3 and ok_evil)
            record_result(
                "CORS-Env",
                "dynamic_mech_cors_origins_parsing",
                all_env_ok,
                f"custom: {ok1}, dev.local: {ok2}, test-researcher: {ok3}, evil: {ok_evil}",
            )
    finally:
        os.environ.pop("MECH_CORS_ORIGINS", None)
        # Restore original backend.main
        importlib.reload(backend.main)


if __name__ == "__main__":
    print("=" * 60)
    print("MECH PLATFORM ADVERSARIAL CHALLENGE SUITE (M1)")
    print("=" * 60)
    
    run_entry_point_parity_tests()
    run_route_functionality_tests()
    run_cors_stress_tests()
    run_dynamic_cors_env_test()
    
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    
    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed}/{total} tests passed ({failed} failures)")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)
    sys.exit(0)
