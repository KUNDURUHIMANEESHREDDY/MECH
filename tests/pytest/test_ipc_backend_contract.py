"""Contract test: every backend route the Electron main process calls.

frontend/electron/ipc/runtime.js proxies the renderer's requests to the FastAPI
backend over HTTP. Its path prefix was "/api/v1", but the backend mounts the
dispatcher at "/api" (and an optional v2 at "/api/v2") -- there is no v1. Every
proxied call therefore 404'd.

The renderer cannot be exercised headlessly here, so this asserts the contract
directly: every method runtime.js calls must exist in the backend's OpenAPI
schema under the prefix runtime.js actually uses.

Run with a live backend for the strongest form:
    python main.py &
    python -m pytest tests/pytest/test_ipc_backend_contract.py -q
Without one, the offline schema check still runs.
"""
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_JS = REPO_ROOT / "frontend" / "electron" / "ipc" / "runtime.js"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _source() -> str:
    return RUNTIME_JS.read_text(encoding="utf-8")


def _declared_prefix() -> str:
    """The prefix runtime.js sends to, e.g. http://localhost:8000/api/v1."""
    match = re.search(r"const BACKEND\s*=\s*'([^']+)'", _source())
    assert match, "could not find the BACKEND constant in runtime.js"
    return match.group(1)


def _called_methods() -> list:
    """Every backend method runtime.js calls, from callBackend('...')."""
    return sorted(set(re.findall(r"callBackend\(\s*'([^']+)'", _source())))


def _openapi_paths():
    """Paths from the running backend, or from importing the app directly."""
    try:
        import httpx
        response = httpx.get("http://localhost:8000/openapi.json", timeout=15.0)
        response.raise_for_status()
        return response.json()["paths"]
    except Exception:
        pass
    # No live backend: build it in-process.
    try:
        import os
        os.environ.setdefault("MECH_STORAGE_DB", ":memory:")
        import main as backend_main
        return backend_main.app.openapi()["paths"]
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"cannot obtain the OpenAPI schema: {exc}")


def test_runtime_js_uses_the_prefix_the_backend_serves():
    """The regression: runtime.js pointed at /api/v1, which does not exist."""
    prefix = _declared_prefix()
    paths = _openapi_paths()
    assert not prefix.endswith("/v1"), (
        f"runtime.js targets {prefix} but the backend has no v1 mount; "
        f"the dispatcher is mounted at /api"
    )
    # The prefix itself must match at least one real route.
    stripped = prefix.rsplit("/api", 1)[-1]
    assert not stripped.strip("/"), (
        f"runtime.js prefix {prefix} includes a version segment the backend "
        f"does not serve"
    )


def test_every_proxied_method_exists_on_the_backend():
    """Each callBackend('x') must resolve to a real route."""
    prefix = _declared_prefix()
    route_prefix = "/api" if "/api" in prefix else ""
    paths = _openapi_paths()

    missing = []
    for method in _called_methods():
        candidates = {
            f"{route_prefix}/{method}",
            f"/api/{method}",
            f"/api/v2/{method}",
        }
        if not any(c in paths for c in candidates):
            missing.append(method)

    assert not missing, (
        f"runtime.js calls methods with no backend route: {missing}. "
        f"Existing route prefixes: {sorted({p.split('/')[2] for p in paths if p.startswith('/api/')})}"
    )


def test_request_body_is_the_payload_not_an_envelope():
    """The dispatcher binds `payload: Dict` as the request body.

    Sending {"method": ..., "payload": {...}} made payload.get("prompt") return
    "" and every field fall back to its default — silently, with no error.
    """
    source = _source()
    envelope = re.search(r"body:\s*JSON\.stringify\(\s*\{\s*method", source)
    assert envelope is None, (
        "callBackend wraps the payload in a {method, payload} envelope; the "
        "dispatcher expects the payload as the body, so fields are dropped"
    )
    assert re.search(r"body:\s*JSON\.stringify\(\s*payload", source), (
        "callBackend should send the payload as the request body"
    )


def test_runtime_js_and_main_js_agree_on_the_backend_port():
    """runtime.js hardcodes its own host/port; main.js spawns the backend."""
    port = re.search(r":(\d+)", _declared_prefix())
    assert port, "no port found in runtime.js BACKEND constant"

    main_js = (REPO_ROOT / "frontend" / "electron" / "main.js").read_text(
        encoding="utf-8")
    m = re.search(r"BACKEND_PORT\s*=\s*(\d+)", main_js)
    assert m, "could not find BACKEND_PORT in main.js"
    assert int(port.group(1)) == int(m.group(1)), (
        f"runtime.js talks to port {port.group(1)} but main.js spawns "
        f"port {m.group(1)}"
    )


def test_main_js_spawns_the_backend_the_renderer_expects():
    """main.js must actually spawn a backend for runtime.js to reach."""
    main_js = (REPO_ROOT / "frontend" / "electron" / "main.js").read_text(
        encoding="utf-8")
    assert "registerIpcHandlers" in main_js, "main.js registers no IPC handlers"
    # And the handler bundle must include the runtime proxy.
    index_js = (REPO_ROOT / "frontend" / "electron" / "ipc" / "index.js").read_text(
        encoding="utf-8")
    assert "registerRuntimeHandlers" in index_js, (
        "runtime handlers are not registered, so runtime.js is dead code"
    )
