"""MECH in-process backend service (no HTTP server, no ports).

Spawned by the Electron main process and driven over stdio JSON-lines
(``{id, method, params}`` in, ``{id, result}`` / ``{id, error}`` / ``{"type":"ready"}`` out).
The real FastAPI app from ``backend.main`` is executed in-process via
``fastapi.testclient.TestClient``: every route (``/api/*``, legacy dispatch,
jobs, interpretability) works exactly as it does under uvicorn, but nothing
ever binds a socket — the Electron app is the only endpoint.

Protocol (mirrors frontend/electron/python.js):
  request : {"id": <int>, "method": "http"|"ping", "params": {...}}
  response: {"id": <int>, "result": ...} | {"id": <int>, "error": "..."}
  startup:  {"type": "ready"}
"""
from __future__ import annotations

import json
import sys
import traceback
from typing import Any

from pathlib import Path  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_ROOT), str(_ROOT / "backend")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from fastapi.testclient import TestClient  # noqa: E402
from backend.main import app  # noqa: E402


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def _normalize_url(url: str) -> str:
    """TestClient wants a path; strip any host the renderer may have kept."""
    for scheme in ("http://", "https://"):
        if url.startswith(scheme):
            rest = url[len(scheme):]
            slash = rest.find("/")
            return rest[slash:] if slash >= 0 else "/"
    return url or "/"


def _handle(client: TestClient, method: str, params: dict[str, Any]) -> Any:
    if method == "ping":
        return {"ok": True, "storage": "local"}

    if method == "http":
        req_method = str(params.get("method", "GET")).upper()
        url = _normalize_url(str(params.get("path") or params.get("url") or "/"))
        headers = params.get("headers") or {}
        body = params.get("body")
        response = client.request(
            req_method,
            url,
            headers={str(k): str(v) for k, v in headers.items()} if headers else None,
            content=body if isinstance(body, str) else None,
        )
        return {
            "status": response.status_code,
            "statusText": response.reason_phrase or "",
            "headers": {str(k): str(v) for k, v in response.headers.items()},
            "body": response.text,
        }

    raise ValueError(f"Unknown method: {method}")


def main() -> int:
    # TestClient runs app startup/lifespan handlers once; then serve the loop.
    with TestClient(app) as client:
        _emit({"type": "ready"})
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            request_id = None
            try:
                request = json.loads(line)
                request_id = request.get("id")
                result = _handle(
                    client,
                    str(request.get("method", "")),
                    request.get("payload") or request.get("params") or {},
                )
                _emit({"id": request_id, "result": result})
            except Exception as error:  # IPC boundary: serialize all failures.
                traceback.print_exc(file=sys.stderr)
                _emit(
                    {
                        "id": request_id,
                        "error": f"{error.__class__.__name__}: {error}",
                    }
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())