"""Python sidecar for the desktop application.

Speaks a tiny JSON-lines protocol over stdio. Each line on stdin is a
request: ``{"id": <int>, "method": <str>, "payload": <dict>}``.

Each line on stdout is a response (or a ``ready`` banner at start):

    {"id": <int>, "result": <obj>}                 on success
    {"id": <int>, "error": "<message>"}            on failure
    {"type": "ready"}                              once at startup
"""

from __future__ import annotations

import json
import os
import sys
import threading
import traceback
from typing import Any, Callable, Dict
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from api.dispatcher import build_dispatcher  # noqa: E402



def _emit(obj: Dict[str, Any]) -> None:
    """Write a single JSON line and flush immediately."""
    sys.stdout.write(json.dumps(obj, default=str) + "\n")
    sys.stdout.flush()


def _safe_call(handler: Callable[[Dict[str, Any]], Any], payload: Dict[str, Any]) -> Any:
    return handler(payload or {})


def serve() -> None:
    dispatcher = build_dispatcher()

    # Signal readiness before processing requests so the parent process
    # can resolve its "ready" wait quickly.
    _emit({"type": "ready", "pid": os.getpid()})

    reader_lock = threading.Lock()

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit({"error": f"invalid_json: {exc}"})
            continue

        msg_id = message.get("id")
        method = message.get("method")
        payload = message.get("payload") or {}

        if not isinstance(msg_id, int) or not isinstance(method, str):
            _emit({"id": msg_id, "error": "malformed_request"})
            continue

        handler = dispatcher.get(method)
        if handler is None:
            _emit({"id": msg_id, "error": f"unknown_method: {method}"})
            continue

        with reader_lock:
            try:
                result = _safe_call(handler, payload)
                _emit({"id": msg_id, "result": result})
            except Exception as exc:  # noqa: BLE001
                _emit({
                    "id": msg_id,
                    "error": str(exc) or exc.__class__.__name__,
                    "trace": traceback.format_exc(limit=4),
                })


def main() -> int:
    try:
        serve()
    except KeyboardInterrupt:
        return 0
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
