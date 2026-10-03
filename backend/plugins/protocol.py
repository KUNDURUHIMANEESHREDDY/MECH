"""Line-delimited JSON protocol between the backend and a plugin worker.

Deliberately **not** ``multiprocessing.Pipe``. That channel pickles whatever
the child sends, so a hostile plugin in the child could execute arbitrary code
in the parent during deserialisation — a second RCE path that the process
boundary would otherwise close. Here both directions are plain UTF-8 JSON, so
a compromised worker can at worst send malformed or oversized JSON, which
:func:`decode` rejects.

Framing is one JSON object per ``\\n``-terminated line, with a hard byte cap so
a worker cannot exhaust the parent's memory by streaming without newlines.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

# 8 MiB. Hook payloads are experiment plans, campaign records, and health
# reports; anything larger is a bug or an attempt to exhaust the parent.
MAX_MESSAGE_BYTES = 8 * 1024 * 1024


class ProtocolError(Exception):
    """Raised when a message is malformed, oversized, or not an object."""


def encode(message: Dict[str, Any]) -> bytes:
    """Serialise one message to a newline-terminated JSON frame."""
    try:
        raw = json.dumps(message, default=str).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"message is not JSON-serialisable: {exc}") from exc
    if len(raw) > MAX_MESSAGE_BYTES:
        raise ProtocolError(
            f"message of {len(raw)} bytes exceeds the {MAX_MESSAGE_BYTES} byte cap"
        )
    return raw + b"\n"


def decode(line: bytes) -> Dict[str, Any]:
    """Parse one JSON frame, enforcing the size cap and object shape."""
    if len(line) > MAX_MESSAGE_BYTES:
        raise ProtocolError(
            f"incoming message of {len(line)} bytes exceeds the "
            f"{MAX_MESSAGE_BYTES} byte cap"
        )
    try:
        parsed = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProtocolError(f"malformed JSON frame: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ProtocolError("frame must be a JSON object")
    return parsed


# -- message constructors --------------------------------------------------- #

def ready(manifest: Dict[str, Any], limits: Dict[str, Any]) -> Dict[str, Any]:
    return {"event": "ready", "manifest": manifest, "limits": limits}


def loaded(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return {"event": "loaded", "manifest": manifest}


def failed(stage: str, error: str) -> Dict[str, Any]:
    return {"event": "failed", "stage": stage, "error": error[:2000]}


def reply(msg_id: int, ok: bool, result: Any = None, error: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {"id": msg_id, "ok": ok}
    if ok:
        out["result"] = result
    else:
        out["error"] = error[:2000]
    return out


def call(msg_id: int, hook: str, args: list) -> Dict[str, Any]:
    return {"id": msg_id, "op": "call", "hook": hook, "args": args}


def shutdown() -> Dict[str, Any]:
    return {"op": "shutdown"}
