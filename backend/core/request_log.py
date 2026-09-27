"""In-memory backend request log for the Logging view.

Records recent HTTP request/response metadata (method, path, status,
duration) in a bounded ring buffer. No request or response bodies are stored,
so prompts and payloads never enter the log.
"""

from __future__ import annotations

import threading
from collections import deque
from time import time
from typing import Any, Dict, List

_MAX_ENTRIES = 200

_lock = threading.Lock()
_entries: deque = deque(maxlen=_MAX_ENTRIES)


def record(method: str, path: str, status_code: int,
           duration_ms: float) -> None:
    entry = {
        "timestamp": time(),
        "method": str(method or "").upper()[:12],
        "path": str(path or "")[:512],
        "status": int(status_code),
        "duration_ms": round(float(duration_ms), 2),
    }
    with _lock:
        _entries.append(entry)


def list_entries(limit: int = 100) -> List[Dict[str, Any]]:
    try:
        count = max(1, min(_MAX_ENTRIES, int(limit)))
    except (TypeError, ValueError):
        count = 100
    with _lock:
        items = list(_entries)[-count:]
    return list(reversed(items))


def clear() -> int:
    with _lock:
        removed = len(_entries)
        _entries.clear()
    return removed
