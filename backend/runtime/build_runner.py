"""Backend build runner for the Build Log view.

Runs the renderer production build (`npm run build:renderer` in the
frontend directory) in a background thread and keeps the tail of its
output. Only one build runs at a time; the allowlisted command is fixed —
callers cannot inject arbitrary commands. Localhost desktop context only.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_BUILD_TIMEOUT_SECONDS = 900
_OUTPUT_TAIL_CHARS = 6000

_lock = threading.Lock()
_state: Dict[str, Any] = {
    "status": "idle",
    "target": None,
    "started_at": None,
    "finished_at": None,
    "returncode": None,
    "output_tail": "",
}


def _frontend_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "frontend"


def status() -> Dict[str, Any]:
    with _lock:
        return dict(_state)


def _run_build() -> None:
    npm = shutil.which("npm")
    started = time.time()
    with _lock:
        _state.update({
            "status": "running",
            "target": "renderer",
            "started_at": started,
            "finished_at": None,
            "returncode": None,
            "output_tail": "",
        })
    output = ""
    returncode: Optional[int] = None
    try:
        if npm is None:
            output = "npm executable not found on PATH."
            returncode = 127
        else:
            proc = subprocess.run(
                [npm, "run", "build:renderer"],
                cwd=str(_frontend_dir()),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=_BUILD_TIMEOUT_SECONDS,
                shell=(os.name == "nt"),
            )
            output = proc.stdout or ""
            returncode = proc.returncode
    except subprocess.TimeoutExpired:
        output = f"Build timed out after {_BUILD_TIMEOUT_SECONDS} seconds."
        returncode = 124
    except Exception as exc:
        output = f"Build failed to start: {exc}"
        returncode = 125
    with _lock:
        _state.update({
            "status": "completed" if returncode == 0 else "failed",
            "finished_at": time.time(),
            "returncode": returncode,
            "output_tail": output[-_OUTPUT_TAIL_CHARS:],
        })


def start() -> Dict[str, Any]:
    with _lock:
        if _state.get("status") == "running":
            return {"accepted": False, "reason": "a build is already running"}
    worker = threading.Thread(target=_run_build, daemon=True)
    worker.start()
    return {"accepted": True, "target": "renderer"}


def history() -> List[Dict[str, Any]]:
    with _lock:
        current = dict(_state)
    if current.get("status") == "idle" and not current.get("output_tail"):
        return []
    return [current]
