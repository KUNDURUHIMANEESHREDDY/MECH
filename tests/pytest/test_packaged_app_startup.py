"""The packaged app must actually start and serve the API.

What this proves
----------------
The installer is the product. A build that packages a backend nobody can
start is not a release, and nothing in the tree would notice: the packaging
tests check the *configuration* (the filter excludes the token, the runtime
is staged, the backend matches the source) and the drift test checks the
source set. None of them starts the thing.

The Electron shell starts the backend in exactly one way
(``frontend/electron/main.js``, ``startBackend``):

* interpreter: ``resources/python/python.exe`` -- resolved with
  ``bundledOnly: true``, so a machine with its own Python is never used;
* entry point: ``resources/backend/main.py``;
* working directory: the resources directory;
* ``PYTHONPATH``: the resources directory and the backend directory;
* readiness: ``GET http://127.0.0.1:8000/health`` answering within 180 s.

This test reproduces that launch exactly and then asserts the API answers,
so "the installer works" stops being an assertion and becomes a measurement.

It runs against the unpacked build, so it needs no installer: the payload is
identical in both, and the payload is the part that has to work.

The test skips, loudly and with a reason, when there is no packaged app to
test. A skip is honest here; a pass without a build would not be.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"

#: The port the Electron shell hardcodes (``BACKEND_PORT`` in main.js).
BACKEND_PORT = 8000
BACKEND_HOST = "127.0.0.1"
HEALTH_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}/health"

#: The shell allows 180 s; the test allows more, because a cold build
#: machine is slower than a developer's.
STARTUP_TIMEOUT_S = 300.0

#: Imports the backend cannot start without (the same list the staging
#: script verifies).
REQUIRED_IMPORTS = (
    "fastapi",
    "uvicorn",
    "torch",
    "transformer_lens",
    "numpy",
    "pandas",
)


def _candidate_roots() -> List[Path]:
    """Where a packaged app may live, most specific first."""
    roots: List[Path] = []
    override = os.environ.get("MECH_PACKAGED_ROOT")
    if override:
        roots.append(Path(override))
    release = FRONTEND / "release"
    if release.is_dir():
        roots.extend(sorted(release.glob("*-unpacked")))
        roots.extend(sorted(release.glob("*-unpacked")))
    return roots


def _packaged_root() -> Optional[Path]:
    for candidate in _candidate_roots():
        if (candidate / "resources").is_dir():
            return candidate
    return None


def _require_packaged_root() -> Path:
    root = _packaged_root()
    if root is None:
        pytest.skip(
            "no packaged application to test. Build one with "
            "`npm --prefix frontend run build`, or set MECH_PACKAGED_ROOT to an "
            "unpacked build directory. Without a build there is nothing to "
            "prove, so this check is skipped rather than passed."
        )
    return root


def _port_is_free() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((BACKEND_HOST, BACKEND_PORT))
        except OSError:
            return False
    return True


def _terminate(process: subprocess.Popen) -> None:
    """Stop the backend and everything it started."""
    try:
        import psutil
    except Exception:
        psutil = None  # type: ignore[assignment]

    if psutil is not None:
        try:
            parent = psutil.Process(process.pid)
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except Exception:
                    pass
            parent.kill()
        except Exception:
            pass
    else:
        process.kill()

    try:
        process.wait(timeout=30)
    except Exception:
        pass


def _read_json(url: str, timeout: float = 5.0) -> Tuple[int, object]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
            return response.status, json.loads(body)
    except urllib.error.HTTPError as error:
        return error.code, None
    except Exception:
        return 0, None


def test_the_packaged_app_carries_its_own_python_runtime() -> None:
    """The shell resolves the interpreter with ``bundledOnly: true``.

    Without ``resources/python/python.exe`` the app raises
    ``BackendStartupError`` at startup, so a build without one is an
    installer that cannot open.
    """
    root = _require_packaged_root()
    interpreter = root / "resources" / "python" / "python.exe"

    assert interpreter.is_file(), (
        f"the packaged app has no bundled interpreter at {interpreter}; the "
        "Electron shell resolves the backend with bundledOnly:true and will "
        "refuse to start without one"
    )


def test_the_bundled_runtime_can_import_the_backend() -> None:
    """The bundled interpreter must have the backend's dependencies.

    A runtime that imports but cannot import ``transformer_lens`` is a
    runtime that fails at the first request, on the user's machine.
    """
    root = _require_packaged_root()
    interpreter = root / "resources" / "python" / "python.exe"
    if not interpreter.is_file():
        pytest.skip("the packaged app has no bundled interpreter")

    code = (
        "import " + ", ".join(REQUIRED_IMPORTS) + "; "
        "print('imports ok')"
    )
    result = subprocess.run(
        [str(interpreter), "-c", code],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        "the bundled interpreter cannot import the backend requirements:\n"
        f"{result.stderr.strip()[-2000:]}"
    )
    assert "imports ok" in result.stdout


def test_the_packaged_backend_starts_and_serves_the_api() -> None:
    """Launch the payload exactly as the Electron shell does, then ask it something.

    This is the check that turns "the installer was built" into "the
    installer works".
    """
    root = _require_packaged_root()
    resources = root / "resources"
    interpreter = resources / "python" / "python.exe"
    entry_point = resources / "backend" / "main.py"

    assert interpreter.is_file(), f"missing bundled interpreter: {interpreter}"
    assert entry_point.is_file(), f"missing backend entry point: {entry_point}"

    if not _port_is_free():
        pytest.skip(
            f"port {BACKEND_PORT} is already in use, so a backend is already "
            "running; starting a second one would measure the wrong process"
        )

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(resources), str(resources / "backend")]
    )
    env["PYTHONUNBUFFERED"] = "1"

    log = (ROOT / ".pytest-tmp-packaged-startup.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        [str(interpreter), str(entry_point)],
        cwd=str(resources),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.time() + STARTUP_TIMEOUT_S
        status = 0
        payload = None
        while time.time() < deadline:
            if process.poll() is not None:
                pytest.fail(
                    "the packaged backend exited during startup with code "
                    f"{process.returncode}:\n{_tail(log)}"
                )
            status, payload = _read_json(HEALTH_URL)
            if status == 200:
                break
            time.sleep(1.0)

        assert status == 200, (
            f"the packaged backend did not answer {HEALTH_URL} within "
            f"{int(STARTUP_TIMEOUT_S)}s (last status {status}):\n{_tail(log)}"
        )
        assert isinstance(payload, dict), (
            f"/health must answer a JSON object, got {type(payload).__name__}"
        )
        assert payload.get("status") == "healthy", (
            f"/health answered {payload!r}; the shell treats a non-2xx or an "
            "unexpected body as 'not ready' and reports a startup failure"
        )
    finally:
        _terminate(process)
        log.close()


def _tail(log) -> str:
    try:
        log.flush()
    except Exception:
        pass
    try:
        text = Path(log.name).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return "(no output captured)"
    return text[-4000:]
