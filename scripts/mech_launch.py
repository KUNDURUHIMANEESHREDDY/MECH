"""Single launcher implementation for the MECH standalone build.

Copied into the standalone folder by `scripts/build_standalone.py`; both
`Start-MECH.bat` and `Start-MECH.ps1` exist only to locate a Python and run
this file.

Why one implementation
----------------------
There used to be two, and they had *diverged*. The batch launcher checked
`if errorlevel 1` after pip; the PowerShell one did not check `$LASTEXITCODE`
at all, so a failed install wrote `.deps-installed` anyway and every subsequent
launch skipped installation and failed later, somewhere unrelated, with a
message about a missing module. Two copies of a launcher is two places for the
fix to land in one and not the other.

The defects this file addresses, in the order they bite
-------------------------------------------------------
1. **A failed install marked itself complete.** The marker is now written only
   after pip returns 0, and it stores the sha256 of `requirements.txt` rather
   than the word "ok" -- so editing the requirements reinstalls, which a boolean
   marker cannot do.
2. **Global site-packages.** Dependencies go into a local `.venv` inside the
   standalone folder. The old launcher installed into whatever Python the user
   happened to have, which made the build non-portable and could break an
   unrelated project on the same machine.
3. **"Python exists" is not "Python is new enough".** `where python` succeeds on
   3.6. The version is compared numerically and names what is required.
4. **The browser opened before the server existed.** `start "" http://...` fired
   immediately after launching Python, so a slow start produced
   `ERR_CONNECTION_REFUSED` in the user's face. `/health` is polled first.

Every step is a separate function returning a value or raising, so the launcher
can be tested without starting a server or opening a browser.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Optional, Tuple

HERE = Path(__file__).resolve().parent

#: The oldest interpreter this code runs on. Compared numerically -- an earlier
#: launcher accepted any `python` that merely existed.
REQUIRED_PYTHON = (3, 11)

HOST = "127.0.0.1"
PORT = 8000
BASE_URL = f"http://{HOST}:{PORT}"
HEALTH_URL = f"{BASE_URL}/health"

#: How long to wait for the backend to answer /health. A cold first start loads
#: torch and the FastAPI app; 60s is generous rather than tight, because the
#: failure mode of being too tight is a spurious error in front of a user who
#: is waiting on a legitimate start.
HEALTH_TIMEOUT_S = 60.0
HEALTH_INTERVAL_S = 0.5

VENV_DIR = HERE / ".venv"
DEPS_MARKER = HERE / ".deps-sha256"


class LaunchError(RuntimeError):
    """A step failed in a way the user needs told about."""


# ── Interpreter ─────────────────────────────────────────────────────────

def check_python_version(version_info: Optional[Tuple[int, int]] = None) -> str:
    """Raise unless this interpreter is new enough. Returns the version string.

    `version_info` is injectable so the comparison can be tested against 3.6 and
    3.12 without those interpreters existing.
    """
    info = version_info or sys.version_info[:2]
    required = ".".join(str(n) for n in REQUIRED_PYTHON)
    if tuple(info[:2]) < REQUIRED_PYTHON:
        found = ".".join(str(n) for n in info[:2])
        raise LaunchError(
            f"MECH needs Python {required} or newer; this is {found}. "
            f"Install a newer Python and run this launcher again.")
    return ".".join(str(n) for n in info[:2])


def venv_python(root: Path = HERE) -> Path:
    """Path to the venv interpreter, or None if there is no venv yet."""
    if sys.platform == "win32":
        candidate = root / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = root / ".venv" / "bin" / "python"
    return candidate if candidate.exists() else None


def ensure_venv(root: Path = HERE, log=print) -> Path:
    """Create the local venv if absent and return its interpreter path.

    Local rather than global: the old launcher ran `pip install -r
    requirements.txt` against the user's own interpreter, which modified shared
    site-packages and made the standalone folder depend on machine state it did
    not carry.
    """
    existing = venv_python(root)
    if existing is not None:
        return existing

    root.mkdir(parents=True, exist_ok=True)
    log(f"[mech] creating a local virtual environment in {root/'.venv'}")
    result = subprocess.run(
        [sys.executable, "-m", "venv", str(root / ".venv")],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise LaunchError(
            "could not create the local virtual environment:\n"
            f"{(result.stderr or result.stdout or '').strip()}")
    created = venv_python(root)
    if created is None:
        raise LaunchError(
            f"virtual environment created but no interpreter at {root/'.venv'}")
    return created


# ── Dependencies ────────────────────────────────────────────────────────

def requirements_digest(requirements: Optional[Path] = None) -> str:
    """sha256 of requirements.txt, or "" when there is nothing to install."""
    path = requirements or (HERE / "requirements.txt")
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def read_marker(path: Path = DEPS_MARKER) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def write_marker(digest: str, path: Path = DEPS_MARKER) -> None:
    path.write_text(digest + "\n", encoding="utf-8")


def deps_are_current(requirements: Optional[Path] = None,
                     marker: Path = DEPS_MARKER) -> bool:
    """Whether the recorded install still matches the requirements.

    The marker holds the requirements digest, not a boolean. A boolean records
    only that an install happened once, so editing `requirements.txt` afterwards
    left the old set installed and the launcher happily skipped the fix.
    """
    digest = requirements_digest(requirements)
    return bool(digest) and read_marker(marker) == digest


def install_dependencies(python: Path, requirements: Optional[Path] = None,
                         marker: Path = DEPS_MARKER, log=print) -> bool:
    """Install requirements and record the digest. Returns True if installed.

    The marker is written only after pip exits 0. This is the whole defect: the
    PowerShell launcher ran pip and then wrote "ok" regardless of what pip
    returned, so a failed install was recorded as a successful one and the
    failure surfaced on a later run as an unrelated ImportError.
    """
    path = requirements or (HERE / "requirements.txt")
    digest = requirements_digest(path)
    if not digest:
        raise LaunchError(f"no readable requirements file at {path}")

    log("[mech] installing backend dependencies (first run, or requirements changed)...")
    result = subprocess.run(
        [str(python), "-m", "pip", "install", "-r", str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-12:]
        raise LaunchError(
            "dependency installation failed:\n  " + "\n  ".join(tail))
    write_marker(digest, marker)
    return True


def ensure_dependencies(root: Path = HERE, log=print) -> Path:
    """Full dependency step: venv, then install only if the digest moved."""
    python = ensure_venv(root, log=log)
    if deps_are_current(root / "requirements.txt", root / ".deps-sha256"):
        log("[mech] dependencies already match requirements.txt")
    else:
        install_dependencies(python, root / "requirements.txt",
                             root / ".deps-sha256", log=log)
    return python


# ── Health ──────────────────────────────────────────────────────────────

def probe_health(url: str = HEALTH_URL, timeout: float = 5.0) -> Optional[dict]:
    """Return the parsed /health body, or None if it is not answering yet."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            if response.status != 200:
                return None
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return None


def wait_for_health(url: str = HEALTH_URL, timeout: float = HEALTH_TIMEOUT_S,
                    interval: float = HEALTH_INTERVAL_S, sleep=time.sleep,
                    probe=probe_health, log=print) -> dict:
    """Block until /health reports ready, or raise.

    The old launcher opened the browser immediately after spawning Python, so a
    start slower than a second produced `ERR_CONNECTION_REFUSED` -- a failure
    that looks like a broken build rather than an impatient script. `/health` is
    the backend's liveness probe and is deliberately cheap: it does not touch
    ML or storage, so it answers as soon as the app is serving.
    """
    deadline = time.monotonic() + timeout
    waited = False
    while True:
        body = probe(url)
        if body is not None and body.get("status") == "healthy":
            if waited:
                log("[mech] backend is ready")
            return body
        if time.monotonic() >= deadline:
            raise LaunchError(
                f"the backend did not answer {url} within {timeout:.0f}s. "
                f"It may still be starting, or it may have failed to start -- "
                f"run 'python main.py' in this folder to see the error.")
        if not waited:
            log("[mech] waiting for the backend to answer /health ...")
            waited = True
        sleep(interval)


# ── Running ─────────────────────────────────────────────────────────────

def start_backend(python: Path, root: Path = HERE, log=print) -> subprocess.Popen:
    """Launch the backend detached so the launcher can poll it."""
    log(f"[mech] starting the backend with {python}")
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(
        [str(python), "main.py"], cwd=str(root), **kwargs)


def open_browser(url: str = BASE_URL, opener=webbrowser.open, log=print) -> None:
    log(f"[mech] opening {url}")
    opener(url)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Start the MECH standalone backend and UI.")
    parser.add_argument("--install-only", action="store_true",
                        help="install dependencies, then exit without starting")
    parser.add_argument("--no-browser", action="store_true",
                        help="start the backend without opening a browser")
    parser.add_argument("--health-timeout", type=float, default=HEALTH_TIMEOUT_S)
    args = parser.parse_args(argv)

    try:
        check_python_version()
        python = ensure_dependencies(HERE)
        if args.install_only:
            print("[mech] dependencies are ready")
            return 0
        process = start_backend(python, HERE)
        try:
            wait_for_health(timeout=args.health_timeout)
        except LaunchError:
            # Do not leave a half-started server behind on the failure path.
            if process.poll() is None:
                process.terminate()
            raise
        if not args.no_browser:
            open_browser(BASE_URL)
        print("[mech] running. Close this window to stop MECH.")
        # Park while the backend serves, so closing the window stops the app.
        return process.wait()
    except LaunchError as exc:
        print(f"[mech] ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
