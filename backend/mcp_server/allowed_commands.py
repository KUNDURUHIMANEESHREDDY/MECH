"""Allowlisted command execution for the MCP control plane.

The rule this module exists to enforce
--------------------------------------
**The MCP caller chooses *what to run* from a fixed menu, never *how to run
it*.** The executable argv is always constructed here from an allowlisted
runner/target plus tightly validated arguments. No caller-supplied executable
name, flag, or absolute path ever reaches :mod:`subprocess`.

Menus:

* ``mech_test_run`` runners: ``pytest`` (backend, ``python -m pytest``) and
  ``vitest`` (frontend, ``npm run test:js -- <filters>``). Test paths must
  already exist inside the contained working directory; flags are pytest
  ``-q``, ``-x``, ``--version``, ``--collect-only`` and ``-k <expr>``,
  vitest ``-q``, ``-x``, ``--version`` and ``-k <expr>``, with a safe
  expression pattern on the ``-k`` value.
* ``mech_build_run`` targets: ``renderer`` only
  (``npm run build:renderer`` in the frontend directory, mirroring
  ``backend.runtime.build_runner``). The working directory is fixed, not
  caller-supplied.

Every spawn uses ``shell=False``, a capped timeout, and truncated output.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

TEST_RUNNERS = ("pytest", "vitest")
BUILD_TARGETS = ("renderer",)

MAX_ARGS = 20
MAX_ARG_CHARS = 512
MAX_TIMEOUT_TEST_S = 600
MAX_TIMEOUT_BUILD_S = 900
MAX_STDOUT_CHARS = 20_000
MAX_STDERR_CHARS = 2_000

_PYTEST_FLAGS = frozenset({"-q", "-x", "--version", "--collect-only"})
_K_VALUE_RE = re.compile(r"^[\w .\-]{1,128}$")


def _frontend_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "frontend"


def _err(reason: str) -> Dict[str, Any]:
    return {"status": "error", "provenance": "unavailable", "reason": reason}


def _ok(**fields: Any) -> Dict[str, Any]:
    return {"status": "ok", "provenance": "live", **fields}


def _coerce_timeout(value: Any, cap: int) -> int:
    if isinstance(value, bool):
        raise ValueError("timeout_s must be an integer")
    try:
        timeout = int(value)
    except (TypeError, ValueError):
        raise ValueError("timeout_s must be an integer")
    return max(1, min(timeout, cap))


def _check_test_path(raw: str, base: Path) -> str:
    """Validate one positional test-path arg; return it unchanged on success."""
    if not isinstance(raw, str) or not raw.strip() or len(raw) > MAX_ARG_CHARS:
        raise ValueError(f"invalid test path: {raw!r}")
    if "\x00" in raw:
        raise ValueError(f"invalid test path: {raw!r}")
    candidate = Path(raw.strip())
    target = candidate if candidate.is_absolute() else base / candidate
    try:
        resolved = target.resolve()
    except OSError as exc:
        raise ValueError(f"test path cannot be resolved: {exc}") from exc
    try:
        resolved.relative_to(base)
    except ValueError:
        raise ValueError(f"test path escapes the working directory: {raw}")
    if not resolved.exists():
        raise ValueError(f"test path does not exist: {raw}")
    return raw


def build_test_argv(runner: str, args: List[str], base: Path) -> List[str]:
    """Construct the executable argv for a test run (pure, no spawning).

    Raises :class:`ValueError` for anything off the menu.
    """
    name = (runner or "").strip().lower()
    if name not in TEST_RUNNERS:
        raise ValueError(
            f"unknown test runner {runner!r}; allowed: {', '.join(TEST_RUNNERS)}")
    if not isinstance(args, list):
        raise ValueError("args must be a list of strings")
    if len(args) > MAX_ARGS:
        raise ValueError(f"too many args (max {MAX_ARGS})")
    for item in args:
        if not isinstance(item, str) or len(item) > MAX_ARG_CHARS:
            raise ValueError(f"invalid arg: {item!r}")

    positional: List[str] = []
    flags: List[str] = []
    i = 0
    while i < len(args):
        item = args[i]
        if item == "-k":
            if i + 1 >= len(args) or not _K_VALUE_RE.match(args[i + 1]):
                raise ValueError("-k requires a safe filter expression")
            flags.extend([item, args[i + 1]])
            i += 2
        elif item in _PYTEST_FLAGS or (name == "vitest" and item in ("-q", "-x", "--version")):
            flags.append(item)
            i += 1
        elif item.startswith("-"):
            raise ValueError(f"flag not allowed: {item}")
        else:
            positional.append(_check_test_path(item, base))
            i += 1

    if name == "pytest":
        return [sys.executable, "-m", "pytest", *flags, *positional]
    npm = shutil.which("npm")
    if npm is None:
        raise ValueError("npm binary not found on PATH")
    try:
        base.relative_to(_frontend_dir().resolve())
    except ValueError:
        raise ValueError("vitest runs inside the frontend directory")
    if flags and "--version" in flags and len(flags) > 1:
        raise ValueError("--version takes no other flags")
    if "--version" in flags:
        return [npm, "run", "test:js", "--", "--version"]
    return [npm, "run", "test:js", "--", *flags, *positional]


def build_build_argv(target: str) -> tuple[list, Path]:
    """Construct the executable argv + fixed cwd for a build (pure).

    Raises :class:`ValueError` for anything off the menu.
    """
    name = (target or "").strip().lower()
    if name not in BUILD_TARGETS:
        raise ValueError(
            f"unknown build target {target!r}; allowed: {', '.join(BUILD_TARGETS)}")
    npm = shutil.which("npm")
    if npm is None:
        raise ValueError("npm binary not found on PATH")
    frontend = _frontend_dir()
    if not frontend.is_dir():
        raise ValueError("frontend directory not found")
    return [npm, "run", "build:renderer"], frontend


def _run(argv: List[str], cwd: Path, timeout: int) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            argv, cwd=str(cwd), capture_output=True, text=True,
            timeout=timeout, shell=False)
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout if isinstance(exc.stdout, str) else ""
        err = exc.stderr if isinstance(exc.stderr, str) else ""
        return {"status": "timeout", "provenance": "live", "cmd": argv,
                "cwd": str(cwd), "stdout": out[-5_000:], "stderr": err[-MAX_STDERR_CHARS:]}
    except OSError as exc:
        return _err(f"exec failed: {exc}")
    return _ok(cmd=argv, cwd=str(cwd), returncode=proc.returncode,
               stdout=proc.stdout[-MAX_STDOUT_CHARS:],
               stderr=proc.stderr[-MAX_STDERR_CHARS:])


def run_test(runner: str, args: Optional[List[str]],
             base: Path, timeout_s: Any = 300) -> Dict[str, Any]:
    """Validate, construct, and run a test command in ``base``."""
    try:
        timeout = _coerce_timeout(timeout_s, MAX_TIMEOUT_TEST_S)
    except ValueError as exc:
        return _err(str(exc))
    if not base.is_dir():
        return _err(f"cwd is not a directory: {base}")
    try:
        argv = build_test_argv(runner, list(args or []), base)
    except ValueError as exc:
        return _err(str(exc))
    return _run(argv, base, timeout)


def run_build(target: str, timeout_s: Any = 900) -> Dict[str, Any]:
    """Validate, construct, and run a build for ``target``."""
    try:
        timeout = _coerce_timeout(timeout_s, MAX_TIMEOUT_BUILD_S)
    except ValueError as exc:
        return _err(str(exc))
    try:
        argv, cwd = build_build_argv(target)
    except ValueError as exc:
        return _err(str(exc))
    return _run(argv, cwd, timeout)
