"""Build a portable MECH standalone folder: backend + built frontend + launcher.

Output: <repo>/MECH-standalone/
  backend/            FastAPI backend (copied)
  frontend/dist/      built Vue UI (copied, served by backend at /)
  main.py             backend entry (copied)
  requirements.txt    python deps (copied)
  mech_launch.py      the launcher -- the only place the launch logic lives
  Start-MECH.bat      double-click shim: find python, run mech_launch.py
  Start-MECH.ps1      PowerPoint... PowerShell shim: same, one line
  README.txt

Usage:
    python scripts/build_standalone.py [--skip-frontend-build]

The build is atomic
-------------------
The previous version deleted the output directory and then copied into it. A
failure halfway through -- a full disk, a file locked by a running copy of the
app, an antivirus scanner -- left no standalone folder at all, having destroyed
the working one. Worse, it destroyed the working one *first*, so the failure was
not recoverable by retrying.

Now the new build is assembled in a temporary sibling directory, validated there
(including importing the copied backend), and only then swapped in. The previous
build is moved aside rather than deleted, so a failed swap can be undone.

Why the launchers are one line each
-----------------------------------
There used to be two launcher implementations, one per shell, and they had
diverged: the batch one checked pip's exit status and the PowerShell one did not,
so on PowerShell a failed install was recorded as successful. `mech_launch.py`
holds the logic and both shells only locate an interpreter. See that file.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "MECH-standalone"

#: Where the launch logic lives in the repo, and where it lands in the build.
LAUNCHER_SOURCE = REPO_ROOT / "scripts" / "mech_launch.py"
LAUNCHER_NAME = "mech_launch.py"

#: Both shells now do one thing: find a Python, run the launcher. Any divergence
#: between them was a divergence in the launch logic.
BAT = """@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>nul || (
  echo [MECH] Python 3.11 or newer not found on PATH. Install it, then retry.
  pause
  exit /b 1
)
python "%~dp0mech_launch.py" %*
if errorlevel 1 pause
exit /b %errorlevel%
"""

PS1 = """# Locates a Python and hands over to mech_launch.py, which holds all the launch
# logic. This file exists so PowerShell users can double-click it; keeping a
# second copy of the sequence here is how the two shells drifted apart before.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
  Write-Host "[MECH] Python 3.11 or newer not found on PATH. Install it, then retry."
  Read-Host "Press Enter to close"
  exit 1
}
python (Join-Path $PSScriptRoot "mech_launch.py") @args
# Exit with the launcher's own status. Without this, a failed dependency install
# exited 0 and the window closed as if everything had worked.
exit $LASTEXITCODE
"""

README = """MECH standalone (backend + UI, one process)
=============================================
Double-click Start-MECH.bat (or Start-MECH.ps1).

- First run creates a local .venv inside this folder and installs the Python
  dependencies into it. Nothing is installed into your system Python, and
  deleting this folder removes the environment completely.
- Dependencies are reinstalled whenever requirements.txt changes, not just on
  the very first run.
- The launcher waits for http://127.0.0.1:8000/health to answer before opening
  your browser, so a slow first start does not look like a broken install.
- Backend API:  http://127.0.0.1:8000/api, /api/v1, /health
- UI served by the SAME backend at http://127.0.0.1:8000/ (no second server,
  no Electron needed).
- Needs Python 3.11+ and Node only for rebuilding the UI, not for running it.
- GPU optional; CPU works.
- If it will not start, run:  python mech_launch.py --no-browser
  That keeps the backend in the foreground so the error is visible.

Rebuild this folder:  python scripts/build_standalone.py
"""


def run(cmd, cwd) -> None:
    print(f"[standalone] {' '.join(cmd)} (cwd={cwd})")
    r = subprocess.run(cmd, cwd=cwd, shell=(sys.platform == "win32"))
    if r.returncode != 0:
        sys.exit(r.returncode)


# ── Validation ──────────────────────────────────────────────────────────

class ValidationError(RuntimeError):
    """The assembled build is not runnable."""


def required_inputs(repo_root: Path = REPO_ROOT) -> List[Path]:
    """Files that must exist before a build can be assembled.

    Checked up front so a missing frontend build fails before anything is
    copied, rather than after the previous standalone folder has been replaced.
    """
    return [
        repo_root / "main.py",
        repo_root / "requirements.txt",
        repo_root / "frontend" / "dist" / "index.html",
        LAUNCHER_SOURCE,
    ]


def validate_inputs(repo_root: Path = REPO_ROOT) -> None:
    missing = [p for p in required_inputs(repo_root) if not p.exists()]
    if missing:
        names = "\n  ".join(str(p.relative_to(repo_root)) for p in missing)
        raise ValidationError(
            f"cannot build: missing {names}\n"
            f"(run without --skip-frontend-build, or build the UI first)")


def validate_build(directory: Path) -> None:
    """Prove the assembled folder is runnable before it replaces the old one.

    The backend import is the check that earns its cost: `copytree` succeeds
    happily while omitting something the app imports at startup, and the only
    place that shows up is a user double-clicking the launcher.
    """
    checks = {
        "main.py": directory / "main.py",
        "requirements.txt": directory / "requirements.txt",
        "frontend index": directory / "frontend" / "dist" / "index.html",
        "launcher": directory / LAUNCHER_NAME,
        "bat shim": directory / "Start-MECH.bat",
        "ps1 shim": directory / "Start-MECH.ps1",
    }
    missing = [label for label, path in checks.items() if not path.exists()]
    if missing:
        raise ValidationError(
            "assembled build is missing: " + ", ".join(missing))

    backend = directory / "backend"
    if not (backend / "main.py").exists():
        raise ValidationError("assembled build has no backend/main.py")

    _assert_backend_imports(directory)


def _assert_backend_imports(directory: Path) -> None:
    """Import the *copied* backend, in isolation, to prove it is complete."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(directory)
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import backend.main"],
            cwd=str(directory), capture_output=True, text=True, env=env, timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValidationError(f"could not check the copied backend: {exc}") from exc
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-15:]
        raise ValidationError(
            "the copied backend does not import:\n  " + "\n  ".join(tail))


# ── Assembly ────────────────────────────────────────────────────────────

def staging_dir(out: Path = OUT) -> Path:
    """Sibling temp directory, so the swap onto `out` is a rename within one
    filesystem. A temp dir under the system temp drive could land on another
    volume, where `os.replace` is a copy."""
    return out.with_name(out.name + ".building")


def backup_dir(out: Path = OUT) -> Path:
    return out.with_name(out.name + ".previous")


def assemble(repo_root: Path, destination: Path) -> None:
    """Copy everything into `destination`, which must not already exist."""
    if destination.exists():
        shutil.rmtree(destination)
    (destination / "frontend").mkdir(parents=True)

    print(f"[standalone] copying backend/ -> {destination/'backend'}")
    shutil.copytree(repo_root / "backend", destination / "backend",
                    ignore=shutil.ignore_patterns(
                        "__pycache__", "*.pyc", "*.pyo"))
    print(f"[standalone] copying frontend/dist -> {destination/'frontend'/'dist'}")
    shutil.copytree(repo_root / "frontend" / "dist",
                    destination / "frontend" / "dist")
    for f in ("main.py", "requirements.txt"):
        shutil.copy2(repo_root / f, destination / f)
    shutil.copy2(LAUNCHER_SOURCE, destination / LAUNCHER_NAME)
    (destination / "Start-MECH.bat").write_text(BAT, encoding="utf-8")
    (destination / "Start-MECH.ps1").write_text(PS1, encoding="utf-8")
    (destination / "README.txt").write_text(README, encoding="utf-8")


def swap(staged: Path, out: Path = OUT) -> None:
    """Move `staged` onto `out`, keeping the old build until the move succeeds.

    `os.replace` cannot overwrite a non-empty directory on Windows, so the old
    build is renamed aside first and only deleted once the new one is in place.
    If the second rename fails the old build is moved back -- the point of the
    whole exercise is that a failed build does not cost you the working one.
    """
    previous = backup_dir(out)
    if previous.exists():
        shutil.rmtree(previous)

    had_previous = out.exists()
    if had_previous:
        os.replace(out, previous)
    try:
        os.replace(staged, out)
    except BaseException:
        if had_previous and previous.exists() and not out.exists():
            os.replace(previous, out)
            print(f"[standalone] swap failed; the previous build was restored")
        raise
    if had_previous:
        shutil.rmtree(previous, ignore_errors=True)


def build(repo_root: Path = REPO_ROOT, out: Path = OUT,
          cleanup: Callable[[], None] = None) -> Path:
    """Assemble, validate, then swap. The previous build survives any failure."""
    validate_inputs(repo_root)
    staged = staging_dir(out)
    try:
        assemble(repo_root, staged)
        validate_build(staged)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        if out.exists():
            print(f"[standalone] build failed; the existing {out.name} is untouched")
        raise
    swap(staged, out)
    if cleanup is not None:
        cleanup()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-frontend-build", action="store_true")
    ap.add_argument("--out", default=str(OUT),
                    help=f"output directory (default: {OUT.name})")
    args = ap.parse_args()
    out = Path(args.out).resolve()

    if not args.skip_frontend_build:
        npm = "npm.cmd" if sys.platform == "win32" else "npm"
        run([npm, "run", "build:renderer"], cwd=REPO_ROOT / "frontend")

    try:
        build(REPO_ROOT, out)
    except ValidationError as exc:
        print(f"[standalone] ERROR: {exc}", file=sys.stderr)
        return 1

    n_files = sum(1 for _ in out.rglob("*") if _.is_file())
    print(f"[standalone] done: {out} ({n_files} files)")
    print(f"[standalone] launch: {out.name}\\Start-MECH.bat")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
