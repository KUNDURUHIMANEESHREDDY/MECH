"""The Windows release must ship its own Python runtime.

Claims under test:

* `frontend/package.json` copies `build/python` to `resources/python` and no
  longer copies a virtual environment (which is not portable).
* `scripts/build.js` stages that runtime and fails the build when it cannot.
* `scripts/python_runtime.js` plans a runtime whose required imports include the
  inference stack, and the staged runtime -- when one has been built -- actually
  imports it. That last test is the only one that proves portability, so it
  skips loudly rather than passing quietly when no artifact exists.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND = REPO_ROOT / "frontend"
PACKAGE_JSON = FRONTEND / "package.json"
BUILD_JS = FRONTEND / "scripts" / "build.js"
RUNTIME_JS = FRONTEND / "scripts" / "python_runtime.js"
STAGED = FRONTEND / "build" / "python"
STAGED_PYTHON = STAGED / ("python.exe" if sys.platform == "win32" else "bin/python3")


def _entries() -> list:
    pkg = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))
    return pkg["build"]["extraResources"]


def test_extra_resources_carry_the_bundled_runtime() -> None:
    to_python = [entry for entry in _entries() if entry.get("to") == "python"]
    assert len(to_python) == 1, (
        "frontend/package.json must declare exactly one extraResources entry "
        f'with "to": "python"; it declares {len(to_python)}'
    )
    entry = to_python[0]
    assert entry["from"] == "build/python", (
        "the bundled runtime must come from the staged build/python directory, "
        f"not {entry['from']!r}"
    )
    # The filter must not ship bytecode caches: they are stale-path landmines.
    assert "!**/__pycache__/**" in entry["filter"]
    assert "!**/*.pyc" in entry["filter"]


def test_no_virtualenv_is_shipped() -> None:
    """A copied venv resolves the build machine's paths, not the app's."""
    offenders = [
        entry for entry in _entries()
        if ".venv" in str(entry.get("from", "")) or ".venv" in str(entry.get("to", ""))
    ]
    assert offenders == [], (
        f"frontend/package.json still packages a virtual environment: {offenders}"
    )


def test_build_fails_when_the_runtime_cannot_be_staged() -> None:
    source = BUILD_JS.read_text(encoding="utf-8")
    assert "python_runtime" in source, (
        "scripts/build.js does not stage the Python runtime"
    )
    assert "stagePythonRuntime()" in source
    # A build that continues without an interpreter produces a broken installer.
    assert "process.exit(1)" in source
    stage_fn = source.split("function stagePythonRuntime()", 1)[1].split("function main()", 1)[0]
    assert "process.exit(1)" in stage_fn, (
        "stagePythonRuntime() must abort the build on failure, not warn"
    )


def test_plan_lists_the_inference_stack() -> None:
    """Drive the staging script: what would it ship, and where to?"""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; the staging script cannot be driven")

    proc = subprocess.run(
        [node, str(RUNTIME_JS), "--plan"],
        capture_output=True,
        text=True,
        timeout=900,
        cwd=str(FRONTEND),
    )
    assert proc.returncode == 0, (
        f"python_runtime.js --plan failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}"
    )
    plan = json.loads(proc.stdout)

    assert Path(plan["dest"]).name == "python"
    assert Path(plan["dest"]).parent.name == "build"
    imports = set(plan["requiredImports"])
    for required in ("fastapi", "torch", "transformer_lens"):
        assert required in imports, (
            f"{required} is not in the staged runtime's required imports: {sorted(imports)}"
        )
    assert plan["version"].startswith("3.11"), (
        f"the runtime must be CPython 3.11 (requirements pin numpy<2 for it), "
        f"got {plan['version']}"
    )


def test_staged_runtime_actually_imports_the_backend() -> None:
    """Execute the shipped interpreter: the only real portability evidence."""
    if not STAGED_PYTHON.exists():
        pytest.skip(
            "NOT VERIFIED: no staged runtime at %s, so the bundled interpreter "
            "was not executed. Run `node frontend/scripts/python_runtime.js` "
            "before claiming the installer is self-contained." % STAGED
        )

    code = "import fastapi, uvicorn, torch, transformer_lens, numpy, pandas; print('ok')"
    proc = subprocess.run(
        [str(STAGED_PYTHON), "-c", code],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, (
        "the staged interpreter cannot import the backend requirements:\n"
        f"{proc.stdout[-1500:]}\n{proc.stderr[-3000:]}"
    )
    assert "ok" in proc.stdout


def test_staged_manifest_matches_the_source_requirements() -> None:
    manifest_path = STAGED / "MECH-RUNTIME.json"
    if not manifest_path.exists():
        pytest.skip("NOT VERIFIED: no staged runtime manifest at %s" % manifest_path)

    import hashlib

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = hashlib.sha256((REPO_ROOT / "requirements.txt").read_bytes()).hexdigest()
    assert manifest["requirementsSha256"] == expected, (
        "the staged runtime was built from a different requirements.txt than the "
        "one in the repository; restage with --force"
    )
