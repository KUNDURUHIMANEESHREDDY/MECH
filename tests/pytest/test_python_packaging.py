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
import os
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


# --------------------------------------------------------------------------
# The torch profile a build ships.
#
# The regression these guard against is not hypothetical: `scripts/build.js`
# called `stage({ force: false })` without a profile, so `isFresh()` compared
# a CPU-staged manifest against the *default* profile, judged it stale, and
# re-staged the runtime from the build machine's CUDA torch. That produced the
# 5.6 GiB bundle NSIS cannot pack, on a machine that had deliberately staged
# the CPU wheel. The profile is now explicit, and it is never changed
# silently.
# --------------------------------------------------------------------------

_PROFILE_DRIVER = (
    "const { resolveProfile } = require('./scripts/python_runtime.js');"
    "const existing = process.env.MECH_PROFILE_EXISTING"
    "  ? JSON.parse(process.env.MECH_PROFILE_EXISTING) : null;"
    "process.stdout.write(JSON.stringify(resolveProfile(existing, process.env)));"
)


def _run_profile_driver(env=None, existing=None):
    """Ask the staging script which profile a build would use."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; the staging script cannot be driven")

    child_env = dict(os.environ)
    child_env.pop("MECH_TORCH_PROFILE", None)
    child_env.pop("MECH_PROFILE_EXISTING", None)
    child_env.update(env or {})
    if existing is not None:
        child_env["MECH_PROFILE_EXISTING"] = json.dumps(existing)

    proc = subprocess.run(
        [node, "-e", _PROFILE_DRIVER],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(FRONTEND),
        env=child_env,
    )
    return proc


def test_the_torch_profile_is_explicit_and_never_changes_silently() -> None:
    """A staged runtime keeps its profile unless the environment names another."""
    # Nothing staged, nothing asked for: the default profile.
    proc = _run_profile_driver()
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert json.loads(proc.stdout) == {
        "mode": "installed",
        "cpu": False,
        "explicit": False,
    }

    # A CPU runtime is already staged and nobody asked for anything else:
    # it stays CPU. This is the case that used to re-stage CUDA.
    proc = _run_profile_driver(existing={"mode": "reinstall", "cpu": True})
    assert proc.returncode == 0, proc.stderr[-1500:]
    resolved = json.loads(proc.stdout)
    assert resolved["mode"] == "reinstall"
    assert resolved["cpu"] is True

    # A default runtime stays default.
    proc = _run_profile_driver(existing={"mode": "installed", "cpu": False})
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert json.loads(proc.stdout)["cpu"] is False

    # Asking for CPU explicitly overrides whatever is staged -- CI depends
    # on this, because a fresh runner has no staged runtime at all.
    proc = _run_profile_driver(
        env={"MECH_TORCH_PROFILE": "cpu"}, existing={"mode": "installed", "cpu": False}
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    resolved = json.loads(proc.stdout)
    assert resolved["mode"] == "reinstall"
    assert resolved["cpu"] is True

    # Asking for the profile that is already staged changes nothing.
    proc = _run_profile_driver(
        env={"MECH_TORCH_PROFILE": "cpu"}, existing={"mode": "reinstall", "cpu": True}
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert json.loads(proc.stdout)["cpu"] is True


def test_an_unknown_torch_profile_is_refused() -> None:
    """A typo must fail the build, not silently pick a profile."""
    proc = _run_profile_driver(env={"MECH_TORCH_PROFILE": "cuda"})
    assert proc.returncode != 0, (
        "an unknown MECH_TORCH_PROFILE must fail; a silently chosen profile is "
        "how a 5.8 GB bundle gets built by accident"
    )
    assert "MECH_TORCH_PROFILE" in (proc.stderr or "")


def _pip_install_args(cpu: bool) -> dict:
    """The exact pip invocation the staging script would run."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; the staging script cannot be driven")

    script = (
        "const { pipInstallArgs, CPU_TORCH_INDEX } = require('./scripts/python_runtime.js');"
        "const args = pipInstallArgs('site', { cpu: process.env.MECH_TEST_CPU === '1' });"
        "process.stdout.write(JSON.stringify({ args: args, index: CPU_TORCH_INDEX }));"
    )
    child_env = dict(os.environ)
    child_env["MECH_TEST_CPU"] = "1" if cpu else "0"
    proc = subprocess.run(
        [node, "-e", script],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(FRONTEND),
        env=child_env,
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    return json.loads(proc.stdout)


def test_the_cpu_profile_resolves_torch_from_the_cpu_index() -> None:
    """The CPU profile must not download the CUDA wheel and then throw it away.

    `requirements.txt` resolves `torch` against PyPI, and the PyPI wheel is the
    CUDA build with gigabytes of `nvidia*` libraries attached. The CPU index is
    therefore passed as an EXTRA index for the CPU profile: pip compares the
    candidates, a local version sorts above the plain release, so `2.14.1+cpu`
    wins over `2.14.1`, and the CUDA wheel is never fetched at all.
    """
    cpu = _pip_install_args(True)
    args = cpu["args"]

    assert "--extra-index-url" in args, (
        "the CPU profile installs requirements.txt without the CPU index, so pip "
        f"downloads the CUDA torch wheel and then discards it: {args}"
    )
    assert args[args.index("--extra-index-url") + 1] == cpu["index"], (
        "the CPU profile must point pip at the CPU torch index"
    )
    # The CPU index is an EXTRA index. Replacing the default index would stop
    # every other dependency from resolving against PyPI.
    assert "--index-url" not in args, (
        "the CPU profile must not replace PyPI with the CPU index: the CPU index "
        f"carries only the torch wheels, not the rest of the stack: {args}"
    )


def test_the_default_profile_resolves_against_pypi_only() -> None:
    """The default profile must not reach for the CPU index behind the operator's back."""
    default = _pip_install_args(False)
    args = default["args"]
    assert "--extra-index-url" not in args, (
        f"the default profile must resolve against PyPI alone: {args}"
    )
    assert "--index-url" not in args, (
        f"the default profile must resolve against PyPI alone: {args}"
    )


def test_a_second_build_keeps_the_staged_profile() -> None:
    """Against the real staged tree: the next build does not change it."""
    manifest_path = STAGED / "MECH-RUNTIME.json"
    if not manifest_path.exists():
        pytest.skip("NOT VERIFIED: no staged runtime manifest at %s" % manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    proc = _run_profile_driver(existing=manifest)
    assert proc.returncode == 0, proc.stderr[-1500:]
    resolved = json.loads(proc.stdout)
    assert resolved["mode"] == manifest["mode"], (
        "the next build would restage the runtime under a different mode: "
        f"{resolved['mode']} != {manifest['mode']}"
    )
    assert resolved["cpu"] == bool(manifest["cpu"]), (
        "the next build would restage the runtime under a different torch "
        f"profile: cpu={resolved['cpu']} != cpu={manifest['cpu']}"
    )
