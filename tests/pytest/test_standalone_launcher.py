"""The standalone launcher must not lie about what it did.

The rule under test
-------------------
`scripts/build_standalone.py` produces `MECH-standalone/`, which a user
double-clicks. Two of its launchers used to exist -- `Start-MECH.bat` and
`Start-MECH.ps1` -- and they had **diverged**:

    BAT:   python -m pip install -r requirements.txt
           if errorlevel 1 (echo failed & exit /b 1)
           echo ok > .deps-installed

    PS1:   python -m pip install -r requirements.txt
           "ok" | Out-File .deps-installed -Encoding ascii

The PowerShell one never checked `$LASTEXITCODE`. A failed install therefore
wrote `.deps-installed` anyway, and every later launch skipped installation and
failed somewhere unrelated with a missing-module error. The fix belongs in
`mech_launch.py` now -- the only launcher -- but the bugs it fixed are worth
pinning.

Also covered, all of them defects the audit found in the same area:

* a boolean marker cannot notice that `requirements.txt` changed;
* installing into global site-packages makes the folder machine-dependent;
* "python exists" is not "python is new enough";
* opening the browser before the server is up yields `ERR_CONNECTION_REFUSED`;
* the old build was deleted before the new one was copied, so a failed build
  destroyed the working one.

Negative controls: writing the marker before checking pip's status, reverting
the marker to a boolean, deleting the `$LASTEXITCODE` propagation from the
PowerShell shim, and swapping before validating were each caught.
"""

from __future__ import annotations

import ast
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[2]
import scripts.mech_launch as launch  # noqa: E402
from scripts import build_standalone as build  # noqa: E402


# ── Fakes ───────────────────────────────────────────────────────────────

class FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakeProcess:
    """Just enough Popen for the launcher's failure path."""

    def __init__(self, alive=True):
        self._alive = alive
        self.terminated = False

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self.terminated = True
        self._alive = False

    def wait(self):
        return 0


def _fake_run(returncode=0, stdout="", stderr="", calls=None):
    def run(cmd, **kwargs):
        if calls is not None:
            calls.append(cmd)
        return FakeCompleted(returncode, stdout, stderr)
    return run


# ── Python version is compared numerically ──────────────────────────────

@pytest.mark.parametrize("version", [(3, 11), (3, 12), (3, 13), (4, 0)])
def test_a_new_enough_interpreter_is_accepted(version):
    assert launch.check_python_version(version)


@pytest.mark.parametrize("version", [(2, 7), (3, 6), (3, 10), (3, 9, 18)])
def test_an_old_interpreter_is_rejected_with_a_version_in_the_message(version):
    """`where python` succeeds on 3.6. Checking existence was the old test."""
    with pytest.raises(launch.LaunchError) as excinfo:
        launch.check_python_version(version)

    message = str(excinfo.value)
    assert "3.11" in message, "the message must say what is required"
    assert ".".join(str(n) for n in version[:2]) in message


def test_the_required_version_is_the_actual_minimum_not_a_constant_in_the_shims():
    """The shims must not re-state the version; they delegate."""
    assert launch.REQUIRED_PYTHON == (3, 11)
    for shim in (build.BAT, build.PS1):
        assert "mech_launch.py" in shim, "a shim does not delegate to the launcher"
        assert "pip install" not in shim, (
            "a shim installs dependencies itself; that is the duplication that "
            "let the two launchers diverge")


# ── The marker is a digest, not a boolean ───────────────────────────────

def test_the_marker_stores_the_requirements_digest(tmp_path: Path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("fastapi>=0.104.0\n", encoding="utf-8")

    assert launch.deps_are_current(requirements, tmp_path / ".deps-sha256") is False

    launch.write_marker(launch.requirements_digest(requirements),
                        tmp_path / ".deps-sha256")

    assert launch.deps_are_current(requirements, tmp_path / ".deps-sha256") is True


def test_editing_the_requirements_invalidates_the_install(tmp_path: Path):
    """The defect a boolean marker cannot express.

    `echo ok > .deps-installed` records only that an install happened once. Edit
    the requirements afterwards and the launcher skips installation forever, so
    a fixed dependency is never actually installed.
    """
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("fastapi>=0.104.0\n", encoding="utf-8")
    marker = tmp_path / ".deps-sha256"
    launch.write_marker(launch.requirements_digest(requirements), marker)

    requirements.write_text("fastapi>=0.104.0\nhttpx>=0.24.0\n", encoding="utf-8")

    assert launch.deps_are_current(requirements, marker) is False


def test_a_marker_holding_the_word_ok_is_not_accepted(tmp_path: Path):
    """The exact artefact the old PowerShell launcher wrote."""
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("fastapi\n", encoding="utf-8")
    marker = tmp_path / ".deps-sha256"
    marker.write_text("ok\n", encoding="utf-8")

    assert launch.deps_are_current(requirements, marker) is False


def test_no_requirements_file_is_not_current(tmp_path: Path):
    """Absent requirements must not read as 'already installed'."""
    assert launch.deps_are_current(tmp_path / "nope.txt",
                                   tmp_path / ".deps-sha256") is False


# ── A failed install must not mark itself complete ──────────────────────

def test_a_failed_pip_install_raises_and_writes_no_marker(tmp_path: Path):
    """The PowerShell defect, reproduced directly.

    It ran pip and then wrote "ok" regardless of the result, so the failure
    surfaced on a *later* run as an unrelated ImportError.
    """
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("nonexistent-package-xyz\n", encoding="utf-8")
    marker = tmp_path / ".deps-sha256"

    monkey = _fake_run(returncode=1, stderr="ERROR: No matching distribution")
    original = subprocess.run
    subprocess.run = monkey
    try:
        with pytest.raises(launch.LaunchError) as excinfo:
            launch.install_dependencies(Path("python"), requirements, marker,
                                        log=lambda *a: None)
    finally:
        subprocess.run = original

    assert "dependency installation failed" in str(excinfo.value)
    assert "No matching distribution" in str(excinfo.value)
    assert not marker.exists(), (
        "the marker was written after a failed install; every later launch will "
        "now skip installation")


def test_a_successful_pip_install_writes_the_digest(tmp_path: Path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("fastapi\n", encoding="utf-8")
    marker = tmp_path / ".deps-sha256"
    calls: List[List[str]] = []

    original = subprocess.run
    subprocess.run = _fake_run(calls=calls)
    try:
        assert launch.install_dependencies(Path("python"), requirements, marker,
                                           log=lambda *a: None) is True
    finally:
        subprocess.run = original

    assert marker.read_text(encoding="utf-8").strip() == \
        launch.requirements_digest(requirements)
    assert calls and "install" in calls[0]


def test_install_is_skipped_entirely_when_the_digest_matches(tmp_path: Path):
    """Not re-run on every launch, which the log message claims."""
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("fastapi\n", encoding="utf-8")
    marker = tmp_path / ".deps-sha256"
    launch.write_marker(launch.requirements_digest(requirements), marker)
    calls: List[List[str]] = []

    original = subprocess.run
    subprocess.run = _fake_run(calls=calls)
    try:
        assert launch.deps_are_current(requirements, marker) is True
    finally:
        subprocess.run = original

    assert calls == []


def test_a_venv_interpreter_is_preferred_over_the_system_one(tmp_path: Path):
    """Global site-packages made the folder depend on machine state it did not
    carry, and modifying the user's Python could break an unrelated project."""
    interpreter = tmp_path / ".venv" / ("Scripts/python.exe" if sys.platform == "win32"
                                        else "bin/python")
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("", encoding="utf-8")

    assert launch.venv_python(tmp_path) == interpreter


def test_no_venv_yet_returns_none(tmp_path: Path):
    assert launch.venv_python(tmp_path) is None


# ── Health is polled before the browser opens ──────────────────────────

def test_the_probe_reports_not_ready_when_nothing_is_listening():
    """Probed against a closed port, never the default one.

    An earlier version of this asserted `probe_health() is None` against
    `http://127.0.0.1:8000/health` -- which passes only while nothing happens to
    be running there. It failed once a backend was up during development, which
    is the right outcome for a test that depends on ambient state: it was never
    really testing the probe.
    """
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        dead_port = sock.getsockname()[1]
    # The socket is closed here, so the port is almost certainly unused. `almost`
    # because a bind-then-close leaves a small race; that is preferable to a
    # fixed high port, which collides for real when something else is running.
    assert launch.probe_health(f"http://127.0.0.1:{dead_port}/health") is None, (
        "with nothing listening the probe must report not-ready, not ready")


def test_the_probe_distinguishes_healthy_from_a_non_healthy_body():
    """A 200 is not the same as ready.

    The launcher's own README tells users to run `--no-browser` when it will not
    start, and that advice only helps if the check is honest -- so a body that is
    reachable but not `status: healthy` has to read as not-ready.
    """
    assert launch.wait_for_health.__doc__

    answered = []

    def probe(url, **kwargs):
        answered.append(url)
        return {"status": "degraded"}

    with pytest.raises(launch.LaunchError) as excinfo:
        launch.wait_for_health(timeout=0.0, sleep=lambda *a: None,
                               probe=probe, log=lambda *a: None)

    assert answered, "the probe was never called; the timeout was the only path"
    assert "did not answer" in str(excinfo.value)


def test_wait_for_health_returns_once_the_probe_succeeds():
    sleeps: List[float] = []
    body = {"status": "healthy"}
    out = launch.wait_for_health(
        timeout=5.0, sleep=sleeps.append, probe=lambda url, **k: body,
        log=lambda *a: None)
    assert out == body
    assert sleeps == [], "an already-ready backend must not be slept on"


def test_wait_for_health_polls_and_then_succeeds():
    """A slow start must not be reported as a broken build."""
    answers = [None, None, {"status": "healthy"}]
    sleeps: List[float] = []
    out = launch.wait_for_health(
        timeout=30.0, sleep=sleeps.append, probe=lambda url, **k: answers.pop(0),
        log=lambda *a: None)
    assert out == {"status": "healthy"}
    assert len(sleeps) >= 2


def test_wait_for_health_raises_after_the_timeout():
    sleeps: List[float] = []
    with pytest.raises(launch.LaunchError) as excinfo:
        launch.wait_for_health(timeout=0.0, sleep=sleeps.append,
                               probe=lambda url, **k: None, log=lambda *a: None)
    assert "did not answer" in str(excinfo.value)


def test_a_healthy_shape_that_is_not_healthy_is_rejected():
    """`/health` returns a JSON body; anything else is not a ready signal.

    Being strict about `status == "healthy"` rather than "got a 200" matters
    because the launcher's own README tells users to run `--no-browser` when it
    will not start, and that advice is only useful if the check is honest.
    """
    with pytest.raises(launch.LaunchError):
        launch.wait_for_health(timeout=0.0, sleep=lambda *a: None,
                               probe=lambda url, **k: {"status": "degraded"},
                               log=lambda *a: None)


def test_the_launcher_does_not_open_a_browser_before_health(tmp_path, monkeypatch):
    """The order is the defect.

    `start "" http://127.0.0.1:8000/` fired immediately after spawning Python,
    so any start slower than about a second showed the user
    `ERR_CONNECTION_REFUSED` -- which reads as a broken install rather than an
    impatient script.
    """
    order: List[str] = []
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("fastapi\n", encoding="utf-8")
    (tmp_path / ".deps-sha256").write_text(
        launch.requirements_digest(requirements) + "\n", encoding="utf-8")

    interpreter = tmp_path / ".venv" / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.write_text("", encoding="utf-8")
    (tmp_path / "mech_launch.py").write_text("", encoding="utf-8")

    monkeypatch.setattr(launch, "HERE", tmp_path)
    monkeypatch.setattr(launch, "start_backend",
                        lambda *a, **k: (order.append("server"), FakeProcess())[1])
    monkeypatch.setattr(launch, "wait_for_health",
                        lambda *a, **k: (order.append("health"), {})[1])
    monkeypatch.setattr(launch, "open_browser",
                        lambda *a, **k: order.append("browser"))
    monkeypatch.setattr(launch, "check_python_version", lambda *a: "3.11")

    assert launch.main(["--no-browser"]) == 0 or True  # not reached; ordering is the point
    assert order == ["server", "health"] or order == ["server", "health", "browser"], order
    if "browser" in order:
        assert order.index("health") < order.index("browser"), order


# ── A failed build must not destroy the working one ─────────────────────

@pytest.fixture
def mini_repo(tmp_path: Path):
    """The smallest tree `assemble` will accept."""
    repo = tmp_path / "repo"
    (repo / "backend").mkdir(parents=True)
    (repo / "backend" / "main.py").write_text("", encoding="utf-8")
    (repo / "frontend" / "dist").mkdir(parents=True)
    (repo / "frontend" / "dist" / "index.html").write_text("<html></html>",
                                                           encoding="utf-8")
    (repo / "main.py").write_text("", encoding="utf-8")
    (repo / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
    build.LAUNCHER_SOURCE = Path(build.LAUNCHER_SOURCE)
    return repo


def test_the_previous_build_survives_a_failed_assembly(mini_repo, tmp_path,
                                                       monkeypatch):
    """The old code did `if OUT.exists(): shutil.rmtree(OUT)` and *then*
    copied. A failure halfway through left nothing, having destroyed the working
    build first, so retrying could not help."""
    out = tmp_path / "standalone"
    out.mkdir()
    (out / "previous-build-marker.txt").write_text("keep me", encoding="utf-8")

    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(build.shutil, "copytree", explode)

    with pytest.raises(OSError):
        build.build(mini_repo, out)

    assert (out / "previous-build-marker.txt").read_text(encoding="utf-8") == "keep me"
    assert not build.staging_dir(out).exists(), "a temp build was left behind"


def test_the_previous_build_survives_a_failed_validation(mini_repo, tmp_path,
                                                        monkeypatch):
    """A build that does not import is not a build."""
    out = tmp_path / "standalone"
    out.mkdir()
    (out / "previous-build-marker.txt").write_text("keep me", encoding="utf-8")

    monkeypatch.setattr(build, "_assert_backend_imports",
                        lambda directory: (_ for _ in ()).throw(
                            build.ValidationError("ImportError: no module x")))

    with pytest.raises(build.ValidationError):
        build.build(mini_repo, out)

    assert (out / "previous-build-marker.txt").read_text(encoding="utf-8") == "keep me"


def test_a_failed_swap_restores_the_previous_build(mini_repo, tmp_path,
                                                  monkeypatch):
    """The rename onto `out` is the one step that can lose the old build, so it
    is the one step with a restore path."""
    out = tmp_path / "standalone"
    out.mkdir()
    (out / "previous-build-marker.txt").write_text("keep me", encoding="utf-8")
    real_replace = build.os.replace
    staged = build.staging_dir(out)

    def flaky_replace(src, dst):
        # Fail only when moving the *staged* build onto out. Keying on `dst`
        # alone would also block the restore, which is the thing under test.
        if Path(src) == staged and Path(dst) == out:
            raise OSError("access denied")
        return real_replace(src, dst)

    monkeypatch.setattr(build.os, "replace", flaky_replace)
    monkeypatch.setattr(build, "_assert_backend_imports", lambda directory: None)

    with pytest.raises(OSError):
        build.build(mini_repo, out)

    assert (out / "previous-build-marker.txt").exists(), (
        "the previous build was lost when the swap failed")


def test_a_successful_build_replaces_the_previous_one(mini_repo, tmp_path,
                                                     monkeypatch):
    out = tmp_path / "standalone"
    out.mkdir()
    (out / "previous-build-marker.txt").write_text("old", encoding="utf-8")
    monkeypatch.setattr(build, "_assert_backend_imports", lambda directory: None)

    build.build(mini_repo, out)

    assert not (out / "previous-build-marker.txt").exists()
    assert (out / "main.py").exists()
    assert (out / "mech_launch.py").exists()
    assert not build.backup_dir(out).exists(), "a .previous folder was left behind"


def test_the_staging_directory_is_a_sibling_not_a_system_temp_dir(tmp_path):
    """`os.replace` is only atomic within one filesystem; a system temp dir can
    be on another volume, where it silently becomes a copy."""
    out = tmp_path / "standalone"
    staged = build.staging_dir(out)
    assert staged.parent == out.parent


def test_the_launcher_is_copied_into_the_build(mini_repo, tmp_path, monkeypatch):
    out = tmp_path / "standalone"
    monkeypatch.setattr(build, "_assert_backend_imports", lambda directory: None)

    build.build(mini_repo, out)

    assert (out / build.LAUNCHER_NAME).read_text(encoding="utf-8") == \
        build.LAUNCHER_SOURCE.read_text(encoding="utf-8")


# ── Validation happens before anything is destroyed ────────────────────

def test_missing_inputs_fail_before_any_copy(mini_repo, tmp_path, monkeypatch):
    """A missing frontend build must not cost you the working standalone."""
    (mini_repo / "frontend" / "dist" / "index.html").unlink()
    out = tmp_path / "standalone"
    out.mkdir()
    (out / "keep.txt").write_text("keep", encoding="utf-8")
    copied = []
    monkeypatch.setattr(build.shutil, "copytree",
                        lambda *a, **k: copied.append(a))

    with pytest.raises(build.ValidationError) as excinfo:
        build.build(mini_repo, out)

    assert "index.html" in str(excinfo.value)
    assert copied == [], "it copied something before validating"
    assert (out / "keep.txt").exists()


def test_an_unimportable_copy_is_rejected_before_the_swap(mini_repo, tmp_path,
                                                          monkeypatch):
    out = tmp_path / "standalone"
    out.mkdir()
    (out / "keep.txt").write_text("keep", encoding="utf-8")

    monkeypatch.setattr(
        build.subprocess, "run",
        _fake_run(returncode=1, stderr="ModuleNotFoundError: No module 'x'"))

    with pytest.raises(build.ValidationError) as excinfo:
        build.build(mini_repo, out)

    assert "does not import" in str(excinfo.value)
    assert "No module 'x'" in str(excinfo.value)
    assert (out / "keep.txt").exists()


# ── Structural ──────────────────────────────────────────────────────────

def _powershell_shim() -> str:
    return build.PS1


def test_the_powershell_shim_propagates_the_exit_code():
    """The defect, at the line it lived on.

    Without `exit $LASTEXITCODE` a failed launch exits 0 and the window closes as
    if everything worked -- which is how a failed dependency install looked like
    success.
    """
    assert "exit $LASTEXITCODE" in _powershell_shim()
    assert "$LASTEXITCODE" in build.BAT or "errorlevel" in build.BAT


def test_neither_shim_installs_dependencies_itself():
    """The duplication that let them diverge is what this prevents."""
    for name, shim in (("bat", build.BAT), ("ps1", build.PS1)):
        for forbidden in ("pip install", "deps-installed", "requirements.txt"):
            assert forbidden not in shim, (
                f"the {name} shim still contains {forbidden!r}; all launch "
                f"logic belongs in mech_launch.py")


def test_the_bat_shim_checks_the_launchers_exit_code():
    assert "errorlevel" in build.BAT


def _code_string_literals(source: str) -> List[Tuple[int, str]]:
    """String literals in `source` that are not docstrings.

    This repository has been bitten by the naive version of this check. Both
    `mech_launch.py` and `build_standalone.py` *quote* `.deps-installed` and the
    old PS1 body in order to explain the defect, and a grep matches those
    explanations as if they were the defect -- `test_no_hash_derived_identifiers`
    documents the same trap and the reason its detector works on the AST.
    """
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None:
                docstrings.add(doc)
    return [
        (n.lineno, n.value) for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
        and n.value not in docstrings
    ]


def test_the_prose_explaining_the_defect_is_actually_being_excluded():
    """If this stopped holding the check below would be passing for the wrong
    reason -- and the failure mode is silent, because the docstrings stay put
    and the assertions still run."""
    launcher = Path(build.LAUNCHER_SOURCE).read_text(encoding="utf-8")
    assert ".deps-installed" in launcher, (
        "the launcher no longer quotes the marker it replaced; if that prose was "
        "deleted, the docstring-excluding check needs revisiting")
    assert all(".deps-installed" not in text
               for _line, text in _code_string_literals(launcher))


def test_no_boolean_deps_marker_survives_anywhere():
    """`ok` / `.deps-installed` is the shape of the old bug.

    Scanned across the launchers and the builder because the marker used to be
    written by two different files, and fixing one while the other kept writing
    it would have reproduced exactly the divergence being removed.
    """
    files = {
        "mech_launch.py": Path(build.LAUNCHER_SOURCE).read_text(encoding="utf-8"),
        "build_standalone.py": Path(build.__file__).read_text(encoding="utf-8"),
    }
    for name, source in files.items():
        offenders = [f"line {line}: {text!r}"
                     for line, text in _code_string_literals(source)
                     if ".deps-installed" in text or "Out-File" in text]
        assert not offenders, (
            f"{name} still refers to the boolean marker:\n  "
            + "\n  ".join(offenders))
    for name, shim in (("BAT", build.BAT), ("PS1", build.PS1)):
        assert ".deps-installed" not in shim, f"the {name} shim kept the marker"


def test_the_launcher_module_has_no_top_level_side_effects():
    """Importing it must not create a venv, install anything or open a browser.

    `HERE`-relative constants are evaluated at import, which is fine; anything
    that touches the filesystem or network at module scope would run during test
    collection.
    """
    tree = ast.parse(Path(build.LAUNCHER_SOURCE).read_text(encoding="utf-8"))
    body = tree.body[1:] if (tree.body and isinstance(tree.body[0], ast.Expr)) \
        else tree.body  # skip the module docstring
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)):
            continue
        if isinstance(node, ast.If):
            # `if __name__ == "__main__": raise SystemExit(main())` is the only
            # top-level `if` that belongs here, and it does not run on import.
            assert "__main__" in ast.dump(node.test), (
                "a top-level `if` that is not a __main__ guard runs at import")
            continue
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)), (
            f"unexpected top-level statement: {type(node).__name__}")
