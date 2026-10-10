"""MCP control plane tests: same Core API as the UI, no UI automation.

Never starts a real society run (that would load weights); cancellation is
tested with a fabricated registry entry. Containment tests never touch paths
outside the repo or the test tmp dir.
"""
import asyncio
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.mcp_server import workspace as ws  # noqa: E402
from backend.mcp_server.server import mcp  # noqa: E402
import backend.mcp_server.server as srv  # noqa: E402

EXPECTED_TOOLS = {
    "mech_project_list", "mech_project_add", "mech_project_open",
    "mech_file_list", "mech_file_read", "mech_file_write",
    "mech_loop_list", "mech_loop_start", "mech_loop_status",
    "mech_loop_events", "mech_loop_stop",
    "mech_agent_list",
    "mech_experiment_list", "mech_experiment_create",
    "mech_runtime_status", "mech_runtime_logs",
    "mech_git_diff", "mech_git_commit",
    "mech_test_run", "mech_build_run",
}
# Capabilities with no live executor must stay absent, not simulated.
# mech_shell_exec was removed (P0-3): arbitrary argv is not a capability the
# control plane offers; tests/builds go through the allowlisted runners.
HONESTLY_ABSENT = {"mech_loop_pause", "mech_loop_resume", "mech_agent_run",
                   "mech_agent_stop", "mech_shell_exec"}


@pytest.fixture(autouse=True)
def _isolate_workspace():
    ws.clear_open_roots()
    yield
    ws.clear_open_roots()


def test_tool_registry_matches_live_capabilities():
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    assert EXPECTED_TOOLS <= names, f"missing tools: {EXPECTED_TOOLS - names}"
    assert not (HONESTLY_ABSENT & names), (
        f"simulated capabilities exposed: {HONESTLY_ABSENT & names}")


def test_containment_refuses_traversal(monkeypatch):
    monkeypatch.delenv("MECH_WORKSPACE_ROOTS", raising=False)
    drive_root = str(ws.REPO_ROOT.anchor)  # e.g. C:\ — outside the repo root
    res = srv.mech_file_read(os.path.join(drive_root, "Windows", "x.ini"))
    assert res["status"] == "error" and "outside" in res["reason"]
    res = srv.mech_file_list(drive_root)
    assert res["status"] == "error" and "outside" in res["reason"]
    res = srv.mech_test_run("pytest", [], cwd=drive_root)
    assert res["status"] == "error" and "outside" in res["reason"]
    res = srv.mech_git_diff(cwd=drive_root)
    assert res["status"] == "error" and "outside" in res["reason"]


def test_file_roundtrip_in_tmp_root(tmp_path, monkeypatch):
    monkeypatch.setenv("MECH_WORKSPACE_ROOTS", str(tmp_path))
    w = srv.mech_file_write(str(tmp_path / "sub" / "note.txt"), "hello mech")
    assert w["status"] == "ok", w
    r = srv.mech_file_read(str(tmp_path / "sub" / "note.txt"))
    assert r["status"] == "ok" and r["content"] == "hello mech"
    listing = srv.mech_file_list(str(tmp_path / "sub"))
    assert listing["status"] == "ok"
    assert [e["name"] for e in listing["entries"]] == ["note.txt"]


def test_test_run_pytest_version_is_live(monkeypatch):
    """The allowlisted runner actually spawns: pytest --version, contained."""
    monkeypatch.delenv("MECH_WORKSPACE_ROOTS", raising=False)
    res = srv.mech_test_run("pytest", ["--version"], cwd=".")
    assert res["status"] == "ok", res
    assert res["returncode"] == 0
    assert "pytest" in (res["stdout"] + res["stderr"]).lower()


def test_test_run_rejects_off_menu(monkeypatch):
    monkeypatch.delenv("MECH_WORKSPACE_ROOTS", raising=False)
    res = srv.mech_test_run("tox", [], cwd=".")
    assert res["status"] == "error" and "unknown test runner" in res["reason"]
    res = srv.mech_test_run("pytest", ["--junitxml=/tmp/x.xml"], cwd=".")
    assert res["status"] == "error" and "not allowed" in res["reason"]
    res = srv.mech_test_run("pytest", ["-k"], cwd=".")
    assert res["status"] == "error" and "-k requires" in res["reason"]
    res = srv.mech_test_run("pytest", ["/etc/passwd"], cwd=".")
    assert res["status"] == "error"
    assert "escapes" in res["reason"] or "does not exist" in res["reason"]


def test_build_run_rejects_off_menu():
    res = srv.mech_build_run("electron")
    assert res["status"] == "error" and "unknown build target" in res["reason"]


def test_git_diff_pathspec_escape_rejected(monkeypatch, tmp_path):
    import tempfile
    monkeypatch.delenv("MECH_WORKSPACE_ROOTS", raising=False)
    outside = tempfile.mkdtemp()
    try:
        res = srv.mech_git_diff(cwd=".", path=outside)
        assert res["status"] == "error" and "outside" in res["reason"]
    finally:
        import shutil
        shutil.rmtree(outside, ignore_errors=True)


def test_git_commit_message_cap():
    res = srv.mech_git_commit(cwd=".", message="x" * 1001)
    assert res["status"] == "error" and "exceeds" in res["reason"]


def test_no_arbitrary_shell_tool():
    assert not hasattr(srv, "mech_shell_exec"), (
        "mech_shell_exec must not exist on the MCP surface")


def test_loop_start_rejects_empty_goal_without_spawning():
    from backend.api import dispatcher as core

    before = len(core._society_runs)
    res = srv.mech_loop_start("")
    assert res.get("status") == "error", res
    assert len(core._society_runs) == before


def test_loop_stop_unknown_run_is_an_error():
    res = srv.mech_loop_stop("r_does_not_exist_123")
    assert res["status"] == "error" and "unknown runId" in res["reason"]


def test_loop_stop_is_cooperative_cancel():
    from backend.api import dispatcher as core

    run_id = "r_test_cancel_probe"
    core._society_runs[run_id] = {"run_id": run_id, "status": "running",
                                  "events": [], "result": None,
                                  "queue": __import__("queue").Queue(),
                                  "created": 0.0}
    try:
        res = srv.mech_loop_stop(run_id)
        assert res["status"] == "cancel_requested", res
        assert "in-flight" in res["note"]
        # Post-cancel events are dropped, not recorded.
        core._society_push(run_id, {"type": "probe"})
        assert core._society_runs[run_id]["events"] == []
    finally:
        core._society_runs.pop(run_id, None)


def test_project_and_runtime_tools_are_live():
    projects = srv.mech_project_list()
    assert projects["status"] == "ok" and "projects" in projects
    agents = srv.mech_agent_list()
    assert "agents" in agents
    status = srv.mech_runtime_status()
    assert status["status"] in {"connected", "unavailable"}
    assert status["provenance"] == "live"


# ── P0-2: project roots must not self-expand the trust boundary ──────────
#
# NOTE: pytest's tmp_path lives under .pytest-tmp, i.e. INSIDE the repo root,
# so it is the wrong fixture for "outside" cases. Outside cases use the real
# system temp dir (outside the repo); tmp_path is only used for inside and
# explicitly-configured cases.


def _outside_tmpdir() -> str:
    import tempfile
    return tempfile.mkdtemp()


def test_project_open_outside_roots_rejected(monkeypatch):
    """An MCP caller cannot authorize an arbitrary directory by opening it."""
    import shutil
    monkeypatch.delenv("MECH_WORKSPACE_ROOTS", raising=False)
    before = set(ws.workspace_roots())
    drive_root = str(ws.REPO_ROOT.anchor)
    with pytest.raises(ValueError, match="outside the allowed workspace roots"):
        ws.register_open_root(drive_root)
    outside = _outside_tmpdir()
    try:
        with pytest.raises(ValueError, match="outside the allowed workspace roots"):
            ws.register_open_root(outside)
    finally:
        shutil.rmtree(outside, ignore_errors=True)
    assert set(ws.workspace_roots()) == before


def test_project_open_dotdot_rejected(monkeypatch):
    monkeypatch.delenv("MECH_WORKSPACE_ROOTS", raising=False)
    # The repo parent exists and is outside the boundary (ancestor, not child).
    with pytest.raises(ValueError, match="outside the allowed workspace roots"):
        ws.register_open_root(str(ws.REPO_ROOT.parent))
    # A nonexistent path is rejected as not-a-directory, never registered.
    with pytest.raises(ValueError, match="not a directory"):
        ws.register_open_root(str(ws.REPO_ROOT / ".." / "no_such_dir_xyz"))


def test_project_open_symlink_escape_rejected(monkeypatch, tmp_path):
    """A link inside the boundary resolving outside must not become a root."""
    import shutil
    outside = _outside_tmpdir()
    try:
        monkeypatch.setenv("MECH_WORKSPACE_ROOTS", str(tmp_path / "work"))
        work = tmp_path / "work"
        work.mkdir(exist_ok=True)
        link = work / "evil_link"
        try:
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(outside, target_is_directory=True)
        except OSError:
            pytest.skip("cannot create symlinks on this machine")
        with pytest.raises(ValueError, match="outside the allowed workspace roots"):
            ws.register_open_root(str(link))
    finally:
        shutil.rmtree(outside, ignore_errors=True)


def test_project_open_inside_repo_accepted(monkeypatch):
    monkeypatch.delenv("MECH_WORKSPACE_ROOTS", raising=False)
    root = ws.register_open_root(".")
    assert root == ws.REPO_ROOT.resolve()
    assert root in ws.workspace_roots()


def test_project_open_configured_subdir_accepted(monkeypatch, tmp_path):
    monkeypatch.setenv("MECH_WORKSPACE_ROOTS", str(tmp_path))
    proj = tmp_path / "proj"
    proj.mkdir(exist_ok=True)
    root = ws.register_open_root(str(proj))
    assert root == proj.resolve()
    target = ws.resolve_contained(str(proj / "note.txt"))
    assert str(target).startswith(str(proj.resolve()))


def test_mech_project_open_tool_rejects_outside(monkeypatch):
    """Tool-level: rejection happens before any project registration."""
    import shutil
    monkeypatch.delenv("MECH_WORKSPACE_ROOTS", raising=False)
    before = set(ws.workspace_roots())
    outside = _outside_tmpdir()
    try:
        res = srv.mech_project_open(outside)
    finally:
        shutil.rmtree(outside, ignore_errors=True)
    assert res["status"] == "error" and "outside" in res["reason"]
    assert set(ws.workspace_roots()) == before


def test_project_open_home_boundary(monkeypatch):
    """Home is rejected by default, accepted only when explicitly configured."""
    from pathlib import Path as _P
    home = str(_P.home())
    monkeypatch.delenv("MECH_WORKSPACE_ROOTS", raising=False)
    with pytest.raises(ValueError, match="outside the allowed workspace roots"):
        ws.register_open_root(home)
    monkeypatch.setenv("MECH_WORKSPACE_ROOTS", home)
    ws.clear_open_roots()
    root = ws.register_open_root(home)
    assert root == _P(home).resolve()


def test_project_open_sibling_of_configured_rejected(monkeypatch):
    """Same-parent sibling of a configured root is still outside it."""
    import shutil
    base = _outside_tmpdir()
    try:
        from pathlib import Path as _P
        allow = _P(base) / "allow"
        sib = _P(base) / "sib"
        allow.mkdir(exist_ok=True)
        sib.mkdir(exist_ok=True)
        monkeypatch.setenv("MECH_WORKSPACE_ROOTS", str(allow))
        ws.clear_open_roots()
        assert ws.register_open_root(str(allow)) == allow.resolve()
        ws.clear_open_roots()
        with pytest.raises(ValueError, match="outside the allowed workspace roots"):
            ws.register_open_root(str(sib))
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_project_open_junction_escape_rejected(monkeypatch, tmp_path):
    """A Windows junction inside the boundary resolving outside is rejected."""
    import shutil
    import subprocess
    outside = _outside_tmpdir()
    try:
        monkeypatch.setenv("MECH_WORKSPACE_ROOTS", str(tmp_path / "work"))
        work = tmp_path / "work"
        work.mkdir(exist_ok=True)
        link = work / "junction_link"
        try:
            if link.exists() or link.is_symlink():
                link.unlink()
        except OSError:
            pass
        proc = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), outside],
            capture_output=True, text=True, timeout=60)
        if proc.returncode != 0 or not link.exists():
            pytest.skip(f"cannot create junctions here: {proc.stderr[:200]}")
        with pytest.raises(ValueError, match="outside the allowed workspace roots"):
            ws.register_open_root(str(link))
    finally:
        # Remove the junction before its target: a dangling reparse point
        # inside the workspace makes later directory walks raise.
        try:
            if link.exists() or link.is_symlink():
                link.unlink()
        except OSError:
            subprocess.run(["cmd", "/c", "rmdir", str(link)],
                           capture_output=True, timeout=30)
        shutil.rmtree(outside, ignore_errors=True)
