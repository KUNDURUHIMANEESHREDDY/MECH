"""MECH MCP control plane: external agents drive MECH through tools, not the UI.

Architecture::

             External AI Agent
                     │  MCP (stdio, or http://127.0.0.1:8000/mcp/)
                     ▼
             backend.mcp_server.server (this module, FastMCP "mech-control")
                     │  direct function calls — no HTTP loopback, no UI
                     ▼
             MECH Core API (`backend.api.dispatcher`)
              projects / files / loops / agents / runtime / experiments
                     ▲
                     │  same functions
             MECH Vue UI (fetch /api/*)

Honesty notes, enforced by construction:

- Only capabilities with a live executor are exposed. There is no
  ``mech_loop_pause`` / ``mech_loop_resume`` / ``mech_agent_run`` because the
  execution engine has no pause primitive and agents only run as society
  roles inside a loop. A missing tool means *capability absent*.
- ``mech_loop_stop`` is cooperative (best-effort): the in-flight model call
  runs to completion, queued events are dropped, and the run is reported
  ``cancelled``. It never claims to have killed a thread.
- Every response carries ``provenance: live`` only when it was measured or
  performed by this process; refusals carry ``unavailable`` with a reason.

Run standalone over stdio (for Codex / Claude / OpenCode agent wiring)::

    python -m backend.mcp_server.server

Or use the HTTP transport mounted by ``backend.main`` at ``/mcp``.
"""
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from backend.mcp_server import workspace as ws

mcp = FastMCP("mech-control")

MAX_READ_CHARS = 200_000
MAX_WRITE_CHARS = 500_000
MAX_LOG_LINES = 2_000


def _ok(**fields: Any) -> dict:
    return {"status": "ok", "provenance": "live", **fields}


def _err(reason: str) -> dict:
    return {"status": "error", "provenance": "unavailable", "reason": reason}


# ── Projects (same store as the UI: backend.storage via dispatcher) ─────────


@mcp.tool(name="mech_project_list")
def mech_project_list() -> dict:
    """List MECH projects. Same store the UI reads."""
    from backend.api import dispatcher as core

    return core.list_projects()


@mcp.tool(name="mech_project_add")
def mech_project_add(path: str, name: str = "") -> dict:
    """Register a project path with MECH. Same store the UI writes."""
    from backend.api import dispatcher as core

    return core.add_project({"path": path, "name": name})


@mcp.tool(name="mech_project_open")
def mech_project_open(path: str) -> dict:
    """Open a project: registers it and allows file tools inside it.

    Returns a workspace listing so the agent can orient immediately.
    """
    from backend.api import dispatcher as core

    try:
        root = ws.register_open_root(path)
    except ValueError as exc:
        return _err(str(exc))
    record = core.add_project({"path": str(root), "name": root.name})
    return _ok(root=str(root), project=record.get("project"),
               listing=_list_dir(root, limit=100))


# ── Files (contained; read/write/list, never outside allowed roots) ──────────


def _list_dir(directory: Path, limit: int) -> dict:
    entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    capped = entries[: max(1, min(limit, 500))]
    return {
        "path": str(directory),
        "entries": [
            {"name": p.name, "dir": p.is_dir(),
             "size": p.stat().st_size if p.is_file() else None}
            for p in capped
        ],
        "truncated": len(entries) > len(capped),
    }


@mcp.tool(name="mech_file_list")
def mech_file_list(path: str = ".", limit: int = 100) -> dict:
    """List a workspace directory. Refuses paths outside allowed roots."""
    try:
        target = ws.resolve_contained(path)
    except ValueError as exc:
        return _err(str(exc))
    if not target.is_dir():
        return _err(f"not a directory: {target}")
    return _ok(**_list_dir(target, limit))


@mcp.tool(name="mech_file_read")
def mech_file_read(path: str, max_chars: int = 20_000) -> dict:
    """Read a workspace file as text. Refuses paths outside allowed roots."""
    try:
        target = ws.resolve_contained(path)
    except ValueError as exc:
        return _err(str(exc))
    if not target.is_file():
        return _err(f"not a file: {target}")
    if target.stat().st_size > MAX_READ_CHARS * 4:
        return _err(f"file too large to read via MCP: {target}")
    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return _err(f"not a UTF-8 text file: {target}")
    except OSError as exc:
        return _err(f"read failed: {exc}")
    cap = max(1, min(max_chars, MAX_READ_CHARS))
    return _ok(path=str(target), size=len(text), truncated=len(text) > cap,
               content=text[:cap])


@mcp.tool(name="mech_file_write")
def mech_file_write(path: str, content: str) -> dict:
    """Write a workspace file (creates parent dirs). Contained like reads."""
    try:
        target = ws.resolve_contained(path)
    except ValueError as exc:
        return _err(str(exc))
    if not isinstance(content, str):
        return _err("content must be a string")
    if len(content) > MAX_WRITE_CHARS:
        return _err(f"content exceeds {MAX_WRITE_CHARS} chars; write in parts")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        return _err(f"write failed: {exc}")
    return _ok(path=str(target), bytes=len(content.encode("utf-8")))


# ── Loops (Research Society runs: start / status / events / stop) ────────────


@mcp.tool(name="mech_loop_list")
def mech_loop_list() -> dict:
    """List society runs (durable records, survive restarts)."""
    from backend.api import dispatcher as core

    return core.society_run_list()


@mcp.tool(name="mech_loop_start")
def mech_loop_start(goal: str, model_name: str = "gpt2") -> dict:
    """Start a society run for a goal. Returns runId + SSE stream path."""
    from backend.api import dispatcher as core

    return core.society_run({"goal": goal, "model_name": model_name})


@mcp.tool(name="mech_loop_status")
def mech_loop_status(run_id: str) -> dict:
    """Status + result of one run (live, or the persisted record)."""
    from backend.api import dispatcher as core

    return core.society_run_status(run_id)


@mcp.tool(name="mech_loop_events")
def mech_loop_events(run_id: str, limit: int = 100) -> dict:
    """Recent trace events for a run (tail of what SSE would stream)."""
    from backend.api import dispatcher as core

    res = core.society_run_status(run_id)
    if isinstance(res, dict) and "events" in res:
        events = res["events"]
        if isinstance(events, list):
            cap = max(1, min(limit, 1000))
            return _ok(run_id=run_id, status=res.get("status"),
                       event_count=len(events), events=events[-cap:])
    return res if isinstance(res, dict) else _err(f"unreadable run '{run_id}'")


@mcp.tool(name="mech_loop_stop")
def mech_loop_stop(run_id: str) -> dict:
    """Request cancellation of a running loop (cooperative, best-effort).

    Queued events are dropped and the run is reported ``cancelled`` once the
    in-flight step returns. An in-flight model call runs to completion;
    this tool never claims to have killed a thread.
    """
    from backend.api import dispatcher as core

    run = core._society_runs.get(run_id)
    if run is None:
        try:
            from backend.core.evidence_graph import load_run_record

            rec = load_run_record(run_id)
            return _err(f"run {run_id} already finished "
                        f"({rec.get('status', 'stored')}); nothing to stop")
        except Exception:
            return _err(f"unknown runId '{run_id}'")
    run["cancel_requested"] = True
    return {
        "status": "cancel_requested",
        "run_id": run_id,
        "provenance": "live",
        "note": ("cooperative cancel: queued events are dropped and the run "
                 "is reported cancelled when the in-flight step returns; an "
                 "in-flight model call runs to completion."),
    }


# ── Agents / experiments (read the same registries as the UI) ────────────────


@mcp.tool(name="mech_agent_list")
def mech_agent_list() -> dict:
    """List known agents and LLM backends.

    Agents run as society roles inside a loop (see mech_loop_start); there is
    deliberately no per-agent run tool, since no such executor exists.
    """
    from backend.api import dispatcher as core

    return core.list_agents()


@mcp.tool(name="mech_experiment_list")
def mech_experiment_list() -> dict:
    """List stored experiments."""
    from backend.api import dispatcher as core

    return core.list_experiments()


@mcp.tool(name="mech_experiment_create")
def mech_experiment_create(experiment: dict) -> dict:
    """Store a new experiment record."""
    from backend.api import dispatcher as core

    if not isinstance(experiment, dict):
        return _err("experiment must be an object")
    return core.create_experiment(experiment)


# ── Runtime (probed status, request logs — same as Health view) ───────────────


@mcp.tool(name="mech_runtime_status")
def mech_runtime_status() -> dict:
    """Probed engine reachability (local/distributed/k8s/slurm/ray)."""
    from backend.api import dispatcher as core

    return core.runtime_status()


@mcp.tool(name="mech_runtime_logs")
def mech_runtime_logs(limit: int = 100) -> dict:
    """Recent backend request log (metadata only, no bodies)."""
    from backend.api import dispatcher as core

    return core.list_logs(max(1, min(limit, 1000)))


# ── Git + tests + builds (allowlisted commands, contained dirs) ──────────
#
# There is deliberately no general shell tool. Every command below is built
# from a fixed menu in backend.mcp_server.allowed_commands; the caller picks
# a runner/target plus tightly validated arguments, never an executable.


def _git() -> Optional[str]:
    return shutil.which("git")


def _contained_pathspec(base: Path, path: str) -> str:
    """Resolve a git pathspec against `base`, refusing escapes."""
    if not isinstance(path, str) or not path.strip() or len(path) > 1024:
        raise ValueError("path must be a non-empty string")
    if path.strip() == ".":
        return "."
    candidate = Path(path.strip())
    target = candidate if candidate.is_absolute() else base / candidate
    try:
        resolved = target.resolve()
    except OSError as exc:
        raise ValueError(f"path cannot be resolved: {exc}") from exc
    try:
        resolved.relative_to(base)
    except ValueError:
        raise ValueError(f"path is outside the working directory: {path}")
    return str(resolved)


@mcp.tool(name="mech_git_diff")
def mech_git_diff(cwd: str = ".", path: str = ".") -> dict:
    """git diff (read-only) inside a contained working directory."""
    if _git() is None:
        return _err("git binary not found on PATH")
    try:
        base = ws.resolve_contained(cwd)
        spec = _contained_pathspec(base, path)
    except ValueError as exc:
        return _err(str(exc))
    try:
        proc = subprocess.run(
            ["git", "diff", "--", spec], cwd=str(base),
            capture_output=True, text=True, timeout=60, shell=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return _err(f"git diff failed: {exc}")
    return _ok(cwd=str(base), returncode=proc.returncode,
               diff=proc.stdout[-50_000:], stderr=proc.stderr[-MAX_LOG_LINES:])


@mcp.tool(name="mech_git_commit")
def mech_git_commit(cwd: str, message: str) -> dict:
    """git add -A + commit inside a contained working directory.

    Blast radius: stages *all* changes under cwd, not a single file. Call
    mech_git_diff first and commit only from a cwd whose full diff you intend.
    """
    if _git() is None:
        return _err("git binary not found on PATH")
    if not isinstance(message, str) or not message.strip():
        return _err("message must be a non-empty string")
    if len(message.strip()) > 1000:
        return _err("message exceeds 1000 characters")
    try:
        base = ws.resolve_contained(cwd)
    except ValueError as exc:
        return _err(str(exc))
    try:
        add = subprocess.run(
            ["git", "add", "-A"], cwd=str(base),
            capture_output=True, text=True, timeout=60, shell=False)
        if add.returncode != 0:
            return _err(f"git add failed: {add.stderr[-500:]}")
        proc = subprocess.run(
            ["git", "commit", "-m", message.strip()], cwd=str(base),
            capture_output=True, text=True, timeout=120, shell=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return _err(f"git commit failed: {exc}")
    return _ok(cwd=str(base), returncode=proc.returncode,
               stdout=proc.stdout[-5_000:], stderr=proc.stderr[-MAX_LOG_LINES:])


@mcp.tool(name="mech_test_run")
def mech_test_run(runner: str = "pytest", args: Optional[list] = None,
                  cwd: str = ".", timeout_s: int = 300) -> dict:
    """Run an allowlisted test suite in a contained dir (pytest | vitest).

    Example: {"runner": "pytest", "args": ["tests/pytest/test_protocol.py", "-q"]}.
    The executable argv is built by backend.mcp_server.allowed_commands; test
    paths must already exist inside cwd. Flags: pytest -q, -x, --version,
    --collect-only, -k <expr>; vitest -q, -x, --version, -k <expr>.
    """
    from backend.mcp_server import allowed_commands as ac

    try:
        base = ws.resolve_contained(cwd)
    except ValueError as exc:
        return _err(str(exc))
    return ac.run_test(runner, list(args or []), base, timeout_s)


@mcp.tool(name="mech_build_run")
def mech_build_run(target: str = "renderer", timeout_s: int = 900) -> dict:
    """Run an allowlisted build (renderer: npm run build:renderer).

    The working directory is fixed to the frontend directory, not
    caller-supplied.
    """
    from backend.mcp_server import allowed_commands as ac

    return ac.run_build(target, timeout_s)


def main() -> None:
    """Serve this control plane over stdio (agent harness wiring)."""
    import asyncio

    asyncio.run(mcp.run_stdio_async())


if __name__ == "__main__":
    main()
