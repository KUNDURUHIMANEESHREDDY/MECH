# MECH MCP control plane

External AI agents control MECH through MCP tools, not by clicking its UI.
The UI, MCP, and CLI all drive the **same Core API** (`backend/api/dispatcher`).

```text
                External AI Agent
                       │  MCP
                       ▼
             ┌───────────────────┐
             │ backend.mcp_server │  FastMCP "mech-control", 20 tools
             └─────────┬─────────┘  direct calls, no HTTP loopback, no UI
                       ▼
             ┌───────────────────┐
             │  MECH Core API     │◄── Vue UI (fetch /api/*)
             │ backend/api/...   │
             └───────────────────┘
```

## Transports

| Mode | How | Use |
|------|-----|-----|
| stdio | `python -m backend.mcp_server.server` | Codex / Claude / OpenCode agent wiring |
| HTTP | `http://127.0.0.1:8000/mcp/` (mounted by `backend/main.py`) | remote agents, same loopback-only bind as `/api` |

## Tools (20)

| Group | Tools |
|-------|-------|
| project | `mech_project_list`, `mech_project_add`, `mech_project_open` |
| file | `mech_file_list`, `mech_file_read`, `mech_file_write` |
| loop | `mech_loop_list`, `mech_loop_start`, `mech_loop_stop`, `mech_loop_status`, `mech_loop_events` |
| agent | `mech_agent_list` |
| experiment | `mech_experiment_list`, `mech_experiment_create` |
| runtime | `mech_runtime_status`, `mech_runtime_logs` |
| git | `mech_git_diff`, `mech_git_commit` (contained pathspec, capped message) |
| test | `mech_test_run` (allowlisted runners pytest/vitest, validated paths/flags) |
| build | `mech_build_run` (renderer only, fixed frontend dir) |

## Deliberately absent (capability absent, not degraded)

- `mech_loop_pause` / `mech_loop_resume` — the engine has no pause primitive.
- `mech_agent_run` / `mech_agent_stop` — agents run as society roles inside a
  loop; execution goes through `mech_loop_start`. See `mech_agent_list` doc.
- `mech_loop_stop` is **cooperative**: post-cancel events are dropped and the
  run reports `cancelled`, but an in-flight model call runs to completion.
  The tool says so in its own response.
- `mech_shell_exec` was **removed on purpose** (arbitrary argv is not a
  capability the control plane offers). Tests run via `mech_test_run`
  (allowlisted pytest/vitest runners) and builds via `mech_build_run`
  (renderer only).

## Safety boundary

- All file/shell/git paths must resolve inside `MECH_WORKSPACE_ROOTS` or
  the repo root. `mech_project_open` may only record a directory *inside*
  that boundary — it can never enlarge it. Traversal, `..` escapes, and
  symlink/junction escapes are refused with a reason, never silently
  remapped. To open a directory outside the repo, set `MECH_WORKSPACE_ROOTS`
  to include it first.
- The HTTP transport binds to the same loopback interface as `/api`
  (`127.0.0.1` unless `MECH_BIND_HOST` is set deliberately) and requires the
  same bearer token (`Authorization: Bearer <MECH_API_TOKEN>`). Loopback and
  Origin are never authorization.
- Mutations are audit-visible through the backend request log
  (`mech_runtime_logs` / Health view).

## Example agent session

```text
mech_project_open("frontend")
mech_file_read("frontend/src/services/api.ts")
  → analyze bug
mech_file_write("frontend/src/services/api.ts", <fixed>)
mech_test_run("vitest", ["src/services/api.test.ts"], "frontend")
  → tests pass
mech_loop_stop("training-01")
mech_loop_start("retrain policy", "gpt2")
mech_loop_events(<new runId>)
```

## Verify

```bash
pytest tests/pytest/test_mcp_control_plane.py -q
```

Covers: registry matches live capabilities (and the honest absences),
containment refusals, project-open boundary (no self-expanding roots),
allowlisted test/build runs (off-menu runners, flags, and paths rejected),
file round-trip, empty-goal
rejection without spawning, unknown-run stop error, cooperative-cancel
semantics, and live project/runtime probes. No test loads model weights.
