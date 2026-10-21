# Security model and trust boundaries

MECH is a single-user desktop research tool: one FastAPI process, one Electron
renderer, one local SQLite database, one bearer token. It is not a multi-tenant
service and it does not implement users, roles, or per-object authorization.
This document states what the boundary actually is, where it is enforced, and
what it does **not** protect. Everything below was read from the code paths
cited; where a control is partial, it says so.

## 1. Trust boundary map

| Boundary | Enforced in | Mechanism |
| --- | --- | --- |
| Network reach | `main.py:44`, `backend/main.py` `__main__` | loopback bind by default; `MECH_BIND_HOST` opt-in |
| HTTP authorization | `backend/main.py::_auth_middleware`, `backend/core/auth.py` | one bearer token, constant-time compare |
| Browser origin | `backend/main.py::_cors_origins`, `_LOOPBACK_ORIGIN_REGEX` | exact allowlist + anchored loopback regex |
| Request size | `backend/main.py::_body_size_middleware` | 2 MiB default, streamed byte count |
| Renderer ↔ main process | `frontend/electron/main.js`, `preload.js` | `contextIsolation: true`, `nodeIntegration: false` |
| Plugin code | `backend/plugins/plugin_sandbox.py`, `plugin_manager.py`, `service.py`, `limits.py` | AST gate + restricted namespace, trusted-root containment, out-of-process worker with OS resource limits |
| Filesystem paths | `backend/mcp_server/workspace.py`, `backend/api/v1_contracts.py`, `backend/api/dispatcher.py` | `resolve()` then containment check |
| Deserialization | `backend/interpretability/sae/loader.py`, `algorithms/tuned_lens.py` | `torch.load(..., weights_only=True)`, JSON manifests |

The process boundary is the only real boundary. Inside it, MECH trusts its own
code, its own storage, and anything running as the same OS user.

## 2. Local API: loopback by default, exposure is opt-in

`python main.py` binds `127.0.0.1:8000` (`main.py:44`, default `"127.0.0.1"`).
`python backend/main.py` does the same with a literal, non-configurable host.

`MECH_BIND_HOST` widens the bind. The comment at `main.py:36-43` is the
contract: binding `0.0.0.0` puts a privileged control plane on the LAN for
anyone holding the token, so the wider bind is a deliberate operator choice
(containers, shared dev machines), never an automatic one. There is no
`0.0.0.0` fallback and no environment variable that disables auth.

The API is plain HTTP. There is no TLS anywhere in the stack, so an exposed
bind means the bearer token and every response travel in cleartext.

## 3. Bearer authentication

`backend/core/auth.py` is the single source of truth.

| Fact | Value / behaviour |
| --- | --- |
| Token source 1 | `MECH_API_TOKEN`, when set and non-empty |
| Token source 2 | `backend/storage/.mech_api_token`, generated with `secrets.token_urlsafe(32)` |
| Token file mode | `chmod 0o600` attempted; on Windows the `chmod` is a no-op and the file is protected only by profile-directory ACLs (`auth.py:83-92`) |
| Comparison | `hmac.compare_digest` |
| Path configurable? | No. `MECH_TOKEN_FILE` was deliberately removed so an attacker-set env var cannot redirect where the token is read from (`auth.py:60-71`) |
| Lifecycle | Created at first `get_bearer_token()`, cached in memory for the process |

**Public paths** are `/`, `/health`, CORS preflight (`OPTIONS`), and static
frontend assets identified by file extension (`auth.py:153-167`). Everything
else returns `401`: `/api/*`, `/api/v1/*`, `/api/v2/*`, `/mcp`, `/health/*`
(including `/health/subsystems`), `/openapi.json`, `/docs`, `/redoc`.

`/openapi.json`, `/docs` and `/redoc` are explicitly checked *before* the static
extension rule, so `.json` cannot be used to bypass auth on the schema.

Three properties are worth stating plainly:

- **Loopback reachability is not authorization, and neither is Origin.** Both
  are documented as non-credentials in `auth.py:1-11` and enforced by
  `_auth_middleware`, which runs before any route.
- **The token is also accepted as a query parameter** (`token` or
  `bearer_token`) — `backend/main.py:184`. This exists so the SSE stream can be
  opened by clients that cannot set headers. It means the credential can appear
  in browser history, shell history and proxy logs. Prefer the header.
- **One token, all privileges.** There is no scoping. A token that can read
  `/api/health` can also install and enable plugins, write experiments, delete
  them, and trigger model runs.

## 4. CORS

`_cors_origins()` returns `["null"]` plus any exact origins listed in
`MECH_CORS_ORIGINS` (comma-separated, trailing slash stripped). `"null"` is
required because Electron loads the packaged renderer from `file://`, which
browsers report as the opaque origin `null`.

`_LOOPBACK_ORIGIN_REGEX = ^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$` is
anchored, so `http://localhost.evil.com` and `http://127.0.0.1.attacker.org` do
not match. Methods are restricted to `GET, POST, PUT, PATCH, DELETE, OPTIONS`;
headers to `Content-Type, Authorization, Accept`; `allow_credentials=True`.

Consequence: any web page served from a `file://` context, or from any loopback
port, is a permitted origin. CORS is not an authorization mechanism here — the
bearer token is. A page that somehow learns the token is not stopped by CORS.

## 5. Request size and input bounds

| Bound | Default | Override | Location |
| --- | --- | --- | --- |
| Whole request body | 2 MiB | `MECH_MAX_BODY_BYTES`, clamped to 1 KiB–64 MiB | `backend/main.py:198-268` |
| Prompt text | 2048 chars | — | `backend/api/dispatcher.py:1190` |
| SAE training corpus | 200k chars | — | `backend/api/dispatcher.py:1616` |
| SAE checkpoint file | 512 MiB | — | `backend/api/dispatcher.py:1614` |
| Dataset preview | 20 rows, 50 columns | clamped, reported via `truncated` | `backend/api/v1_contracts.py:63-67` |

The body middleware counts *streamed bytes*, not `Content-Length`, so a lying or
chunked header cannot slip a large payload past it; oversized bodies get `413`,
a malformed `Content-Length` gets `400`.

## 6. Electron IPC

`frontend/electron/main.js:52-58` sets `contextIsolation: true`,
`nodeIntegration: false`, `sandbox: false`. The renderer therefore has no
Node.js access; everything goes through the `appApi` object exposed by
`contextBridge` in `frontend/electron/preload.js:39-95`.

- New windows are denied and handed to `shell.openExternal`
  (`main.js:84-87`), so the renderer cannot spawn its own BrowserWindow.
- The preload script resolves the API token from `MECH_API_TOKEN`, then from
  candidate `.mech_api_token` paths, and exposes it as
  `window.__MECH_API_TOKEN__` (`preload.js:7-37`).
- `main.js:61-70` injects the same token into the page after `dom-ready` **and
  writes it to `localStorage` as `mech_api_token`**.

That last point is the most important thing in this section: **any script
executing in the renderer can read the control-plane token.** XSS in a view, a
malicious plugin contributing UI, or a compromised renderer dependency is
equivalent to full API access. `contextIsolation` limits Node.js escape; it does
not protect the token from in-renderer script.

The preload surface is a fixed vocabulary of `ipcRenderer.invoke` channels
(settings, projects, sessions, experiments, GPT-2 runs, logs, `openExternal`,
`showInFolder`). It is not a generic "invoke any channel" bridge.

## 7. Plugin install and path policy

Install is `POST /api/plugins/install` with a local directory path
(`backend/api/dispatcher.py:697-703` → `backend/plugins/service.py`).

| Control | Implementation |
| --- | --- |
| Remote installs | refused: any `://`, `http:`, `https:`, `ftp:` prefix (`service.py:120`) |
| Trusted roots | bundled `backend/plugins/library/` plus `MECH_PLUGIN_TRUSTED_ROOTS` (os.pathsep-separated), resolved and compared by ancestor relation (`service.py:96-141`) |
| Manifest `entry` | resolved and required to stay inside the plugin directory (`service.py:66-78`) |
| Plugin name | `^[a-z0-9][a-z0-9_.-]{1,63}$`, re-checked at install and again in `PluginManager.install_plugin` (`plugin_manager.py:42-45`) |
| Overwrite | refused; an installed name must be uninstalled first (`plugin_manager.py:47-53`) |
| Write target | `backend/plugins/installed/<name>.py`, with an absolute-path containment assert before writing (`plugin_manager.py:63-67`) |
| Registry | `backend/plugins/installed/registry.json` |
| Static scan | `PluginSandbox.analyze_ast` rejects forbidden names (`eval`, `exec`, `open`, `__import__`, `getattr`, …), forbidden attributes (`__class__`, `__subclasses__`, `__globals__`, …), non-allowlisted imports, and live-module escape attributes on allowlisted stdlib modules (`plugin_sandbox.py:98-160`) |
| Runtime confinement | `load_from_file_sandboxed` re-runs the AST gate and executes the module with a restricted `__builtins__` in a namespace carrying only `__file__` (`plugin_loader.py:116-174`) |

**Where plugin code actually runs.** Enabling an installed plugin does *not*
execute it in the backend process. `PluginService._load_enabled` starts a
separate worker (`backend/plugins/runner.py`, `worker.py`) and returns a
`RemotePluginProxy` that forwards each hook over a JSON channel; plugin code
never runs in the backend process (`backend/plugins/service.py:162-191`). The
worker is the real containment boundary, bounded by OS-level limits
(`backend/plugins/limits.py`):

| Platform | Bounds |
| --- | --- |
| POSIX | `setrlimit` for CPU seconds (120), address space (4 GB), file size (64 MB), open files (256), process count (64); `setsid` so the worker cannot signal the backend's process group |
| Windows | a Job Object bounding active processes, per-process and total job memory and CPU time, with `KILL_ON_JOB_CLOSE` so the worker cannot outlive the backend |

Limits are clamped *down*, never raised: a caller may not lift its own
containment cap. A platform that refuses a limit degrades to a logged warning,
so a missing capability is visible rather than silent.

The AST gate and restricted `__builtins__` are defence in depth in front of
that: `PluginSandbox.analyze_ast` rejects forbidden names (`eval`, `exec`,
`open`, `__import__`, `getattr`, …), forbidden attributes (`__class__`,
`__subclasses__`, `__globals__`, …), non-allowlisted imports, and live-module
escape attributes on allowlisted stdlib modules (`plugin_sandbox.py:98-160`).
`load_from_file_sandboxed` re-runs the gate and executes with a restricted
namespace carrying only `__file__` (`plugin_loader.py:116-174`) — that path *is*
in-process, and is not the one the service uses.

The module's own comments record why the static layer cannot be the whole
story: a name-level rule cannot tell a module from a field, so the dynamic
capability layer is what holds the line (`plugin_sandbox.py:136-157`). Treat
"install only directories you trust" as the real policy.

## 8. Workspace containment

Paths that arrive from a client are resolved and then checked for containment,
not string-matched:

- `backend/mcp_server/workspace.py::resolve_contained` / `register_open_root`
  resolve the candidate (following symlinks and junctions) and require it to be
  relative to an allowed root; relative paths resolve against the repo root, not
  the process cwd. Operating on `/` or a sibling directory is rejected with the
  reason and the required environment change.
- `backend/api/v1_contracts.py::_resolve_workspace_path` refuses any preview
  path that escapes the workspace root with a structured `400`
  `outside_workspace`, and a missing file is a `404`.
- `backend/api/dispatcher.py::_resolve_local_checkpoint` accepts **bare
  filenames only** inside the configured SAE checkpoint directory. Absolute
  paths, `..`, hidden names, UNC shares and separators are `400`; the resolved
  path must remain inside the directory after symlink resolution. The `env:`
  indirection is honoured only from the server environment, never from a
  request payload, so a caller cannot aim the loader at an arbitrary server
  path (`dispatcher.py:1643-1679`).

## 9. Serialization safety

- SAE checkpoints are read with `torch.load(path, map_location="cpu",
  weights_only=True)`. A checkpoint that needs arbitrary classes is refused with
  the reason rather than retried with `weights_only=False`
  (`backend/interpretability/sae/loader.py:396-428`).
- Tuned-lens translator weights are loaded the same way
  (`backend/interpretability/algorithms/tuned_lens.py:709`).
- Plugin registries, artifact manifests and dataset manifests are JSON.
- There is no `pickle.load`, `yaml.load` or `eval()` on request data in the
  backend.

`weights_only=True` is not a style preference: `weights_only=False` is a pickle
load, and a checkpoint is exactly the kind of file that arrives from elsewhere.

## 10. What this model does NOT protect

State each of these as a known limit, not a bug report:

1. **No per-user authorization.** One token authenticates one principal with all
   privileges. There is no ownership check on experiments, artifacts, sessions or
   plugins.
2. **No rate limiting.** No route enforces a request rate; the only throttles are
   the body cap and per-feature queue limits (e.g. SAE training holds a single
   slot, `dispatcher.py:1620`).
3. **No transport security.** Plain HTTP, and the token may be carried in a query
   string.
4. **Same-user processes are inside the boundary.** Anything running as the
   operator can read `backend/storage/.mech_api_token` and call the API.
5. **The renderer holds the token.** See §6. Renderer-side XSS is control-plane
   compromise.
6. **The plugin worker is bounded, not jailed.** Resource limits and a separate
   process contain runaway CPU/memory/processes, but the worker still runs as the
   same OS user, sees the same filesystem, and is not network-isolated. See §7.
7. **No secret redaction guarantee for exports.** Export paths are chosen by the
   caller; MECH does not scan an export for credentials before writing it.
8. **No audit trail of *who*.** The request log (`backend/core/request_log.py`,
   written by `backend/main.py:148-164`) records method, path, status and
   duration — no bodies, no identity. With one principal there is no identity to
   record.
9. **`MECH_BIND_HOST` has no allowlist.** Any interface/host string uvicorn
   accepts is accepted; exposure is the operator's decision and the process
   prints the host it bound at startup (`main.py:45`).

## 11. Reporting

Security reports go through a GitHub private security advisory on
`KUNDURUHIMANEESHREDDY/MECH`. See [SECURITY.md](../SECURITY.md).

## See also

- [data-and-privacy.md](data-and-privacy.md) — what is stored locally and what
  leaves the machine.
- [api_reference.md](api_reference.md) — the auth summary as it appears in the
  API reference.
- `docs/audit/source-stability-guard.md` — the test-run integrity guard.
