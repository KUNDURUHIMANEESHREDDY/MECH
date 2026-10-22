# Changelog

All notable changes to MECH are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

No version has been tagged yet: `git tag` is empty and `pyproject.toml` declares
`0.1.0`. Entries therefore accumulate under `Unreleased` until the first release
is cut; nothing here is invented to fill a released-version heading.

## [Unreleased]

### Added

**API + CLI — the `.interp-project` project export** (`backend/api/export_contracts.py`,
`backend/api/project_export.py`, `backend/cli/main.py`)

- `POST /api/exports/{project_id}` and `POST /api/v1/exports/{project_id}`
  exist now, mounted from the same contract-router loop as the other
  contract families (`backend/main.py`), so both prefixes reach the same
  implementation. The handler is a plain `def` — it does blocking file IO,
  so FastAPI runs it in a threadpool — and it takes no destination from
  the client: the body model forbids unknown fields, so a request
  carrying `dest_dir` is rejected rather than quietly ignored. The
  response names the archive relative to the export root and never by an
  absolute server path.
- `mech projects export <project_id>` (`--estimate`, `--yes`) is the CLI
  form. It calls the same library functions as the route; there is no
  second implementation of the archive format and no `--dest` flag.
- A project that no stored session names does not exist. The export is
  refused (`404 project_not_found` over HTTP, exit `4` in the CLI) and no
  file is written. Before this, an export of an unknown project reported
  success and wrote an archive with nothing in it.
- Large exports need confirmation. `MECH_EXPORT_CONFIRM_BYTES` (default
  100 MiB) is the threshold; a missing, malformed or negative value falls
  back to the default. An estimate above the threshold is refused with
  `409 confirmation_required`, carrying the estimate and the threshold,
  until the caller confirms (`{"confirm": true}` over HTTP, `--yes` in the
  CLI). The estimate is computed from the exact bytes written as
  `sessions.json`, so it cannot drift from the archive.
- The archive is written to a temporary file in the destination directory
  and renamed onto its final name (spec §20.1). A failure while writing
  leaves no archive behind, not a truncated one where a finished archive
  was expected.
- The default export root moved from `Path.cwd()/exports` to
  `Path.home()/.mech/exports`, still overridden by `MECH_EXPORT_ROOT`. The
  packaged app runs with the working directory set to the install
  directory, so the old default was unwritable there and would have
  produced 500s — and archives written into the install directory are
  destroyed by the next upgrade.

**API — spec §21.1 endpoint families** (`backend/api/v1_contracts.py`,
`backend/api/artifact_registry.py`)

- `GET /datasets` — real `DatasetManager` + on-disk `dataset.json` inspection with
  sha256 integrity; a missing hash is reported as `not_recorded`, and an empty
  directory yields an empty list with an explicit status rather than a
  placeholder row.
- `POST /datasets/preview` — at most 20 rows / 50 columns read from a real file,
  with that file's digest. Paths resolving outside the workspace are `400
  outside_workspace`; non-UTF-8 and non-row files are structured `4xx`.
- `GET /artifacts/{id}` and `GET /artifacts/{id}/content` — disk-backed manifest
  under `backend/storage/artifacts` (`MECH_ARTIFACTS_DIR`), digest of bytes
  actually written, unknown id `404`, content served only when the file
  re-hashes correctly.
- `POST /interventions/validate` — preflight only (`executed: false`), validates
  indices against the loaded model config or the adapter catalogue; an
  unsupported kind is refused with reasons, never answered "ok".
- `GET /experiments/{id}/report` — reproduces stored records only; a run with a
  NULL `primary_score` is listed under `not_measured`, never as zero, and a
  reached linkage cap is stated in `notes`.
- Mounted at **both** `/api` and `/api/v1` (`backend/main.py:346-371`), same
  router, so the frontend keeps calling `/api/*` and the spec's `/api/v1/*` also
  resolves.

**API — spec §21.2 job event stream** (`backend/api/events.py`)

- `GET /events/schema`, `GET /events`, `GET /events/stream` (SSE).
- Versioned envelope (`EVENT_VERSION = 1`): `event_id`, `event_version`,
  `job_id`, `experiment_id`, `timestamp`, `type`, `stage`, `message`,
  `progress`, `data`, `sequence`.
- Events are observations of the queue's persisted state; a job that never ran
  cannot produce a success event. `progress` is `null` when no fraction exists —
  never `0` to mean unknown.
- Monotonic `sequence` with `Last-Event-ID` / `?since=` resume, bounded retained
  buffer, unknown `job_id` → `404` before the stream opens.

**API — runtime job routes on the versioned path**

- The single runtime router is now mounted at `/api/v1` **and** `/api/v2`
  (`backend/main.py:332-340`), so `GET/POST /api/v1/jobs` and
  `/api/v1/jobs/{id}/cancel` exist alongside the older `/api/v2` paths. One
  `QUEUE` behind both.

**Interpretability — NNsight adapter** (`backend/interpretability/nnsight_adapter.py`)

- Lazy-loading adapter with `describe()` / `capabilities()`, trace-based request
  construction (`build_trace_request`) and execution (`run_trace`).
- `nnsight` is not installed and is not a declared requirement, so the adapter
  reports `available: false`, `version: null`, every operation `false`, and
  `status: NOT_EXECUTED` with a reason naming the package. Request construction
  still works so a caller can inspect what a run would send.
- Absence (fail-closed) is cached as a negative, not as a success; the trace
  execution path is written to NNsight's documented API but has **not** been
  exercised against an installed `nnsight` in this environment.

**CLI — spec §22** (`pyproject.toml`, `backend/cli/main.py`)

- `mech` console script (`mech = "backend.cli.main:main"`).
- Subcommands: `doctor`, `models list`, `models inspect <id>`, `run forward`,
  `run logit-lens`, `experiments list`, `experiments show <id>`,
  `experiments export <id> --format json|markdown`, `sae list`, `sae inspect
  <id>`, `cache status`.
- Each command calls the same backend service the UI calls
  (`backend.api.dispatcher`, `backend.services.gpt2_engine`,
  `backend.interpretability.sae`, `backend.storage.DesktopStorage`); no
  inference or metric is reimplemented.
- Exit codes: `0` ok, `1` error, `2` usage, `3` NOT_EXECUTED, `4` not found. An
  unavailable engine or checkpoint exits non-zero and prints `NOT_EXECUTED` with
  the reason, never a placeholder table or zero. No command downloads weights.
- `pyproject.toml` declares no runtime dependencies (`requirements.txt` stays the
  install manifest) and no pytest configuration, so collection is unchanged.

**Frontend — spec §7.1 IA routes**

- Registry entries for `activations` (Activations), `logitlens` (Logit Lens),
  `saelab` (SAE Lab), `datasets` (Datasets) added to
  `frontend/src/desktop/routeRegistry.ts`, with kebab-case aliases and
  `requirement` flags matching real backend needs.
- Four view components added and wired to the real routes through
  `frontend/src/services/interpretabilityClient.ts`:
  `ActivationBrowserView.vue` (`POST /api/gpt2/activations`, then
  `/gpt2/layer_activations` per layer), `LogitLensView.vue`
  (`POST /api/gpt2/logit_lens_all`, labelled as a projection at the point of
  use), `SAELabView.vue` (`GET /api/sae/status` then `POST /api/sae/inspect`,
  with an explicit NOT AVAILABLE state), `DatasetsView.vue`
  (`GET /api/datasets`, `POST /api/datasets/preview`). Numeric values come only
  from responses; an absent endpoint renders the unsupported state rather than
  an empty chart.

**Documentation — spec §34**

- `docs/security.md`, `SECURITY.md`, `docs/data-and-privacy.md`,
  `docs/supported-models.md`, `docs/experiments.md`, `docs/logit-lens.md`,
  `docs/sae-lab.md`, and this file. All describe implemented behaviour, with
  partial and absent capabilities named as such.

### Changed

- The export's metadata now separates the two reasons a session was left
  out. `sessions_omitted_unattributed` counts sessions that name no
  project at all, and the new `sessions_omitted_other_projects` counts
  sessions that name a *different* project. They used to share one
  bucket, which misnamed the first and told the caller how many other
  projects exist. This changes a number inside the archive: the pinned
  assertion in `tests/pytest/test_broken_and_leaking_methods.py` moved
  from `sessions_omitted_unattributed == 2` to `1`, with
  `sessions_omitted_other_projects == 1`.

- `docs/` now distinguishes the three record types (experiment, run, job) and the
  two failure vocabularies (`NOT_EXECUTED` vs `UNAVAILABLE`/`FAILED`) explicitly
  rather than leaving them to be inferred.
- `docs/security.md` records the token also being accepted as a query parameter
  (`backend/main.py:184`) and the renderer holding the token
  (`frontend/electron/preload.js`, `main.js:61-70`) as stated limits, not
  incidents.

- **The control plane no longer accepts the bearer token in the query
  string.** `?token=` / `?bearer_token=` was accepted by the auth
  middleware and by `POST /session`, and uvicorn's access log writes the
  full query string -- so the credential reached stdout in cleartext, and
  the Electron app pipes backend stdout into its logger. The seam is now
  opt-in: `MECH_ALLOW_QUERY_TOKEN=1` (default off) keeps it reachable for
  the transports that cannot set a header (`EventSource`, `<img src>`).
  `tests/pytest/test_auth_control_plane.py` pinned the old behaviour as a
  feature; it now pins the new one, in both directions.
  (`backend/main.py`, `backend/core/auth.py`, `docs/auth.md`)

- **A cookie-authenticated request is now checked against its `Origin`.**
  `SameSite=Strict` does not stop a *same-site* request, and a site is
  (scheme, registrable domain) -- the port is not part of it. A page on
  `http://localhost:3000` is therefore same-site with
  `http://localhost:8000`: the browser attached the session cookie to its
  credentialed requests, and the CORS allowlist (which accepted every
  loopback port) made the responses readable. Any page on any loopback
  port had the operator's full control plane. The server now refuses a
  cookie-authenticated request whose origin it does not trust: the
  server's own origin, an origin listed in `MECH_TRUSTED_ORIGINS`, or --
  with `MECH_TRUST_LOOPBACK_ORIGINS=1` -- any loopback origin. The opaque
  `null` origin is refused. (`backend/main.py`, `backend/core/auth.py`,
  `docs/auth.md`)

- **`MECH_CORS_ORIGINS` entries are validated, and an added origin no
  longer carries credentials.** An entry that is not a well-formed origin
  (a path, a query string, a fragment, a wildcard, credentials in the
  URL) is refused with a warning, and an origin added to the allowlist no
  longer receives `Access-Control-Allow-Credentials` unless it is also
  listed in `MECH_TRUSTED_ORIGINS`. (`backend/main.py`,
  `backend/core/auth.py`)

- **The body-size ceiling sits behind the auth gate, and every response
  carries CORS headers.** The middleware stack is stated once and asserted
  by the tests: CORS outermost, then the auth gate, then the body
  ceiling, then the request log. An anonymous oversized POST is answered
  with 401 before its body is read (it used to be answered with 413 after
  the server had buffered megabytes for an anonymous caller), and a 401
  now carries the CORS headers a browser client needs in order to read
  the refusal. (`backend/main.py`)

- **The MCP HTTP transport at `/mcp` works.** It was mounted, but every
  authenticated request 500'd with "Task group is not initialized":
  FastMCP's streamable-HTTP app only serves requests from inside its
  session manager's task group, and a mounted sub-application never
  receives a lifespan of its own. The transport is now built and started
  by the application's lifespan. (`backend/main.py`)

- **One FastAPI application in the tree.** `backend/interpretability/api.py`
  built a second one, with no bearer gate, no CORS policy and no
  body-size ceiling. Nothing mounted it, nothing outside its own package
  imported it, and the endpoints it served were mock data, so it was
  deleted rather than re-mounted. (`backend/interpretability/`)

- **A failed evidence write no longer reports a plain success.** A
  Society run whose durable evidence record could not be written is
  reported as `completed_unrecorded`, with the reason, instead of
  `completed`: the work ran, the proof of it did not, and the two must
  not be confused. The exception is logged, not swallowed.
  (`backend/api/dispatcher.py`)

- **The dead in-process plugin loader is gone.**
  `PluginManager.load_active_plugins` executed plugin code inside the
  backend process and had zero callers anywhere in the tree; the
  production path is `PluginService.enable` -> `start_remote_plugin`, a
  separate worker process. (`backend/plugins/plugin_manager.py`)

### Fixed (this branch, described because they change documented behaviour)

- Logit lens no longer applies `ln_f` twice to the final layer. Probabilities,
  not just token identity, are now correct; the returned list is explicitly
  marked `probability_is_truncated` and carries `probabilities_reported_mass`
  (`backend/services/gpt2_engine.py:1794-1838`). Correct is to float32
  resolution: the lens and the model's own logits are the same quantity put
  through two matmul shapes over the same weights, so the two agree to about
  1e-6 rather than to the last bit.
- A local SAE checkpoint identifier that is a Windows path is refused on every
  host, not only on Windows. The check used `os.path.basename`, which splits on
  the host's separator, so on POSIX `C:\Windows\sae.pt` was a bare filename
  and resolved as a relative name inside the checkpoint directory while the
  same request was refused on Windows: one identifier, two behaviours,
  depending on where the server runs.
- SAE checkpoints are centred on the checkpoint's own `b_pre`, falling back to
  `b_dec` for `mats_sae_training`/Cunningham format, and
  `SAEConfig.pre_encoder_center` records which was used
  (`backend/interpretability/sae/loader.py`).
- SAE inspection refuses rather than returning constant features; reconstruction
  fidelity is reported as unmeasured when the checkpoint has no decoder
  (`backend/api/dispatcher.py:1867-1869`).
- Plugin install confined to trusted roots; plugin `name`/`entry` validated;
  sandbox escape via `collections._sys` closed at the dynamic capability layer.

### Frontend verification pass (desktop shell, all 36 tools)

The redesign was verified against a running backend rather than rebuilt. Every
change below was found by a check that could fail, and is pinned by one.

- **The sidebar status pill lied about readiness.** It probed
  `GET /api/models`, which serves a static reference catalogue
  (`provenance: "reference"`) that answers even when no weights are cached, and
  it rendered that catalogue's length as "N models available" - a catalogue
  length presented as research capability. The shell now probes
  `GET /api/status` and reports the four documented states
  (connecting / ready / disconnected / failed) from
  `frontend/src/services/runtimeStatus.ts`, which existed but was never
  imported. A refused token is reported as unauthorized, a backend that
  answers and then stops is reported as disconnected rather than as a backend
  that never existed, and a Retry control re-probes. Pinned by
  `frontend/tests/playwright/runtime-status.spec.js` (6 specs) and
  `frontend/tests/playwright/trust.spec.js`.
- **Clicking a tool could show the previous tool's results.** Vue's
  `<Suspense>` keeps the last resolved subtree on screen while the next async
  component loads, so the address bar and the panel label said "Circuit
  Explorer" while Model Explorer's neurons were still rendered underneath. The
  shell now resolves a route's chunk before switching, and shows an explicit
  busy state while it does. Pinned by the rewritten "switching tools preserves
  no stale data" spec.
- **The dev server reloaded the page 648 times a minute.** Vite watched
  `frontend/build/python/**` (a bundled CPython with thousands of
  site-package files), so any pip install or app build reloaded the app per
  file and reset the active route mid-click - every navigation looked broken.
  `frontend/vite.config.mts` now ignores `build/`, `release/`, `dist/`,
  `.venv/`, `test-results/` and `*.log`.
- **A browser could not authenticate at all.** The renderer holds no token by
  design (the Electron main process signs requests), so every `/api` call from
  `npm run dev:renderer` answered 401 and the Playwright specs could never see
  live data. The dev proxy now signs with `MECH_API_TOKEN` or the backend's own
  token file, read in the dev-server process. The token is not in the bundle:
  `tests/pytest/test_no_vite_token_in_bundle.py` still passes.
- **Counts derived from failed reads were rendered as 0.** Ten views
  (reports, analytics, benchmark, reasoning, evidence fusion, campaigns,
  circuit explorer and others) showed "0" next to an offline or failed read -
  a zero is a measurement, and there was none. They now show an em dash while
  loading, offline or partial.
- **Every tool rendered its own `<main>` inside the shell's `<main>`** (17
  views), giving screen-reader users duplicate main landmarks. The roots are
  now `<section>`; the shell keeps the single landmark.
- **The Research Society panel could not end.** `GET /api/society/stream`
  replays the trace and then holds the connection open with keep-alives: it
  never emits a terminal frame and never closes. A run that ended `blocked`
  left the panel reading "Society run in progress" indefinitely - measured at
  70s of a panel that spun while the backend had already reported the run
  finished. The view now runs a status watchdog alongside the stream and ends
  on the backend's own terminal status, with the backend's reason and no
  publication claim. Verified by `frontend/tests/e2e/live-society-check.mjs`.
- **The backend's persisted job queue had no interface at all.** The Jobs
  tool (`frontend/src/services/jobs.ts`, `frontend/src/components/JobsView.vue`)
  now lists the real queue, shows the backend's own bounds, kinds, statuses,
  errors and results, submits the two kinds the backend implements, and
  cancels a live job - a 409 (already finished) is reported, never turned into
  a successful cancellation. It displays no progress percentage, ETA or stage
  list, because the backend sends none.
- **Verification harnesses** were added under `frontend/tests/e2e/`:
  `route-sweep.mjs` clicks all 36 sidebar entries in both a live and an
  offline backend and reports mounted views, failed API calls, duplicate
  landmarks and fabricated values; `live-research-check.mjs`,
  `live-neuron-check.mjs` and `live-society-check.mjs` compare what the screen
  shows against what the API returned.
- **The e2e suite no longer asserts claims the system must not make.** The
  Society spec required `Backend status: completed` from every run, and the
  analytics spec required an "unavailable" string from a view that has live
  records; both now follow the run and the data. Stale selectors
  (`explorer-load-model`, `explorer-prompt-input`) were repaired against the
  current controls, and the Playwright budget was raised because a cold route
  chunk can take longer than the old 30s timeout to compile.

### Gaps closed after the frontend verification pass

Four gaps the verification pass reported were open. All four are closed, each
with a check that can fail.

- **A hung Society run locked the tool out until the backend restarted.**
  `_society_runs` was the only admission ledger: admission counted
  `status == "running"`, and nothing checked the worker thread or a deadline,
  so a run whose pipeline hung held one of the two admission slots forever and
  every later `POST /api/society/run` was refused with `active run limit
  reached (2/2)`. `backend/api/dispatcher.py` now gives every admitted run a
  deadline (`MECH_SOCIETY_RUN_TIMEOUT_S`, default 600s) after which it is
  finalized as a failure naming the timeout; a run whose worker thread is
  gone without a result is finalized `interrupted`; the admission counter
  reaps before it counts, and `GET /api/society/runs/{run_id}` never reports
  `running` for a run that can no longer progress. The 2-run limit itself is
  unchanged - it was deliberate. New route `POST /api/society/runs/{run_id}/cancel`
  (404 unknown, 409 already finished, cooperative otherwise), mirroring
  `cancel_job` in `backend/api/runtime_api.py`. Covered by
  `tests/pytest/test_society_run_lifecycle.py`.
- **The Society panel's Stop button did not stop anything.** It stopped the
  panel's own polling and told the user the backend had no cancellation
  endpoint, while the run kept going and kept its slot. It is now "Cancel run"
  and calls the new endpoint, distinguishing "cancellation requested" from
  "cancelled" from "already finished" (a finished run is re-read and shown
  with its real result instead of a cancellation that did not happen). The new
  backend status `interrupted` is treated as a failure, not a completion.
- **The Society lifecycle is back in the standard suite.**
  `frontend/tests/playwright/society-lifecycle.spec.js` submits a run and
  checks the parts that must hold on every run: the panel shows the run the
  backend created, the panel's status is the backend's status, no report, gate
  or publication appears while the run is in flight, and "Cancel run" performs
  the operation it advertises - the backend records the cancellation request,
  and the panel never claims the run was cancelled while the backend still
  reports it as running. A run the backend refuses (it admits two at once) is
  skipped with a visible reason instead of failing on back-pressure. The
  *completion* path -- a run that ends `blocked`, with the backend's reason --
  takes two minutes on an idle CPU and eight or more on a loaded one, so it
  is owned by `frontend/tests/e2e/live-society-check.mjs`, which has that
  budget. The duplicate Society spec in `trust-online.spec.js` was removed:
  two specs starting a run each would race for the same two admission slots.
- **SAE was only verifiable in its "no checkpoint configured" state.** The
  platform now inventories the checkpoints that exist
  (`backend/interpretability/sae/local_checkpoints.py`), `GET /api/sae/status`
  reports `checkpoint_identifier`, `checkpoint_dir` and `available_checkpoints`,
  and the SAE Lab view offers the real files as a labelled select and sends
  the chosen one. `MECH_SAE_CHECKPOINT=<file name>` now resolves inside the
  checkpoint directory the way the variable name implies, instead of against
  the server's working directory. Reconstruction fidelity is measured when the
  checkpoint carries a decoder - it was hardcoded to `False`, which reported a
  computable quantity as absent - and the route states when a checkpoint
  records no hook point, so the loader cannot refuse a layer mismatch and the
  reader can see that the number refers to a tensor that may not be the one
  the dictionary was trained on. `docs/sae.md` and
  `frontend/tests/e2e/live-sae-check.mjs` cover it; the harness asserts that
  every rendered feature and activation is the value the API returned, and
  that nothing is rendered that the API did not return.
- **Forty-five dead files were removed** from `frontend/src/`: the four
  pre-redesign shell components (`Sidebar.vue`, `Topbar.vue`, `StatusBar.vue`,
  `ActivityBar.vue`) and the legacy service/util/domain modules that nothing
  imported after the rebuild. Reachability was established from
  `frontend/src/main.ts` and cross-checked against the tests, the Electron
  preload and the docs before deletion.

- The locally trained SAE checkpoints under
  `backend/storage/sae_checkpoints/` record no hook point, and they were
  trained on a block-0 MLP contribution plus the residual stream, while
  `POST /api/sae/inspect` encodes the residual stream at the requested layer.
  The loader cannot refuse that mismatch, and the measured reconstruction
  error is correspondingly poor (relative L2 0.97 for the layer-8 sample
  checkpoint). The response carries that caveat; the numbers are real
  measurements of a dictionary applied to a tensor it was not trained on.
### Known incomplete in this tree

- `SAEInspector.inspect_feature` remains unimplemented and returns
  `unavailable` with its reason.
- `mech doctor` prints "weights load on demand from local files (never
  downloaded)" for the engine checkpoint, but `gpt2_engine.load()` falls back to
  a Hugging Face download when the local cache is empty. The doctor string, not
  the engine, is wrong.
- The NNsight execution path is written against NNsight's documented trace API
  but has not been exercised against an installed `nnsight`; absence is the only
  verified state.

### Migrations and compatibility notes

Database migrations run automatically and in place on startup
(`DesktopStorage.initialize()` → `_migrate`, `backend/storage/database.py:308-338`):

| Change | Effect on an existing database |
| --- | --- |
| `experiments.updated_at`, `sessions.updated_at` added | `ALTER TABLE ADD COLUMN`, then existing rows seeded `updated_at = created_at`. No data is dropped; `created_at` is never rewritten on subsequent updates. |
| `benchmark_runs` unchanged | existing run rows remain valid; `primary_score` stays nullable |
| `MECH_STORAGE_DB=:memory:` | now one shared in-memory database per process via a shared-cache URI, so the schema does not vanish between calls |

Other versioned artefacts:

| Artefact | Version | Note |
| --- | --- | --- |
| Job event envelope | `EVENT_VERSION = 1` (`backend/api/events.py`) | clients should pin it; a shape change bumps it |
| Artifact manifest | `manifest_version = 1` (`backend/api/artifact_registry.py`) | unknown versions raise rather than being guessed |
| API paths | `/api` and `/api/v1` both live | `/api/v2/jobs` continues to work; `/api/v1/jobs` is additive |

Behavioural compatibility notes for existing callers:

- `POST /sae/inspect` now returns `unavailable` + `reconstruction_measured:
  false` where it previously returned invented features. A caller that treated
  any 200 as success must read `status`.
- `logit_lens` response gained `probability_is_truncated`,
  `probabilities_reported`, `probabilities_total` and
  `probabilities_reported_mass`. The `prob` values changed for the final layer
  because of the `ln_f` fix; if you cached numbers from before, they were wrong.
- `mech` exits non-zero (3) for a capability that is absent. Scripts that
  checked only for output text should check the exit code.
