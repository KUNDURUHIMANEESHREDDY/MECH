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

### Fixed (this branch, described because they change documented behaviour)

- Logit lens no longer applies `ln_f` twice to the final layer. Probabilities,
  not just token identity, are now correct; the returned list is explicitly
  marked `probability_is_truncated` and carries `probabilities_reported_mass`
  (`backend/services/gpt2_engine.py:1794-1838`).
- SAE checkpoints are centred on the checkpoint's own `b_pre`, falling back to
  `b_dec` for `mats_sae_training`/Cunningham format, and
  `SAEConfig.pre_encoder_center` records which was used
  (`backend/interpretability/sae/loader.py`).
- SAE inspection refuses rather than returning constant features; reconstruction
  fidelity is reported as unmeasured when the checkpoint has no decoder
  (`backend/api/dispatcher.py:1867-1869`).
- Plugin install confined to trusted roots; plugin `name`/`entry` validated;
  sandbox escape via `collections._sys` closed at the dynamic capability layer.

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
