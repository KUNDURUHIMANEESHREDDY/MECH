# Experiments, runs and jobs

Three different records, three different lifespans. Mixing them is the most
common way a result gets read as more than it is, so this page keeps them
separate and states exactly where each lives and what each carries.

| Record | What it is | Where it lives | Lifespan |
| --- | --- | --- | --- |
| **Experiment** | a named container with an id and an opaque JSON payload | `experiments` table in `backend/storage/mech.db` | until deleted |
| **Run** | one benchmark measurement with provenance | `benchmark_runs` table, same DB | until deleted |
| **Job** | one unit of queued work and its outcome | `backend/storage/v2_jobs/<job_id>.json` | queue's own state; not a research record |

## 1. Experiment

`GET /api/experiments` returns `{"experiments": [...]}` (whole stored payloads).
`POST /api/experiments` stores the body verbatim and assigns an id when the
client did not supply one: `exp_<hex>` from `backend/core/identifiers.py`
(`backend/api/dispatcher.py:480-488`). `DELETE /api/experiments/{id}` returns
`{"status": "deleted" | "not_found", "id": ...}` — a missing id is a normal
answer, not an error.

Storage is deliberately schema-free: `item_id`, `payload` (JSON text),
`created_at`, `updated_at` (`backend/storage/database.py:257-262`). `created_at`
is never rewritten on update — only `updated_at` moves — so list order means
creation order (`database.py:749-752`).

Because the payload is opaque, **MECH does not validate provenance inside an
experiment**. If your client records model, prompt and seed, they are preserved;
if it does not, they are absent. The run record is where provenance is
structured and enforced.

Run linkage is by convention, in either of two places:

- `run_ids: [...]` on the experiment record, or
- `detail.experiment_id` on the run.

`GET /experiments/{id}/report` (mounted at `/api` and `/api/v1`) is the only
thing that reads both. It scans at most the 50 explicit `run_ids` and the 500
most recent runs, and **says so in `notes` when either cap was reached** rather
than presenting a partial scan as an empty store
(`backend/api/v1_contracts.py:900-1005`).

## 2. Run

A run is written by `record_run(...)` (`backend/storage/database.py:767`) and is
the only record with enforced provenance fields:

| Field | Required | Notes |
| --- | --- | --- |
| `run_id` | no | generated (`uuid4().hex`) if omitted; `INSERT OR REPLACE`, so re-recording an id overwrites the earlier attempt |
| `benchmark` | **yes** | non-empty string |
| `status` | **yes** | non-empty string; free-form, not constrained to the vocabulary below |
| `provenance` | **yes** | non-empty string; conventionally `live` / `seeded` / `reference` / `unavailable` |
| `primary_metric` | no | metric name |
| `primary_score` | no | **number or `None`**. A non-number raises `StorageError`; `None` is stored as SQL NULL |
| `model_id` | no | |
| `weights_fingerprint` | no | from `gpt2_engine.weights_fingerprint()` |
| `git_commit` | no | |
| `traces` | no | JSON list, bounded by `_dumps_traces` |
| `detail` | no | JSON object |

`primary_score = None` is the load-bearing rule: a run that never measured the
metric stores NULL and reads back NULL. Substituting `0.0` would turn "not
measured" into a measured zero, which downstream comparisons and charts would
treat as evidence (`database.py:782-803`).

Reads: `DesktopStorage.history(benchmark=None, limit=20)` and
`_get_run(run_id)`; HTTP surfaces are `GET /api/benchmarks/history` and
`GET /api/benchmarks/compare`.

## 3. Job

`backend/api/runtime_api.py` owns a process-wide queue (`QUEUE`) exposed at
**both** `/api/v1/jobs` and `/api/v2/jobs` — one router mounted twice, so there
is exactly one queue behind both paths (`backend/main.py:332-340`).

| Field | Meaning |
| --- | --- |
| `job_id` | `j` + 12 hex chars |
| `kind` | handler name; unknown kinds fail with "no handler for job kind" |
| `payload` | the submitted request |
| `status` | `queued`, `running`, `completed`, `failed`, `cancelled`, `unavailable`, `interrupted` |
| `created_at` / `started_at` / `finished_at` | UTC ISO-8601 |
| `result` / `error` | exactly one is populated at terminal state |
| `cancel_requested` | flag, checked by handlers that cooperate |
| `max_concurrency` | concurrency the queue was configured with |

Terminal states are `completed, failed, cancelled, unavailable, interrupted`
(`runtime_api.py:49-51`). A job persisted as non-terminal at startup is marked
`interrupted` and is **not resumed** — the reason is recorded on the record.

Bounds: `MECH_V2_MAX_CONCURRENCY` (1–8, default 2) and `MECH_V2_MAX_QUEUED`
(1–512, default 64). Submitting into a full queue raises rather than silently
dropping the request.

The mapping from handler result to job status is conservative
(`runtime_api.py:354-376`): a handler status of `unavailable`, `blocked` or
`not_run` becomes job `unavailable` — **not** `completed` — with the handler's
own reason as `error`. `error`/`failed` becomes `failed`; `cancelled` becomes
`cancelled`; only anything else becomes `completed`.

## 4. Status vocabulary: NOT_EXECUTED vs FAILED vs UNAVAILABLE

`backend/core/execution_status.py` defines the contract:

| Status | Means | Carries data? |
| --- | --- | --- |
| `IMPLEMENTED` | registered, ran, produced data | yes |
| `NOT_EXECUTED` | registered but not implemented — **nothing ran**; a property of the build | **no** |
| `UNAVAILABLE` | implemented but cannot run now (e.g. no weights loaded); transient | no |
| `FAILED` | ran and failed; transient and diagnosable | no |
| `RUNNING` | still going | — |

`not_executed_payload(reason, ...)` builds the refusal and **asserts** that no
data field leaked into it; `DATA_FIELDS` lists the keys that would mean
something was measured (`execution_status.py:60-101, 127-153`). A `NOT_EXECUTED`
payload that contains `top_token`, `activations`, `logit_diff`, `observed_metrics`
or any other data field raises `AssertionError`.

Why the distinction is kept: an engine that was never written is a permanent
property of the build; an engine that failed has a log, a retry and a fix.
Collapsing both into `unavailable`/`error` lets a permanent gap disappear into
the same channel as an outage — which is exactly what the module records having
happened (70 `unavailable` sites, 55 `error` sites, three distinct situations).

`UNIMPLEMENTED_CAPABILITIES = {deepseek, mistral, cloud, cluster}`
(`execution_status.py:117-124`) is the verified gap list; `CapabilityRegistry`
reports `False` for those and `NOT_EXECUTED` for anything unregistered.

Note the layering: the *job* status vocabulary is lower-case and separate. A
`NOT_EXECUTED` engine result surfaced through a job arrives as job status
`unavailable` with the reason in `error`.

## 5. Event stream

`backend/api/events.py` turns queue observations into a versioned event stream.
Events are derived from the persisted job state — they are not a second source
of truth, so a job that never ran can never produce a success event.

| Surface | Purpose |
| --- | --- |
| `GET /events/schema` | machine-readable envelope schema and type list |
| `GET /events` | retained events, oldest first (`?after=`, `?job_id=`, `?limit=`) |
| `GET /events/stream` | Server-Sent Events |

Envelope, versioned with `EVENT_VERSION = 1`: `event_id`, `event_version`,
`job_id`, `experiment_id`, `timestamp`, `type`, `stage`, `message`, `progress`,
`data`, `sequence`.

- `sequence` is process-wide and strictly increasing; `Last-Event-ID` or
  `?since=` resumes from it.
- `progress` is a float in `[0, 1]` **or null**. The queue reports no fraction
  today, so real events carry `progress: null`; zero is never used for
  "unknown".
- `type` is selected from the observed `status` only (`job.queued`,
  `job.started`, `job.succeeded`, `job.failed`, `job.cancelled`,
  `job.unavailable`, `job.interrupted`); an unmapped status emits `job.status`
  and says so in `stage`.
- The retained ring buffer is bounded (1000 events). A client that has fallen
  off it recovers by reading `GET /api/v2/jobs/{job_id}`, not by assuming
  continuity.
- An unknown `job_id` is a `404` *before* the stream opens.

## 6. Comparison

`GET /api/benchmarks/compare?run_a=<id>&run_b=<id>`
(`backend/benchmarking/run_history.py::compare_payload`).

- Unknown run id → `404`.
- `comparable` is a summary boolean; `not_comparable` carries the reasons
  verbatim.
- `score_delta` is `None` exactly when the runs are not comparable. A zero would
  read as "no change", which the stored rows may not support.
- Nothing is recomputed: the delta comes from the stored `primary_score` values.

A run with `primary_score = NULL` is not comparable — it has no score — and is
reported as such rather than as zero.

## 7. Export

| Path | Status |
| --- | --- |
| `mech experiments export <id> --format json` | **works** — prints the stored record as JSON |
| `mech experiments export <id> --format markdown` | **works** — prints a Markdown table of stored fields plus the raw record, stating that the exporter computed nothing |
| `POST /api/exports/{project_id}` (also `/api/v1/exports/{project_id}`) | **works** — writes a `.interp-project` archive of that project's sessions into the export root and answers with the archive name, its size and its sha256. A project no stored session names is `404 project_not_found` and nothing is written; an export estimated above `MECH_EXPORT_CONFIRM_BYTES` (default 100 MiB) is `409 confirmation_required` until the request carries `{"confirm": true}` |
| `mech projects export <project_id>` | **works** — the same archive from the command line. `--estimate` prints the estimate and writes nothing; `--yes` confirms an export above the threshold; an unknown project exits non-zero |
| `GET /experiments/{id}/report` | **works** — stored-record report (see §1) |

`mech experiments export` reads `DesktopStorage` and prints; it does not write a
file unless you redirect stdout. The project archive is different: it writes a
file, into the export root -- `~/.mech/exports` by default (override
`MECH_EXPORT_ROOT`), a user-writable directory outside the install directory,
because the packaged app runs with the working directory set to the install
directory and archives written there are destroyed by upgrades. The archive is
written to a temporary file and renamed into place, so a failed export leaves no
half-written archive behind.

An archive carries that project's sessions and a `metadata.json` that says what
was included, what was omitted and why, when it was written and a digest of
every member. Reports are not in it: the storage authority has no `reports`
table and no project link, so the archive records that they were left out
rather than dropping the file silently.

Reports and exports **reproduce** stored rows. They do not compute new values,
and they do not raise an evidence level: an exported run keeps whatever
`provenance` and evidence level it was stored with.

## 8. Provenance fields you will see

| Field | Values |
| --- | --- |
| `provenance` | `live` (measured in this process), `seeded` (fixture/seed data), `reference` (published/registry value), `unavailable` (nothing was measured) |
| `evidence_level` | `OBSERVATIONAL` (1) → `ATTRIBUTIONAL` (2) → `INTERVENTIONAL` (3) → `CAUSALLY_VALIDATED` (4) → `REPLICATED` (5) (`backend/core/provenance.py:644-656`) |
| `field_provenance` | per-field map, so one payload can mix live and reference fields honestly |
| `measured` / `validation_eligible` / `publication_eligible` | booleans set by the engine; a refusal sets all three to `false` |
| `weights_fingerprint` | hash-derived identity of the weights, not the model name |
| `model_identity` | includes `revision` (`config._commit_hash`) and `content_hash_source` |

A result that cannot declare where it came from is a bug, not a convenience.

## See also

- [security.md](security.md) — auth on these routes.
- [data-and-privacy.md](data-and-privacy.md) — where the rows physically are.
- `docs/SCIENTIFIC_VALIDATION.md`, `docs/scientific_methods.md`,
  `docs/reproducibility.md` — the validation vocabulary these fields feed.
