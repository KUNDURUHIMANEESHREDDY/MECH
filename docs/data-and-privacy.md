# Data and privacy

MECH stores everything on the machine it runs on. There is no account, no
server-side component, and no telemetry SDK. This document lists the exact paths,
what each holds, what can leave the machine, and how to delete it. Paths are
relative to the repository root unless absolute; every one is the value used by
the cited module, not an idealised layout.

## 1. Storage map

| Path | Contents | Created / owned by | Override |
| --- | --- | --- | --- |
| `backend/storage/mech.db` | settings, recent projects/files, experiments, sessions, plugin registry rows, benchmark runs | `backend/storage/database.py` | `MECH_STORAGE_DB` |
| `backend/storage/.mech_api_token` | control-plane bearer token, one line | `backend/core/auth.py` | `MECH_API_TOKEN` (value, not path) |
| `backend/storage/evidence/*.json` | one JSON envelope per evidence record (`r<hex>.json`) | `backend/core/evidence_graph.py` | `MECH_EVIDENCE_DIR` |
| `backend/storage/sae_checkpoints/` | local SAE checkpoint files | `backend/api/dispatcher.py:1612` | `MECH_SAE_DIR` |
| `backend/storage/artifacts/` | `manifest.json` + content-addressed artifact bytes | `backend/api/artifact_registry.py:77-82` | `MECH_ARTIFACTS_DIR` |
| `backend/storage/v2_jobs/` | one JSON file per runtime job | `backend/api/runtime_api.py:73-77` | `MECH_V2_JOB_DIR` |
| `backend/research_datasets/` | dataset directories with `dataset.json` manifests | `backend/api/v1_contracts.py:128-132` | `MECH_DATASETS_DIR` |
| `backend/plugins/installed/` | `<name>.py` plus `registry.json` | `backend/plugins/plugin_manager.py:12` | — (fixed) |
| `backend/runtime.db` | runtime/executor state | present in the checkout as runtime state | — |
| `exports/` | user-requested exports (archives, code listings) | export paths chosen by the caller | — |
| `~/.cache/neural-debugger` | default cache directory in settings | `DEFAULT_SETTINGS["cachePath"]` | app settings |
| Hugging Face cache (e.g. `~/.cache/huggingface`) | downloaded GPT-2 and SAE weights | `transformers`, `huggingface_hub` | `HF_HOME` |

SQLite runs in WAL mode (`database.py:235`); `mech.db-wal` and `mech.db-shm`
appear beside the database while it is open. Shutdown checkpoints the WAL
(`backend/main.py` lifespan), so a clean exit does not leave a large `-wal`
behind.

Virtualenvs, `frontend/node_modules/`, build output and `*.db` are gitignored.
The token file is runtime state and is never committed.

## 2. What each store actually contains

**SQLite (`mech.db`).** Seven tables (`database.py:237-296`):

| Table | Row shape |
| --- | --- |
| `settings` | key/value JSON, timestamps |
| `recent_projects` | path, name, opened_at |
| `recent_files` | path, project path, opened_at |
| `experiments` | `item_id`, whole payload as JSON, created_at, updated_at |
| `sessions` | same shape as experiments |
| `plugins` | name, path, manifest JSON, enabled flag, installed_at |
| `benchmark_runs` | run_id, benchmark, metric, score, status, provenance, model id, weights fingerprint, git commit, timestamps, traces JSON, detail JSON |

Experiments and sessions are opaque JSON: whatever the client posted is what is
stored (`add_experiment` / `add_session`). MECH does not strip prompt text or
notebook content out of them.

**Evidence records.** Each `<run_id>.json` is a signed measurement envelope:
model identity and content hash, execution context, evidence level, payload
commitment. Records that fail the legacy check are moved into a quarantine
directory by `quarantine_legacy_records()` at startup and are never served as a
current run again (`evidence_graph.py:878-958`).

**Artifacts.** `manifest.json` maps a SHA-256 digest to a record; bytes live
under the same root. The digest is of bytes actually written, and content is
only served when the file re-hashes to its id. An unknown id is a `404`.

**Jobs.** One JSON file per runtime job in `v2_jobs/`, containing kind, payload,
status, timestamps, result and error. Jobs are the queue's own state, not a
research record.

**Datasets.** `<dir>/dataset.json` describes a dataset; previews are read from
the file's own bytes and are not persisted. `POST /datasets/preview` refuses
paths outside the workspace root (`MECH_WORKSPACE_ROOT`, default the repository
root).

## 3. What never leaves the machine

- Experiment, session, run and evidence records.
- Prompts you type, unless you use the Research Society's agent layer with an
  engine you configured (see §4).
- Datasets, dataset previews, artifact bytes and SAE checkpoints.
- The API token, the SQLite database, plugin source, and logs.
- Anything the app writes to `exports/`.

There is no telemetry: no analytics SDK, no crash reporter, no phone-home, no
"usage ping" (`backend/` has no `posthog`, `sentry_sdk` or equivalent import).

## 4. What can leave the machine

Exactly three things, all pull-based and all operator-triggered:

| Egress | Trigger | Where |
| --- | --- | --- |
| GPT-2 weights and tokenizer | first `load()` when the local cache is empty | `backend/services/gpt2_engine.py:117-127` — tries `local_files_only=True` first, then downloads from the HF Hub |
| SAE checkpoints | `POST /sae/inspect` with a remote source, or an explicit registry/HF fetch | `backend/interpretability/sae/loader.py:531-594` |
| Prompt text to an LLM engine | only when `MECH_LLM_ENGINE` names an engine **and** the agent layer runs | `backend/agents/llm/engine_factory.py` |

`MECH_LLM_ENGINE` is unset by default, which means *no engine*, and the Society's
hypothesis step reports `NOT_EXECUTED` rather than inventing one
(`engine_factory.py:16-31`). Setting it to `ollama` keeps generation on
`http://127.0.0.1:11434` by default; setting it to `openai` sends the prompt to
`https://api.openai.com` with `OPENAI_API_KEY`. That is a real egress of research
text and it is the operator's decision, not a default.

Other model families (`backend/science/models/small_models.py`) additionally
require `MECH_ALLOW_MODEL_DOWNLOAD=1` before they will touch the network.

## 5. Retention and deletion

There is no automatic expiry. Nothing prunes `experiments`, `sessions`,
`benchmark_runs`, `evidence/`, `v2_jobs/` or `exports/` on a timer.

| To delete | Do this |
| --- | --- |
| One experiment | `DELETE /api/experiments/{id}` — `database.py::delete_experiment` |
| One session | `DELETE /api/sessions/{id}` |
| One benchmark run | `database.py::delete_run` |
| One plugin | `DELETE /api/plugins/{name}` — unregisters, then unlinks the file |
| Everything | stop the app and delete `backend/storage/mech.db*`, `backend/storage/evidence/`, `backend/storage/artifacts/`, `backend/storage/sae_checkpoints/`, `backend/storage/v2_jobs/`, `backend/plugins/installed/` |

Two bounds exist but they are memory bounds, not retention policy:
`_SAE_MAX_RUNS = 20` caps in-memory SAE training records
(`dispatcher.py:1621`), and the event stream keeps a bounded ring buffer of 1000
events (`backend/api/events.py`). Neither deletes persisted files.

Deleting the token file regenerates a new token on the next start; the Electron
preload and `main.js` read the file each launch, so no reinstall is needed.

## 6. Privacy stance

- **Local by default.** Loopback bind, local SQLite, local weights cache.
- **No account, no sync, no cloud.** Exposing the API with `MECH_BIND_HOST` is a
  networking choice; it does not create a remote service or send data anywhere
  new. See [security.md](security.md) §2 for the cleartext consequence.
- **No telemetry, now or opt-in.** Nothing in the backend reports usage. If this
  ever changes it must be opt-in and documented here first.
- **Export is explicit.** Files under `exports/` exist because a caller asked for
  them; MECH does not scan them for secrets ([security.md](security.md) §10).
- **Model weights are third-party data.** Downloads are cached by
  `transformers`/`huggingface_hub` under their own cache, subject to their
  retention, not MECH's.

## See also

- [security.md](security.md) — trust boundaries and what is not protected.
- [experiments.md](experiments.md) — the provenance fields carried on runs.
- [sae-lab.md](sae-lab.md) — checkpoint provenance and integrity.
