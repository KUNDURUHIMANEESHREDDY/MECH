# Security & Integrity Remediation — Change Log

Audit: architecture + security + scientific-integrity review of the MECH
backend (502 files). Work was done as ten sequential fixes, each following
red → green, then a two-axis review (Standards + Spec) and a fix pass on the
findings.

**Three companion files, same directory:**
- `security_remediation_code.txt` — the **actual source code** of every change,
  grouped by fix, extracted verbatim from the live files (3 427 lines).
  Read this to see the code.
- `security_remediation_changes.patch` — the same work as a `git apply`-able
  diff (44 files: unified diffs for tracked files vs `b19f393`, full contents
  for the 13 new files).
- this file — the what/why narrative and the remaining-work list.

**The code changes are in `security_remediation_changes.patch`** (same
directory) — 44 files: unified diffs for tracked files against base commit
`b19f393`, plus full contents for the 13 new files. Verified by applying it
to a clean detached worktree of `b19f393` and running the new suites there
(71 passed), so it is self-sufficient rather than dependent on other
uncommitted work in the tree. Apply with:

```bash
git apply --ignore-whitespace docs/notes/security_remediation_changes.patch
```

`--ignore-whitespace` is required because this repo has `core.autocrlf=true`;
without it the CRLF worktree copies do not match the LF diff context.

Status at the end of this pass: **1483 tests passed, 5 failed**, all five
pre-existing and owned by other workstreams (see "Remaining failures").

Everything below is uncommitted working-tree state on top of `b19f393`.

---

## P0-1 — Mandatory bearer authentication

**Problem:** the REST and MCP control planes had no inbound authentication.
Anyone who could reach the local server had full control: plugin
install/enable, arbitrary workspace reads/writes, git commits, and arbitrary
process execution. CORS trusted loopback origins plus the opaque `"null"`
origin, so browser-origin policy was carrying the load that authentication
should have.

**Fix**
- `backend/core/auth.py` (new, 150 lines) — token source of truth, in order:
  1. `MECH_API_TOKEN` env var (operator / test override), else
  2. a random `secrets.token_urlsafe(32)` token persisted to
     `backend/storage/.mech_api_token` with `0600`, so the trusted Electron
     renderer can read it across restarts.
  Verification is constant-time (`hmac.compare_digest`); comparison never
  short-circuits on length. `extract_bearer` rejects non-Bearer schemes and
  empty credentials. `is_public_path` is the single allowlist decision.
- `backend/main.py` — `_auth_middleware` gates everything except `/`,
  `/health`, CORS preflight (`OPTIONS`), and the built frontend's static
  files. `/api/*`, `/api/v1/*`, `/mcp`, `/health/subsystems`, `/openapi.json`,
  `/docs`, `/redoc` all return `401` without a valid token. Startup creates
  the token and logs its location.
- `main.py` — the comment claiming "no authentication" corrected.
- `frontend/src/services/api.ts` — `apiToken()` / `authHeaders()` read the
  token from a global, `localStorage`, or `VITE_MECH_API_TOKEN`, and every
  request sends it. A 401 is surfaced as a specific error rather than a bare
  status.
- `.gitignore` — the token file is never committed.

**Proof:** before, `GET /api/status` anonymous returned `200`. After: `401
{"detail": "missing credentials"}`; with a valid token `200`; `/health`
stays public; `/health/subsystems` anonymous returns `401`.

**Tests:** `tests/pytest/test_auth_control_plane.py` (new, 10 tests) —
anonymous REST, anonymous MCP, invalid token, malformed headers, valid token,
public paths, constant-time check. Existing suites updated to authenticate:
`conftest.py` (shared `pytest-shared-token`), `test_protocol.py`,
`test_health_probes.py`, `test_advanced_evals.py`, `test_contract_schema_drift.py`,
`test_sae_routes.py`, `test_remediation_regression.py`, `test_ioi_name_defaults.py`,
`test_api_exposure.py`.

---

## P0-2 — MCP workspace containment cannot self-expand

**Problem:** `mech_project_open()` accepted any existing directory and called
`register_open_root()`, which added it to the trusted set that
`resolve_contained()` then enforced. The containment model was therefore
self-expanding: an MCP caller could authorize `/`, a home directory, or any
sibling simply by asking to open it — no bypass of `resolve_contained()`
required.

**Fix** — `backend/mcp_server/workspace.py` (68 → 102 lines)
- Split the immutable trust boundary (`configured_roots()`:
  `MECH_WORKSPACE_ROOTS` + repo root) from the mutable set
  (`workspace_roots()`: configured + opened).
- `register_open_root()` now requires the resolved candidate to sit inside
  `configured_roots()`; anything else raises `ValueError` before registration.
  Relative paths still anchor to the repo root.
- `clear_open_roots()` added for test isolation.

**Proof:** a shell probe registered an arbitrary system-temp directory as a
root (the hole), then confirmed rejection of the drive anchor, a system-temp
directory, and the repo parent, while the repo root itself and a configured
subdirectory still opened normally.

**Tests:** six new cases in `test_mcp_control_plane.py` — outside roots,
`..` escape, symlink escape, inside-repo accepted, configured subdir
accepted, and tool-level rejection before any registration. Home
rejected-by-default plus accepted-when-configured, same-parent sibling, and a
Windows junction escape were added after review.

---

## P0-3 — Arbitrary shell execution removed from MCP

**Problem:** `mech_shell_exec(cmd=[...])` passed any argv to
`subprocess.run(shell=False)`. `shell=False` prevents shell-string injection
but the *command* was still arbitrary — combined with P0-1's missing auth,
that was unauthenticated arbitrary process execution.

**Fix**
- `backend/mcp_server/allowed_commands.py` (new, 172 lines) — a deep module
  with a small interface. The caller chooses *what* to run from a fixed menu,
  never *how*. `build_test_argv` / `build_build_argv` are pure constructors;
  `run_test` / `run_build` validate, spawn with `shell=False`, cap timeouts,
  and truncate output.
  - runners: `pytest` (`python -m pytest`), `vitest` (`npm run test:js`)
  - build target: `renderer` only (`npm run build:renderer`, fixed frontend cwd)
  - test paths must already exist inside the contained cwd
  - flags: pytest `-q -x --version --collect-only -k <expr>`; vitest
    `-q -x --version -k <expr>`; `-k` values pattern-checked
- `backend/mcp_server/server.py` (384 → 311 lines) — `mech_shell_exec`
  **deleted**, replaced by thin `mech_test_run` / `mech_build_run` wrappers.
  `mech_git_diff` now resolves its pathspec against the base and refuses
  escapes; `mech_git_commit` caps messages at 1000 chars and its docstring
  states the full-cwd blast radius (`git add -A`).
- `docs/MCP_CONTROL_PLANE.md` — tool table updated (20 tools), the deliberate
  absence of `mech_shell_exec` recorded, safety boundary and example session
  rewritten.

**Proof:** `grep -rn mech_shell_exec backend` → no matches. Tool registry test
pins the new set and `HONESTLY_ABSENT` includes the removed tool.
`tests/pytest/test_mcp_control_plane.py`: 21 passed, 1 skipped (symlink
privilege on Windows).

---

## P1-4 — Run IDs cannot traverse

**Problem:** run records were written to
`os.path.join(evidence_dir, f"{run_id}.json")` with no validation, and
`GET /society/runs/{run_id}` fell back to `load_run_record(run_id)` for runs
missing from memory — so a path-like ID could read another JSON file the
backend account could reach.

**Fix** — `backend/core/evidence_graph.py`
- `RUN_ID_RE = ^r[0-9a-f]{12}$`, matching the real generator
  (`dispatcher.py` mints `"r" + uuid4().hex[:12]`). The audit's sketch
  (`^r_[a-f0-9]{32}$`) would have rejected every genuine run.
- `validate_run_id()` raises before any filesystem touch.
- `_record_path()` additionally resolves the target and verifies
  `target.relative_to(evidence_root)` before `open()` — defense in depth.
  `save_run_record` validates before even `makedirs`.

**Tests:** `tests/pytest/test_run_id_validation.py` (new, 37 tests) covering
`../`, `../../`, absolute POSIX/Windows, drive-qualified, UNC, URL-encoded
traversal, non-strings, legacy test-style IDs, generator conformance over 100
sampled IDs, and a route-level proof that a traversal ID returns
`unknown runId` rather than a sibling file's contents.

---

## P1-5 — Tamper-evident evidence envelope

**Problem:** `save_run_record` wrote raw JSON and `load_run_record` simply
parsed it — no signature, hash, or schema validation at the persistence
boundary, for records that feed scientific evidence.

**Fix** — `backend/core/evidence_graph.py`
- v1 envelope: `{schema_version: 1, run_id, created_at, record, attestation}`.
- Real Ed25519 via `backend/science/integrity/signing.py` — no default key
  ever. With no key configured the envelope is written unsigned
  (`attested: false`) with the reason stated, so records stay readable but can
  never pass as tamper-evident.
- The signed payload binds **schema version, run ID, timestamp, verifying
  key, and canonical record** (sorted-key compact JSON). Rewriting any of
  them invalidates the signature.
- `envelope_status()` verifies without serving, distinguishing `valid`,
  `invalid`, `missing`, `corrupt`, `legacy`, and quarantined; algorithm is
  enforced as Ed25519 on verification.
- `load_run_record` unwraps but never upgrades; `list_run_records` reads
  through envelopes under the envelope's own run ID.
- `dispatcher.py` — the persisted `/society/runs/{id}` fallback now carries
  `envelope_status`.

**Scope discipline:** a signature here proves only that *this backend wrote
these bytes*. It says nothing about whether the numbers inside are real
measurements — that is the eligibility layer's job, and the docstring says so.

**Tests:** `tests/pytest/test_evidence_envelope.py` (new, 13 tests) — signed
and unsigned round-trips, record tamper, run-ID mismatch, swapped key,
algorithm downgrade, timestamp rewrite, legacy, corrupt/missing, list
through-envelope, route status.

---

## P1-6 — Atomic evidence writes

**Problem:** `save_run_record` wrote directly to the target file, so a crash
mid-write left a partial record that readers would parse or choke on.

**Fix** — `backend/core/evidence_graph.py`
- Write to a uniquely named sidecar in the same directory (`mkstemp`) →
  `flush` → file `fsync` → best-effort directory `fsync` → `os.replace`
  under a process-wide `_save_lock`.
- Sidecar cleaned up on any failure.

**Two real bugs found by the tests and fixed:**
1. A fixed `.tmp` name collided across threads on Windows (`PermissionError`).
   Switched to unique-per-write sidecars.
2. Even then, same-target `os.replace` calls race on Windows file locking —
   measured `WinError 5` under a 10-thread hammer. Hence the lock.

**Honest limitation, stated in the comment:** the rename is atomic in the sense
that a concurrent reader sees the old file or the new one, never torn bytes.
Crash/power-loss durability of the rename is best-effort — `_fsync_dir` is a
no-op on Windows, where opening a directory fails.

**Tests:** `tests/pytest/test_evidence_atomicity.py` (new, 5 tests) — no
sidecar residue, stale-crash orphan ignored, 10-thread hammer stays complete,
500 KB record survives, and fault-injected crash mid-commit keeps the previous
record byte-identical.

---

## P1-7 — Historical evidence quarantined

**Problem:** the repo shipped 26 pre-envelope records in
`backend/storage/evidence/`, 9 of them `status: completed`, `provenance: live`,
`validation_eligible: true`, `publication_eligible: true` — presented on
restart as if they were current Society runs.

**Fix** — `backend/core/evidence_graph.py` + `backend/main.py`
- `quarantine_legacy_records()` moves every non-envelope `*.json` to
  `evidence_historical/` (override: `MECH_EVIDENCE_HISTORICAL_DIR`), stamped
  `origin: historical_fixture` with all eligibility flags forced `False` at
  top level and inside `result`. Unparseable bytes move **verbatim** (forensics
  over formatting). A destination equal to the source is refused before any
  file is touched. Runs idempotently at backend startup.
- The stamp is load-bearing, not decorative: `save_run_record` refuses to
  re-mint a stamped record; `load_run_record` and `list_run_records` refuse to
  serve or list anything carrying it, so a file copied back cannot pass as
  evidence; `envelope_status` reports `invalid` for a stamped record even
  under a valid signature.
- Audit surface: `list_quarantined_records()`, `load_quarantined_record()`,
  and `load_quarantined_file()` (bare-filename regex, so odd legacy names are
  still readable). Corrupt files list as `origin: "unparseable"` rather than
  vanishing.

**Tests:** `tests/pytest/test_evidence_quarantine.py` (new, 8 tests) —
move+stamp, idempotence, re-promotion refused, copied-back stamp not served,
self-destruct config refused, odd-name loader, verbatim corrupt bytes,
stamped-envelope reporting, and gates closed under `evidence_policy`.

---

## P1 — Provenance semantics: attested evidence

**Problem:** three related holes let a record look scientific without being
measured.

1. `dispatcher._mark()` accepted a caller-supplied `"live"` label gated only
   on `engine.is_available()` — which says torch *imported*, not that a
   forward pass ran.
2. `evidence_policy` predicates checked `status + provenance + both
   eligibility flags` but never `attested`.
3. `TraceableEvidenceGraph._step_allows_evidence()` returned `True` for every
   node that was not `discover`/`validate`, so an `executor` step with a
   numeric `delta` produced a live-labelled Evidence node — the laundering
   mechanism the graph's own comments described.

**Fix**
- `backend/core/provenance.py` — `attest_measurement` sets `attested: True`
  (fresh results, and repairs a bare `live` label that arrived unattested);
  `withhold` sets `attested: False` on every path; `attested` added to
  `_FRAME_KEYS` so it is not treated as a measured field.
- `backend/api/dispatcher.py` — `_mark` is now preserve-or-withhold: an engine
  label passes through untouched, an unlabeled record is withheld. The
  `"live"` argument is legacy dead weight, documented as such.
- `backend/agents/evidence_policy.py` — all four predicates
  (`discovery_is_live`, `validation_is_live`, `reproduction_is_live`,
  `gate_is_live`) plus `blocked_reason` now require `attested is True`.
- **Measurement layers assert** (they measured):
  `live_discovery`, `live_validation`, `benchmark_runner`, `ioi_pipeline`
  (three sites, mock-conditional), `logit_lens_pipeline`, `greater_than_pipeline`,
  `induction_heads_pipeline` (conditional on `measured`).
- **Wrappers propagate** (they relay): `discoverer`, `critic.validate`,
  `critic.reproduce`, `executor.reproduce`, `society` gate, the
  `/benchmarks/run` route, `scribe.report`, `scribe.publish` — all
  `.get("attested", False)`, never asserted.
- `critic.py:211` fixed: a metric-mapping failure returned
  `provenance: "live"` with `status: unavailable`, which is incoherent.
- `backend/core/evidence_graph.py` — `_step_allows_evidence` inverted to
  require step-`attested` + live provenance + a gated node
  (`discover`/`validate`) + policy pass. No generic numeric extraction implies
  evidence.
- `backend/services/gpt2_engine.py` — `_attest_info()` wraps `load()`/`info()`
  so a config-derived description of loaded weights is attested at the source.
  (Without this, `_mark`'s new withhold path made both model routes report
  `unavailable` even with weights loaded — a regression this change introduced
  and then fixed at the correct layer.)

**Chain, verified link by link:** engine `attest_measurement` → discoverer
spread → society trace step → graph + policy; validation engine → critic →
society gate → `publication_block_reason`; reproduction pipelines → critic
reproduce → society repro/gate → publication.

**Tests:** `tests/pytest/test_attested_evidence.py` (new, 13 tests) plus
honest fixture updates — `_live_payload` gained `attested`, the confidence
fixtures gained `attested`, two blocked cases added to the discoverer matrix,
and the two MCP modules I own were classified `LIVE_BY_CONSTRUCTION` in
`test_provenance_origination.py` (verified: they set no eligibility flag
anywhere).

---

## P1 — Runtime controls

**Admission control** — `backend/api/dispatcher.py`
- 2 active Society runs, 5 queued; goal ≤ 4096 chars; `model_name` ≤ 128.
- Refusal happens before any state is allocated and returns
  `{"status": "busy", ...}`.
- Retention (`_SOCIETY_MAX_RUNS = 50`) kept separate from concurrency.
- The queue cap fires on the queue alone — my first version used
  `active + queued >= MAX_ACTIVE + MAX_QUEUED`, which admitted a run when 5
  were already queued. Caught by the test, corrected.

**Body limits** — `backend/main.py`
- Global 2 MB cap (`MECH_MAX_BODY_BYTES`, clamped 1 KB–64 MB).
- Checks declared `Content-Length` *and* counts streamed bytes, so a chunked
  or lying header cannot slip a large payload past. Returns `413` before the
  body reaches the dispatcher.

**Model identity** — `backend/api/dispatcher.py`
- `GET /models/{name}` no longer returns fixed `12 / 768 / 50257 / 12` for any
  name. That answered for `gemma-2b` and `llama-3-8b`, which have different
  shapes and no loader here.
- Now: real config-derived dimensions for the loaded weights; `status:
  "unavailable"` with **no** dimensions when nothing is loaded; and for a
  mismatched name, `model_mismatch: true` with `model_requested` /
  `model_loaded` and an explicit reason.

**Tests:** `tests/pytest/test_runtime_controls.py` (new, 11 tests) — active
limit, queue capacity, finished runs don't block, goal and model-name caps,
declared and chunked body rejection, normal body still accepted, and three
model-identity cases.

---

## Test-fixture corrections (needed to stay honest)

Tests that encoded the old behaviour were updated, not weakened:
- `_live_payload` in `test_discoverer_spread_order.py` gained `attested: True`,
  plus two new blocked cases (`attested: False`, `attested: None`) — a payload
  that runs live but does not attest must not reach the verdict.
- The confidence fixtures in `test_confidence_defaults_are_not_fabricated.py`
  gained `attested: True`; without it `validation_is_live` correctly refuses
  and every case below the gate returns `False` vacuously.
- `test_get_model_info` in `test_advanced_evals.py` rewritten: it asserted
  the hardcoded literals I removed. It now accepts either an honest refusal or
  real config dimensions, and fails if the old keys reappear.
- The P0-2 junction test now removes its reparse point before its target —
  a dangling junction inside the workspace made the repo-wide import scanners
  raise `FileNotFoundError`.

## Docs
- `docs/api_reference.md` — authentication, request limits, and the
  attestation contract.
- `docs/MCP_CONTROL_PLANE.md` — 20-tool surface, `mech_shell_exec` removal,
  immutable workspace boundary, allowlisted test/build.

---

## Remaining failures (5, all pre-existing)

Verified not mine; each belongs to another workstream:

| Test | Cause |
|---|---|
| `test_entry_points.py::test_no_unannounced_unresolvable_imports` | trips on the gitignored `MECH-standalone/` build copy |
| `test_benchmark_report_honesty.py::test_no_unresolvable_imports_remain` | same `MECH-standalone/` copy |
| `test_standalone_launcher.py::test_health_is_accepted_only_when_it_reports_healthy` | untracked file from another workstream |
| `test_health_probes.py::test_evidence_probe_checks_the_boundary_imports` | red since P0-1; expects a removed `boundary` component |
| `test_provenance_origination.py::test_every_module_stamping_live_is_classified` | `backend/analysis/experiment_runner.py` and `backend/experiments/attention_experiment.py` stamp `provenance: "live"` (verified `HEAD=0`, i.e. another workstream); needs an honest classification pass of its own |

## Files touched by this work

New (13):
```
backend/core/auth.py                    backend/mcp_server/allowed_commands.py
backend/mcp_server/workspace.py         backend/mcp_server/server.py
docs/MCP_CONTROL_PLANE.md
tests/pytest/test_auth_control_plane.py       tests/pytest/test_mcp_control_plane.py
tests/pytest/test_run_id_validation.py        tests/pytest/test_evidence_envelope.py
tests/pytest/test_evidence_atomicity.py       tests/pytest/test_evidence_quarantine.py
tests/pytest/test_attested_evidence.py        tests/pytest/test_runtime_controls.py
```

Modified (31): `backend/core/provenance.py`, `backend/core/evidence_graph.py`,
`backend/agents/{evidence_policy,discoverer,critic,executor,society,scribe}.py`,
`backend/interpretability/discovery/live_discovery.py`,
`backend/validation/{live_validation,benchmark_runner}.py`,
`backend/science/reproducibility/{ioi,logit_lens,greater_than,induction_heads}_pipeline.py`,
`backend/services/gpt2_engine.py`, `backend/api/dispatcher.py`,
`backend/main.py`, `main.py`, `.gitignore`, `frontend/src/services/api.ts`,
`docs/api_reference.md`, and 12 test modules.

## Not yet done from the audit
- Plugin `rename`/`link`/`symlink` **destination** confinement (P2-14) — the
  capability hook validates `args[0]` only.
- Legacy `TraceableEvidenceGraph.save` / in-process plugin
  `load_from_file` removal (P2-15, P2-16).
- Runtime "reachable" → `installed / configured / reachable / execution_ready`
  (P2-18).
- Dual storage authorities (`DesktopStorage` vs `backend.core.database`)
  consolidation (P2-20).
- Persisted run ordering by `created_at` rather than filename (P2-20).
- Windows reparse-point / `O_NOFOLLOW` hardening of workspace writes (P2-19).