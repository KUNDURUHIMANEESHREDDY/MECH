# Audit remediation status

This records what was actually done to the findings in the external MECH audit,
what was deliberately not done, and — most importantly — **what has not been
verified**, because the machine was not permitted to run Python while this work
was carried out.

Read the verification section before trusting any of it.

---

## Verification status

Python was not run for any of this work. No pytest, no imports, no
compile-and-execute of project code.

**What *was* machine-checked:** `ruff 0.16.9` was installed as a standalone
binary — it does not need Python — and used to check syntax and name resolution
across `backend`, `tests` and `scripts`. That covers the failure modes that
would have made the work unrunnable:

* **Syntax errors (E9): zero**, in all 37 changed files and repo-wide.
* **Undefined names (F821): zero**, in all changed files, and repo-wide after
  the fixes below.

This is not the same as the tests passing. Ruff proves the code parses and that
names resolve; it says nothing about behaviour. The distinction matters:

| Risk | Status |
|---|---|
| A syntax error in a rewritten module | **Ruled out** by ruff (E9). |
| An undefined name, i.e. a latent `NameError` | **Ruled out** in changed files (F821). |
| An existing test that pinned old behaviour | Still open. Fails on the first run; cheap to find. |
| A behavioural regression in a test that was green | **Still open, and expensive to find.** The attestation work changes what is admissible as a live result at all. |
| A test that passes for the wrong reason | **Still open.** The failure mode this repository has been bitten by repeatedly. |

The previous full-suite figure, **858 passed / 2 skipped**, predates all of this
work and is not a current number.

### What ruff found

Worth recording, because it is the argument for having run it:

* `tests/pytest/test_no_hash_derived_identifiers.py` used `pytest.mark.parametrize`
  and `pytest.skip` with **no `import pytest`** — a `NameError` at collection,
  in a file written this session and reviewed by reading twice.
* `tests/pytest/test_discovery_no_fabrication.py` referenced an undefined
  `rel` inside an `assert` message. Since the message is only evaluated when the
  assertion fails, the guard reported `NameError` instead of its own diagnostic
  — precisely when the diagnostic was needed.
* Four modules annotated with typing names they never imported (`Set`, `Dict`
  twice, `Optional`). Harmless under `from __future__ import annotations`, but
  any runtime type introspection — `typing.get_type_hints`, which FastAPI calls
  on route handlers — raises `NameError`.

Two `F821` reports were left in place as false positives:
`test_challenger_m1_adversarial.py` reports `storage` as undefined at lines 329
and 340, but it is bound at 319 in the same function and read by nested
closures, which is valid Python. That test is in the passing baseline. Editing
working test code on the strength of a lint believed to be wrong is how a guard
gets broken.

### Tests that must be run, in this order

1. `tests/pytest/test_evidence_boundary.py` — the attestation rewrite changes
   what counts as a live result. Expect breakage here first.
2. `tests/pytest/test_plugin_read_confinement.py` — read allow-listing is the
   change most likely to break normal plugin operation, because it gates every
   file read a plugin makes.
3. `tests/pytest/test_no_hash_derived_measurements.py`,
   `tests/pytest/test_broken_and_leaking_methods.py`,
   `tests/pytest/test_no_hash_derived_identifiers.py`,
   `tests/pytest/test_project_validator_cannot_rubber_stamp.py`
4. `tests/pytest/test_non_gpt2_adapters_are_real.py` — see backlog #13 below.
5. The full suite, to re-establish a trustworthy number.

---

## Fixed

Each entry names the counterexample that was constructed, because that is what
distinguishes a fix from a claim.

### Evidence boundary — `462a9a5`

The audit's counterexample for the attestation, which verified under the old
shape-only check and became publishable:

```python
RunAttestation(run_id="anything", executor_id="anything", model_id="gpt2",
               weights_sha256="<64 hex>", dataset_sha256="<64 hex>",
               code_revision="anything")
```

and the counterexample for the result record, which was publishable without
`admit()` ever being called:

```python
EvidenceResult(status="completed", provenance="live", executor_id="x",
               eligibility={"publication_eligible": True})
```

Both are now refused. The attestation is an Ed25519 signature over a canonical
payload that includes a digest of the measurement itself, so it is bound to one
result; a bounded ledger stops one execution id attesting two measurements;
`EvidenceResult` requires a capability only the boundary holds.

**Known limitation, not a caveat to skip:** the capability closes the
*structural* loophole — no module can mint a live result with a dict literal. It
is not a defence against code that deliberately imports the private token, because
Python has no real private state.

### Plugin read confinement and Windows containment — `6cd22d5`

The escape: `Path("~/.ssh/id_rsa").read_text()`, which needed no network escape,
because the plugin IPC channel is already authorised and returned the content
inside an ordinary dict.

Reads are now allow-listed. The grant list is the plugin's own directory, temp,
the Python installation, site-packages, the model caches, and the interpreter's
import roots.

**Residual risk, stated rather than buried:** because the import roots are
granted so that `backend/plugins/library/ioi_experiment_logger` can still import
`backend.plugins.plugin_base`, a third-party plugin can read MECH's own source.
Host *data* — the database, results, artifacts — is outside the granted roots.
Denying the source too needs either a separately importable plugin API package
or dropping bundled plugins' access to it; both are larger changes and neither
could be validated here.

Windows containment failed open because the runner tested
`job.get("assigned") is False`, and `assign_windows_job` returns a result with
*no* `assigned` key when the Job Object types cannot be built. Now anything short
of an explicit `True` is a refusal, and the decision is the pure function
`containment_refusal()`.

### Three places dressing up arithmetic as measurement — `fadae6a`

The audit found none of these. They are the same class of defect.

* `RepresentationBenchmark.run_benchmark` ignored its argument and computed
  purity and completeness from `hash()` of its own task names — purity 0.88–0.97,
  `overall_rigor` near 0.85, `threshold` 0.75, so it passed by construction. And
  because Python salts string hashing per process, the numbers *moved between
  runs* while looking like measurements, so the fabrication also defeated
  reproducibility checking.
* `ProvenanceService.record_provenance` wrote
  `f"sha256_{hash(...) & 0xffffffff:08x}"` — eight hex characters of SipHash
  wearing a digest's name, in the one field whose job is to let someone check
  whether two records describe the same run.
* `torch.load(..., weights_only=False)` in `sae/loader.py` — a pickle load on a
  checkpoint path.
* `Critic.is_confident` accepted `decision in ("", "Accept")`, and `decision` was
  `""` whenever peer review was absent. Confidence 0.90 with no peer review
  whatsoever passed.

### A method that could never return, and an export that leaked — `9770763`

`config_agreement()` appended rows carrying only `{"field": field}` and then read
`r["agrees"]`, so it raised `KeyError` on every call. Architecture validation had
never run. The `declared` and `actual` locals it computed and discarded are what
made the cause hard to see.

`project_export.py` ran `db.query(ReportRecord).all()`, so exporting project A
put project B's reports into A's archive. Filtering was not available as a patch:
`ReportRecord` has no `project_id` column at all. Adding one needs a migration,
and `create_all` does not alter existing tables. So the export fails closed —
sessions only — and records the omission and its reason inside the archive's own
`metadata.json`. A test asserts the schema gap is still open, so whoever adds the
column is asked to restore the export properly rather than delete the test.

### Salted-hash identifiers — `a19d636`

Fixed the dispatcher, Scribe, Society and discovery engine. Fourteen sites remain
and are recorded explicitly in `KNOWN_REMAINING` in
`tests/pytest/test_no_hash_derived_identifiers.py`, with a ratchet so the list can
only shrink deliberately.

### The validator, and the protocol channel — `c7b5af2`

`ProjectValidator` had eight literal `True` values and returned `"ready"` for
every project, including one that does not exist. Its output means "you may
publish this", so being wrong in that direction is the worst place in the
codebase to have a stub. Now probe-driven, with `None` for "nobody checked" so
`all(checks.values())` still fails closed. **Behaviour change:** with no probes
it returns `"unverified"`, not `"ready"`.

Plugin `print()` went to stdout, which is the JSON protocol channel. It now goes
to stderr.

---

## Audit claims that were wrong

Recorded because they cost time and because an audit that is right 90% of the
time is still worth reading.

* **"`backend.science.integrity` does not export `sign`, `verify`,
  `SigningUnavailable`."** It does. The cause was **my own bug** in the export
  script that produced the snapshot the audit read: it filtered every file whose
  name began with an underscore, intending to drop scratch files, and therefore
  dropped 60 of the project's 62 `__init__.py` files. Fixed; the regenerated
  snapshot includes all 62.

* **"There are numerous direct `provenance="live"` paths"** — overstated. There
  are three production writers (`agents/scribe.py`,
  `distributed/evidence_aggregator.py` twice), and all three derive the label
  rather than asserting it. The real defect is different and is stated in the
  section below.

---

## Not done, and why

**The EvidenceBoundary still has no production callers.** `admit()` is called
from tests only. The fix made the boundary *correct* and *unforgeable*, but a
correct gate that nothing walks through is still not a gate. Wiring the three
real live-writers through it is the remaining half of this finding and was left
alone deliberately: it touches the Society publication path, the one flow with a
large existing test suite, and it could not be run.

**Plugin hook timeouts are still not enforced.** `RemotePluginProxy` stores
`_timeout` and never reads it; `_roundtrip()` blocks on `stdout.readline()`. A
plugin calling `time.sleep()` hangs the hook indefinitely — the CPU limit does
not help, because sleeping is not consuming CPU. This is a denial-of-service
hole and it is the single most important thing left.

**API authentication is absent.** Loopback binding and the CORS policy are
mitigations, not authentication.

**Manifest signing covers three component hashes, not the publication-critical
metadata.** `experiment_id`, `status`, `statistical_verdict` and
`reproducibility_score` are outside the signed root.

**`Required`-metric gates, the reproducibility snapshot, plugin artifact
integrity, and the legacy in-process `importlib` plugin loader** are all still
open.

**The architecture consolidation is not started.** Duplicate registries under
`backend/mech_platform/` and `backend/research_platform/` are the root of the
"fix applied in one, absent in the other" hazard. One duplicate was removed this
session — `research_platform`'s `provenance_service` now re-exports the
`mech_platform` original — but the pattern is untouched.

**Backlog #13 is unverified.** Its implementation passed 67 tests at the last
complete run. Five guards were then rewritten after the negative-control sweep
showed 6 of 10 fabrications escaped them, and the rewritten guards have not been
executed. The escapes all came from whole-file text greps; the replacements are
per-function assertions.

---

## A note on method

Two process notes, because both changed conclusions.

**A recursive search lied.** PowerShell's `backend\**\*.py` stops at two
directory levels and silently reported "no `torch.load` anywhere in backend",
which contradicted the audit. ripgrep found it three levels down at
`backend/interpretability/sae/loader.py`. The audit was right and the search was
wrong. Prefer the AST and ripgrep.

**Guards over prose get deleted rather than obeyed.** Six of ten negative
controls in this session escaped guards built on whole-file text searches, and
the consistent cause was that the guard could not see the code it was about.
Every new test here constructs the forgery and asks the system, rather than
grepping for a banned spelling.