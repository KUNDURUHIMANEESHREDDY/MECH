# Release gate — `539ac7c`

**Verdict: NOT DONE. Not release-ready.**

This is a gate report against one commit, not another audit. Everything below is
either a command that was run or a source location that was read. Where I could
not establish something, it says so rather than being left implied.

## Candidate

| | |
|---|---|
| Commit | `539ac7c` — P1-21: Society anchor loaded weights with the wrong op |
| Branch | `master`, tracking `origin/master` |
| Unpushed | 26 commits ahead of `origin/master` (`516561e`) |
| Working tree | 166 dirty entries from a concurrent session, uncommitted and out of scope |

The candidate is the local `HEAD`. It has **not been pushed**, so `origin/master`
(`516561e`) is the last released state.

## Gate results

| # | Gate | Result | Command / evidence |
|---|---|---|---|
| 1 | Syntax + undefined names (E9, F821) | **PASS** | `ruff check --isolated --no-cache --select E9,F821 backend tests scripts` → *All checks passed* (ruff 0.17.0, installed locally to match CI) |
| 2 | Full suite, candidate tree | **2 FAILED** | see below |
| 3 | Targeted audit suites (6) | **269 PASS** | `pytest test_export_secret_exclusion test_capture_fail_closed test_capture_model_identity test_head_ablation_ranking test_capture_screenshots test_standalone_launcher` |
| 4 | Tests actually collected | **1784 collected** | `--collect-only`; not silently absent |
| 5 | Skips are honest, not silent | **PASS** | 9 skips, each a named platform prerequisite — 6 symlink privilege, `os.fork` absent, module-cache isolation, one empty parameter set. No security or measurement test skips itself. |
| 6 | Secrets absent from history | **PASS** | No credential paths in the 26 commits; `backend/storage/.mech_api_token` gitignored and untracked; no literal credential assignments in any diff |
| 7 | Society workflow verified | **NOT DONE** | route blocked, see below |
| 8 | Earlier security findings reconciled | **PARTIAL** | one remains open, see below |
| 9 | Independent review | **NOT DONE** | not obtained |
| 10 | Released | **NOT DONE** | nothing pushed |

### Gate 2 — the two failures, and why they matter

`2 failed, 1772 passed, 9 skipped, 1 xfailed` (10m08s).

**That run was on the working tree, which carries the concurrent session's 166
uncommitted entries — not on a clean checkout of the candidate.** Only the two
failing tests were re-run against a pristine `539ac7c` tree (below), so the
full-suite figure is indicative and the failure analysis is confirmed. The
clean-checkout full run remains an open item in gate 3.

Both failures are **introduced by this candidate**, not inherited:

| Test | Cause |
|---|---|
| `test_health_probes::test_evidence_probe_checks_the_boundary_imports` | `624bd3f` deletes `backend/core/evidence_boundary.py`; `backend/api/health.py` still imports it → `No module named 'backend.core.evidence_boundary'` |
| `test_provenance_origination::test_every_module_stamping_live_is_classified` | two modules stamp `provenance='live'` unclassified; the test's own message forbids classifying them just to make it pass |

Established by running both in a clean worktree at `origin/master` — **2 passed** —
and again in a clean worktree at `539ac7c` — **2 failed**. Both worktrees were
removed and the temp branch deleted afterwards.

`test_provenance_origination`'s message is explicit: *"Do not add it just to make
the test pass."* Classifying those modules asserts that they *measure*, which is
the same judgement the deferred Society contract question needs. Answering it here
would prejudge that.

### Gate 7 — Society route

`load → loaded`, `reproduce → completed`, `inspect → ok`, `patch → ok`, then
`discover → unavailable`. The route ends in `phase="failed"` with:

```
Discovery blocked: stage status 'unavailable' is not complete.
```

Not a regression from `539ac7c` (the contract mismatch it fixed is gone), and not
fixed by it. `live_discovery.MIN_PROMPTS_FOR_VALIDATION = 10` while its `run()`
defaults to `n_prompts=4`, and `discover_and_orchestrate()` forwards no prompt
count, so `discovery_is_live()` cannot be satisfied at the default. Its docstring
states this is deliberate — the cost decision is left with the caller.

The capture script reports it honestly and exits 1. **The Society workflow is not
verified and must not be presented as such.**

### Gate 8 — earlier security findings, reconciled

Mapped against `docs/AUDIT_REMEDIATION.md` § *Not done, and why*.

| Earlier open finding | Status on `539ac7c` | Evidence |
|---|---|---|
| API authentication absent | **CLOSED** | `cfd6cc9` mandatory bearer auth + request-body ceiling; auth suites pass |
| Plugin hook timeout DoS — "the single most important thing left" | **CLOSED** | `_read_with_timeout()` in `backend/plugins/proxy.py:102-120` bounds `readline()` by `self._timeout` via a queue; `_roundtrip` uses it instead of a bare `readline()` |
| Plugin read confinement / Windows containment | **CLOSED** | `6cd22d5`, `dde24ae` |
| Attestation laundering / shape-only signature | **CLOSED** | `462a9a5`, `cb14eb7` |
| Manifest signing incomplete; Required-gates; reproducibility snapshot; plugin artifact integrity; legacy `importlib` loader; architecture consolidation; backlog #13 | **UNVERIFIED** | not examined in this bounded pass — do not read silence as closed |
| **EvidenceBoundary has no production callers** | **CLOSED, by removal not wiring** | `backend/core/evidence_boundary.py` was deleted in `624bd3f` — this gate report got this wrong, recording the module as still present with 0 call sites. It duplicated `load_run_record` *without* the quarantine checks, so a caller walking through it bypassed the stamp enforcement. All load paths now route through `evidence_graph.py`, which enforces `HISTORICAL_ORIGIN`. The audit's proposed remedy, wiring the live-writers through `admit()`, is moot: the duplicate is gone rather than bridged. |

The last one is the finding I'd flag hardest. A correct, unforgeable gate that no
production code walks through is still not a gate. The earlier record said wiring
it was left alone deliberately because it touches the Society publication path;
that decision still stands, and it still needs making.

## Blockers to release

1. **CI would go red on push.** Two tests pass at `516561e` and fail at `539ac7c`,
   both caused by commits in this candidate.
2. **Society workflow unverified** — blocked at the discovery gate.
3. **EvidenceBoundary has no production callers** — the gate is real, unused.
4. **No independent review** of the security-critical paths.
5. **Unpushed, unverified working tree** — 166 dirty entries from a concurrent
   session mean no working copy here is a clean checkout of the candidate.

## What was *not* done, on purpose

- No fix for either failing test. Fixing `test_provenance_origination` by
  classification would answer a question already deferred; fixing
  `test_health_probes` needs a decision on whether evidence-boundary logic still
  belongs somewhere.
- No push. A red CI is a release decision, not a side effect.
- No unrestricted audit. Items marked UNVERIFIED above are unexamined, not closed.

## What the next pass should do, in order

1. Decide `test_health_probes`: drop the `boundary` component from the probe and
   its test, or restore a shim. Requires a judgement on where evidence-boundary
   logic lives.
2. Decide `test_provenance_origination`: classify the two modules in
   `MEASUREMENT_LAYERS`, or route them through `backend.core.provenance`. Both
   assert something about what those modules measure — answer alongside the
   Society contract question, not before it.
3. Run the **full suite on a clean checkout of the resulting commit**, not the
   working tree.
4. Decide the EvidenceBoundary wiring — the one genuinely open security finding.
5. Only then push, and obtain independent review of the security-critical paths.

Steps 1 and 2 are the release-blocking pair; 3–5 are confirmation.

## Honest limits of this gate

A passing test proves its assertion holds, not that the boundary it samples is
correct. The 269 targeted results and the full-suite run establish that the six
remediated areas behave as their tests assert; they do not establish that the
security boundary as a whole is sound. The verdict above reflects that.
