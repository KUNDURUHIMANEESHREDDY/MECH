# The source-stability guard

**Status:** fixed in this session.
**Found by:** the flaky-test investigation, after chasing five intermittent
failures that each passed in isolation.

## The symptom

Full-suite runs produced a *different* set of failures nearly every time. Over
five runs, the same nine-minute suite reported:

| run | failures beyond the two known pre-existing ones |
|---|---|
| 1 | `runtime_engine_states` ×7 |
| 2 | `evidence_quarantine` |
| 3 | `challenger_m1_adversarial` |
| 4 | `route_authorization_matrix` (401) |
| 5 | `live_discovery_eligibility` (ImportError) |

Every one of them passed in isolation. All the obvious explanations were ruled
out with evidence: no test-ordering plugin is installed; `MECH_API_TOKEN` and
the auth cache were instrumented across a full run and **never mutated**;
`get_bearer_token()` re-reads the environment on every call with no
memoisation; importing `backend.main` never binds a port (`uvicorn.run` is under
`__main__`), so an ambient server on 8000 cannot affect `TestClient`; the
`/openapi.json` path is genuinely protected with no rate limiter anywhere; the
WAL test passes idle *and* under six-way CPU load.

## Root cause

The repository was being written to **while the suite ran**.

Runs 1–4 were never diagnosed, but the mechanism is the same and run 5 names it
directly. `test_live_discovery_eligibility.py` failed with:

```
ImportError: cannot import name 'set_evidence_level'
  from 'backend.core.provenance' (backend/core/provenance.py)
```

`set_evidence_level` is defined at `backend/core/provenance.py:251`. The import
succeeds in any clean process and the test passes 3/3 in isolation. But
`provenance.py`'s mtime was **06:35:07** — inside that run's window — with its
`.pyc` recompiled at **06:35:25**. The file was being written at the instant the
test's subprocess imported it, and a half-saved file fails `from x import y`
with exactly that error.

That test spawns a fresh interpreter, so it is the only one in the suite that
reads the source from disk instead of inheriting modules its parent already
imported. Every other test was insulated by imports that had already happened.
The tree also carries ~187 uncommitted modifications from a peer session.

So the damage was never a broken product. It was a run whose result could not be
trusted, reported in exactly the shape of a real regression: same `FAILED` line,
same exit status 1. Which red you got depended on where in someone else's save
cycle the run happened to land.

## The fix

`scripts/source_stability.py` + two hooks in `tests/pytest/conftest.py`.

**Policy:** a test run that straddles a source-file write is not a green run.

- At session start, snapshot every watched source file.
- At session end, snapshot again.
- If the two differ, print a report stating plainly that this is **not** a test
  failure, name the files with both states, and force `session.exitstatus = 1`.
- Never emit a bare `FAILED` line — that is the failure this prevents.

**Watched:** `.py` / `.pyi` under `backend/`, `tests/`, `scripts/` — 572 files
today. `__pycache__`, caches, build output, virtualenvs and `.git` are excluded,
because a `.pyc` rewritten mid-run is a *consequence* of the edit being caught,
and reporting the symptom would flag every run.

**Escape hatch:** `MECH_SKIP_SOURCE_STABILITY_CHECK=1` for one session.
Fail-closed by default, but a guard that blocks all testing whenever anyone is
editing would be disabled permanently by its first user.

**Cost:** 1.66s to hash all 572 files twice, against a ~10-minute suite.

## Two defects the guard's own tests caught

**Exclusion matched absolute paths.** The exclusion list names `__pycache__`,
`build`, `dist`, `node_modules` and friends. Matching those against the
*absolute* path excluded any tree that merely lives underneath such a directory
— and `pytest.ini` sends `tmp_path` into `.pytest-tmp`, which is on that list.
Every unit test for the guard therefore watched nothing, and the guard reported
a clean run for a tree it had never looked at. Now matched against the path
relative to the watched root.

**A timestamp alone is not a change signal.** The first version recorded
`st_mtime_ns` only, reasoning that a hash would miss a same-length rewrite.
Measuring that on this machine killed it — five writes to one file produced
**two** distinct mtimes, because Windows advances the last-write-time on a
coarse timer:

```
write 0  mtime_ns=1791555323115779500
write 1  mtime_ns=1791555323115779500
write 2  mtime_ns=1791555323117793100
write 3  mtime_ns=1791555323117793100
write 4  mtime_ns=1791555323117793100
```

Neither signal is sufficient and each catches what the other misses:

- the **digest** catches a content change hidden behind a coarse timestamp;
- the **timestamp** catches the *target* failure — a file half-written mid-run
  and restored by the end, whose content matches but whose bytes were in flux
  while a subprocess was importing them.

A file is changed when the mtime, the size, or the digest differs.

## Verification

`tests/pytest/test_source_stability.py`, 38 tests, all passing. Includes
end-to-end coverage through a real `pytest` subprocess in a synthetic tree wired
to the real guard: exit 0 over a still tree, exit nonzero when a watched source
is rewritten mid-session, `FAILED` absent when the run is blocked, and the
bypass letting it through.

Mutation-tested; each protection fails with a precise message and every file was
restored byte-exact and re-verified:

| mutation | result |
|---|---|
| drop the digest comparison | 2 failed |
| drop the mtime comparison | 2 failed |
| drop the size comparison | 1 failed |
| never flag anything | 14 failed |
| exclusion against the absolute path | 30 failed |
| conftest never sets a nonzero exit | 1 failed |

Full suite with the guard installed: **2 failed, 1757 passed** — both failures
pre-existing. The guard also proved itself in anger, firing on
`tests/pytest/conftest.py` because the author edited it mid-run, which is
exactly the case it exists to catch.

## Also observed, not fixed

- Two concurrent `pytest` processes wipe each other's `--basetemp=.pytest-tmp`
  (`FileExistsError` / `WinError 145`). Reachable by accident; the guard does not
  address it.
- `tests/pytest/test_live_discovery_eligibility.py` writes its helper script to
  the *relative* path `_discovery_id_stability_probe.py`, so it lands in the repo
  root only when pytest happens to be invoked from there, and depends on cwd
  rather than on `tmp_path` like the rest of the suite. It does clean up after
  itself. Not a flake source today, but it is the reason that test is
  cwd-sensitive where its peers are not.
