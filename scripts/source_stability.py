"""Attest that the source tree held still for the duration of a test run.

Why this module exists
----------------------
A full-suite run here takes about nine minutes. During one such run, one test
failed::

    test_live_discovery_eligibility.py::test_discovery_id_is_stable_across_processes
    ImportError: cannot import name 'set_evidence_level'
      from 'backend.core.provenance' (backend/core/provenance.py)

Nothing was actually wrong. `set_evidence_level` is defined at
`backend/core/provenance.py:251`; the import succeeds in any clean process, and
the test passes three times out of three in isolation. But `provenance.py`'s
mtime was 06:35:07 -- inside that run's window -- with its `.pyc` recompiled at
06:35:25. The file was being written at the instant the test's subprocess
imported it, and a half-saved file fails `from x import y` with exactly that
error.

That test spawns a fresh interpreter, so it is the only one in the suite that
reads the source from disk instead of inheriting the parent's already-imported
modules. Every other test was insulated by imports that had already happened.
The damage was therefore not a broken product -- it was a run whose result could
not be trusted, reported in exactly the same shape as a real one: same `FAILED`
line, same exit status 1. The tree also carries ~187 uncommitted modifications
from another session, and the failure moved between runs, because which red you
got depended on where in someone else's save cycle the run happened to land.

The policy in one paragraph
---------------------------
A test run that straddles a source-file write is not a green run. Snapshot every
watched source file at session start, snapshot again at session end, and if the
two differ the session reports failure with the exact paths and both states --
never success, and never a bare `FAILED` line that looks like a real regression.

Why a digest *and* a timestamp
------------------------------
The first version recorded `st_mtime_ns` alone, on the reasoning that a hash
would not notice an editor rewriting a file with identical bytes. Measuring that
assumption on this machine killed it. Five consecutive writes to one file
produced **two** distinct mtimes::

    write 0  mtime_ns=1791555323115779500
    write 1  mtime_ns=1791555323115779500
    write 2  mtime_ns=1791555323117793100
    write 3  mtime_ns=1791555323117793100
    write 4  mtime_ns=1791555323117793100

Windows does not update the last-write-time on every write; it advances on a
coarse timer, so two saves close together can share a timestamp. Neither signal
is sufficient alone, and each catches what the other misses:

* the **digest** catches a genuine content change that a coarse timestamp hid;
* the **timestamp** catches the *target* failure -- a file half-written during
  the run and restored by the end, whose content matches but whose bytes were in
  flux while some subprocess was importing them.

A file is therefore declared changed when the mtime, the size, or the digest
differs. The cost is one read of each watched file twice per suite, which is
negligible against a nine-minute run.

Scope
------
Only Python sources are watched, because that is the demonstrated failure mode:
a partially written `.py` breaks the import that reads it, and a `.py` is also
the thing a test imports. A watched file is one under a `WATCH_ROOTS` entry with
a `WATCH_SUFFIXES` extension. `__pycache__`, caches, build output, virtualenvs
and `.git` are excluded, because they legitimately change during a run -- the
`.pyc` rewrite is a *consequence* of the edit this is meant to catch, and
including it would report the symptom rather than the cause.

The exclusion names match the structure *inside* the watched tree, never the
absolute path. This is not cosmetic: `pytest.ini` sends `tmp_path` into
`.pytest-tmp`, a name on the list, so matching absolute paths excluded every
tmp-based tree wholesale -- including this module's own unit tests, which then
watched nothing while reporting a clean run.

Escape hatch
------------
`MECH_SKIP_SOURCE_STABILITY_CHECK=1` disables the check for one session. That is
deliberate: a guard that blocks all testing whenever anyone is editing the tree
would be turned off permanently by its first user. The default is fail-closed;
opting out is an explicit, greppable act.

Nothing here imports `backend.*`. It is pure `os`/`pathlib`, because it has to
run before the test session and must not perturb what it is measuring.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

#: Directories, relative to the repo root, whose sources are watched.
WATCH_ROOTS: Tuple[str, ...] = ("backend", "tests", "scripts")

#: Suffixes that count as source. See the module docstring's Scope section.
WATCH_SUFFIXES = frozenset({".py", ".pyi"})

#: Directory names never descended into. Each of these changes during a normal
#: run for reasons unrelated to an edit, so watching them would produce exactly
#: the false positive this guard exists to prevent.
EXCLUDED_DIRS = frozenset({
    "__pycache__", ".git", ".hg", ".svn", ".mypy_cache", ".ruff_cache",
    ".pytest_cache", ".pytest-tmp", "node_modules", ".venv", "venv", "env",
    "dist", "build", ".tox", ".idea", ".vscode", "htmlcov", ".next",
})

#: Environment variable that disables the guard for one session.
BYPASS_ENV = "MECH_SKIP_SOURCE_STABILITY_CHECK"


@dataclass(frozen=True)
class FileStamp:
    """What we need to know about one file at one instant."""
    mtime_ns: int
    size: int
    digest: str

    def render(self) -> str:
        import datetime as _dt

        when = _dt.datetime.fromtimestamp(self.mtime_ns / 1_000_000_000)
        return "%s  %d B  sha256:%s" % (
            when.strftime("%Y-%m-%d %H:%M:%S"), self.size, self.digest[:12])


@dataclass(frozen=True)
class Change:
    """One watched file whose state at the end differed from the start."""
    path: str
    before: Optional[FileStamp]
    after: Optional[FileStamp]

    @property
    def kind(self) -> str:
        if self.before is None:
            return "added"
        if self.after is None:
            return "deleted"
        return "modified"

    def render(self) -> str:
        if self.kind == "added":
            return "  %s\n      added at %s" % (self.path, self.after.render())
        if self.kind == "deleted":
            return "  %s\n      removed (was %s)" % (self.path, self.before.render())
        return ("  %s\n      at session start: %s\n      at session end:   %s"
                % (self.path, self.before.render(), self.after.render()))


Snapshot = Dict[str, FileStamp]


def find_repo_root(start: Optional[Path] = None) -> Path:
    """The nearest ancestor that is this repository's root.

    A guard that silently watches nothing is worse than one that cannot find
    the root, so this raises instead of returning a plausible wrong directory.
    """
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "backend").is_dir() and (candidate / "pytest.ini").is_file():
            return candidate
    raise NotADirectoryError(
        "could not locate the repository root from %r; expected a directory "
        "containing both backend/ and pytest.ini" % str(here))


def _is_watched(filename: str, rel_dir: Path) -> bool:
    """Whether one file inside the tree being watched counts as source.

    `rel_dir` is the directory **relative to the root being watched**, never the
    absolute path. This distinction is not cosmetic: a tree that happens to live
    underneath a directory named `.pytest-tmp` -- which is where `pytest.ini`
    sends `tmp_path`, and therefore where the unit tests for this module run --
    would otherwise be excluded wholesale, and the guard would silently watch
    nothing. Exclusion describes the *structure inside* the watched tree, not
    where that tree is parked on disk.
    """
    if Path(filename).suffix not in WATCH_SUFFIXES:
        return False
    return not (set(rel_dir.parts) & EXCLUDED_DIRS)


def _digest(path: Path) -> str:
    """sha256 of the file's bytes, or "" if it could not be read.

    Read in chunks from a single open handle: a file being written can change
    length between two reads, and a caller that cannot read a file at all must
    record that instead of inventing a stable value.
    """
    import hashlib

    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


def snapshot(root: Optional[Path] = None,
             roots: Optional[Iterable[str]] = None) -> Snapshot:
    """`{path: FileStamp}` for every watched source file under `root`.

    Paths are relative to the repo root and forward-slashed, so two snapshots
    taken by different callers on different platforms are directly comparable.
    """
    base = root or find_repo_root()
    watched_roots = tuple(roots) if roots is not None else WATCH_ROOTS
    out: Snapshot = {}
    for name in watched_roots:
        top = base / name
        if not top.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(top):
            # Pruned in place, so the descent never enters an excluded
            # directory. Visiting `__pycache__` and skipping its files would
            # still pay the cost of walking it.
            dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
            rel_dir = Path(dirpath).relative_to(base)
            for filename in filenames:
                if not _is_watched(filename, rel_dir):
                    continue
                full = Path(dirpath) / filename
                try:
                    st = full.stat()
                except OSError:
                    # Present in the listing, gone by the stat: that is a change,
                    # and the diff catches it via the omission. Never crash the
                    # session start because one file was being moved.
                    continue
                out[full.relative_to(base).as_posix()] = FileStamp(
                    st.st_mtime_ns, st.st_size, _digest(full))
    # Sorted so a snapshot dict renders stably and two snapshots taken from
    # identical trees compare equal regardless of walk order.
    return dict(sorted(out.items()))


def diff(before: Snapshot, after: Snapshot) -> List[Change]:
    """Every watched file whose state differs between the two snapshots.

    Ordered by path so the report is diffable across runs.
    """
    changes: List[Change] = []
    for path in sorted(set(before) | set(after)):
        was, now = before.get(path), after.get(path)
        if was is None and now is None:
            continue
        if was is not None and now is not None and _unchanged(was, now):
            continue
        changes.append(Change(path, was, now))
    return changes


@dataclass
class Report:
    """The outcome of one attested span of time."""
    started: Snapshot
    ended: Snapshot
    changes: List[Change]
    stable: bool
    bypassed: bool


def _unchanged(was: FileStamp, now: FileStamp) -> bool:
    """Whether one file is the same at both ends of the span.

    Deliberately three-way. The digest catches a content change that a coarse
    timestamp hid; the timestamp catches the target failure -- a file half-written
    mid-run and restored by the end, whose *content* matches but whose bytes were
    in flux while some subprocess was importing them. Declaring either signal
    sufficient silently misses one of the two.
    """
    return (was.mtime_ns == now.mtime_ns
            and was.size == now.size
            and was.digest == now.digest)


def render_report(report: Report) -> str:
    """What a human reads when asked whether this run can be trusted.

    Written to say what to *do*, not merely what happened: a run blocked here is
    not a test failure, and a message that reads like one more regression in the
    tail would send the reader chasing a defect that does not exist.
    """
    if report.bypassed:
        return ("source-stability attestation skipped: %s is set, so this run "
                "makes no claim that the source tree held still."
                % BYPASS_ENV)
    if report.stable:
        return ("source-stability attestation passed: %d watched source file(s) "
                "identical at session start and end."
                % len(report.started))

    total = len(set(report.started) | set(report.ended))
    lines = [
        "",
        "*" * 78,
        "RUN NOT TRUSTWORTHY: the source tree changed while the tests ran.",
        "*" * 78,
        "",
        "This is NOT a test failure. A watched source file was created or",
        "rewritten between session start and session end, so any result above --",
        "pass or fail -- may describe a half-written file rather than the",
        "software.",
        "",
        "The usual cause is another session or an editor's autosave writing to",
        "the same tree. A .pyc rewritten mid-run is a symptom of that edit, not",
        "the edit itself, which is why caches are excluded from the watch.",
        "",
        "What to do: stash or commit the concurrent work, then re-run. Failures",
        "that survive on a still tree are real; the ones that vanish were this.",
        "",
        "To run regardless: set %s=1." % BYPASS_ENV,
        "",
        "%d of %d watched source file(s) changed:" % (
            len(report.changes), total),
    ]
    for change in report.changes:
        lines.append(change.render())
    lines.append("*" * 78)
    lines.append("")
    return "\n".join(lines)


class SessionGuard:
    """Snapshot at session start, attest at session end.

    Exists to be driven by the pytest hooks in `tests/pytest/conftest.py`, but
    deliberately imports no pytest, so it stays usable from other harnesses and
    unit-testable without one.
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        self._root = root or find_repo_root()
        self._started: Optional[Snapshot] = None

    @property
    def root(self) -> Path:
        return self._root

    @property
    def bypassed(self) -> bool:
        """Whether the opt-out is set *as an enabling value*.

        Not merely truthy: `0`, `false`, `no` and friends are what an operator
        types when trying to RE-ENABLE the check, so treating them as an opt-out
        disables the guard at the moment of its most likely restoration. Any
        value that is not an explicit enable is a refusal.
        """
        value = os.environ.get(BYPASS_ENV, "").strip().lower()
        return value in ("1", "true", "yes", "y")

    def start(self) -> Snapshot:
        self._started = snapshot(self._root)
        return self._started

    def finish(self) -> Report:
        started = self._started if self._started is not None else {}
        ended = snapshot(self._root)
        changes = diff(started, ended)
        return Report(started=started, ended=ended, changes=changes,
                     stable=not changes, bypassed=self.bypassed)

    def verify(self) -> Report:
        """Start if needed, then attest. Useful when no session wraps the call."""
        if self._started is None:
            self.start()
        return self.finish()


def recent_modifications(seconds: int = 120,
                         root: Optional[Path] = None) -> List[Tuple[str, FileStamp]]:
    """Watched source files modified within the last `seconds`, oldest first.

    An advisory, never a gate: an edit that landed *before* a run began cannot
    corrupt it. It is worth reporting anyway, because the dangerous case -- a
    peer saving every few minutes -- is indistinguishable from an ordinary
    recent edit by any means other than saying so.
    """
    import time as _time

    base = root or find_repo_root()
    cutoff = _time.time_ns() - int(seconds) * 1_000_000_000
    return sorted((p, s) for p, s in snapshot(base).items()
                  if s.mtime_ns >= cutoff)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI: `--json` dumps a snapshot; `--since N` adds the advisory; otherwise
    attest the current (effectively zero-length) span and print the verdict."""
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(
        description="Snapshot or attest the repository's watched source files.")
    parser.add_argument("--json", action="store_true",
                        help="emit the snapshot as JSON and exit")
    parser.add_argument("--since", type=int, default=0, metavar="SECONDS",
                        help="also list files modified within the last N seconds")
    args = parser.parse_args(argv)

    base = find_repo_root()
    snap = snapshot(base)

    if args.json:
        print(_json.dumps(
            {p: {"mtime_ns": s.mtime_ns, "size": s.size, "sha256": s.digest}
             for p, s in snap.items()},
            indent=2, sort_keys=True))
        return 0

    print("watched sources: %d under %s" % (len(snap), ", ".join(WATCH_ROOTS)))

    if args.since:
        fresh = recent_modifications(args.since, base)
        print("\nmodified in the last %d second(s): %d" % (args.since, len(fresh)))
        for path, stamp in fresh:
            print("  %-70s %s" % (path, stamp.render()))
        if fresh:
            print("\n(note: a recent edit is only a risk if the tree changes "
                  "again while\n the tests are running. This listing is "
                  "advisory.)")

    report = SessionGuard(base).verify()
    print()
    print(render_report(report))
    return 0 if (report.stable or report.bypassed) else 1


if __name__ == "__main__":
    raise SystemExit(main())