"""The source-stability guard: a run that straddles an edit is not a green run.

The defect
----------
A nine-minute suite run failed one test with::

    ImportError: cannot import name 'set_evidence_level'
      from 'backend.core.provenance'

`set_evidence_level` exists at `backend/core/provenance.py:251`. The import
succeeds in a clean process and the test passes in isolation, because the file
was being *written* at the instant a subprocess imported it. The suite reported
that in exactly the shape of a real regression -- `FAILED` line, exit status 1 --
so a reader could not tell a broken product from a broken run.

That test spawns a fresh interpreter, which is why it alone saw the dust: it
reads the source from disk, while every other test inherits modules its parent
had already imported. The red therefore moved between runs, tracking whichever
moment in someone else's save cycle the run happened to straddle.

What is asserted here
---------------------
Unit: the snapshot, the diff, the report, the bypass. End-to-end: a real
`pytest` process, in a synthetic tree wired to the *real* guard, over a passing
test file -- exit 0 while the tree holds still, exit nonzero the instant a
watched source is rewritten mid-run. The second is the whole deliverable, so it
is driven through an actual subprocess rather than by calling the hooks.

Every test that asserts "diff reports this change" is paired with a still-tree
assertion, so `diff` cannot pass merely by reporting everything.
"""

from __future__ import annotations

import os
import shutil
from unittest import mock
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
CONFTEST_SOURCE = (REPO_ROOT / "tests" / "pytest" / "conftest.py").read_text(
    encoding="utf-8")

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import source_stability as ss  # noqa: E402


# ── fixtures and helpers ─────────────────────────────────────────────────

def write(path: Path, text: str = "") -> None:
    path.write_text(text, encoding="utf-8")


def retime(path: Path) -> None:
    """Force a new mtime without necessarily changing the length.

    Editors save in place, so the interesting case is a file whose mtime moves
    and whose size does not.
    """
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns + 1_000_000_000,
                       st.st_mtime_ns + 1_000_000_000))


@pytest.fixture
def repo(tmp_path):
    """A minimal tree that looks enough like the repo for the guard to run.

    The guard keys off `backend/` plus `pytest.ini`, so a synthetic tree must
    have both, or `find_repo_root` walks straight past it and the test exercises
    nothing but that walk.
    """
    (tmp_path / "backend").mkdir()
    write(tmp_path / "backend" / "pkg.py", "VALUE = 1\n")
    write(tmp_path / "pytest.ini", "[pytest]\naddopts =\n")
    return tmp_path


# ── finding the root, walking the tree ───────────────────────────────────

def test_the_repo_root_is_found_from_a_subdirectory():
    assert ss.find_repo_root(REPO_ROOT / "backend" / "agents") == REPO_ROOT


def test_the_repo_root_is_found_from_inside_a_test_file():
    assert ss.find_repo_root(Path(__file__).parent) == REPO_ROOT


def test_a_tree_without_the_marker_is_rejected_rather_than_guessed(tmp_path):
    """A guard that silently watches nothing is worse than one that refuses.

    The tree must sit *outside* the repository, because `find_repo_root` walks
    upward and `tmp_path` here lives under `.pytest-tmp` -- an ancestor of which
    is the repo itself.
    """
    outside = Path(tempfile.mkdtemp(prefix="mech-root-probe-"))
    try:
        assert not (outside / "backend").exists()
        with pytest.raises(NotADirectoryError):
            ss.find_repo_root(outside)
    finally:
        shutil.rmtree(outside, ignore_errors=True)


def test_the_snapshot_covers_the_real_repo():
    snap = ss.snapshot(REPO_ROOT)
    assert len(snap) > 100, f"only {len(snap)} watched sources; the walk is incomplete"
    assert "backend/core/auth.py" in snap


def test_the_snapshot_never_contains_a_pycache_or_offroot_entry():
    """The `.pyc` rewrite is a *consequence* of the edit being caught.

    Watching `__pycache__` would report the symptom and nothing else, so a run
    would flag even for an edit the guard was meant to attribute to elsewhere.
    """
    snap = ss.snapshot(REPO_ROOT)
    assert not [p for p in snap if "__pycache__" in p]
    assert not [p for p in snap
                if not p.startswith(("backend/", "tests/", "scripts/"))]


def test_the_snapshot_paths_are_relative_and_forward_slashed(repo):
    """Absolute or backslashed paths make two snapshots of one tree unequal."""
    assert set(ss.snapshot(repo)) == {"backend/pkg.py"}


def test_the_snapshot_is_ordered(repo):
    for name in ("c.py", "a.py", "b.py"):
        write(repo / "backend" / name, "#\n")
    assert list(ss.snapshot(repo)) == sorted(ss.snapshot(repo))


def test_an_absent_watch_root_is_skipped_rather_than_raising():
    """`scripts/` need not exist in a tree the guard is pointed at."""
    assert ss.snapshot(REPO_ROOT, roots=("backend",)) != {}


def test_a_tree_parked_under_an_excluded_name_is_still_watched(tmp_path):
    """Regression: a bug the tests above caught on their first run.

    The exclusion list names `__pycache__`, `build`, `dist`, `node_modules` and
    friends. Matching them against the **absolute** path excluded a tree that
    merely happens to live underneath such a directory -- and `pytest.ini` sends
    `tmp_path` into `.pytest-tmp`, which is on that list, so this module's own
    unit tests lived in the blind spot. Every unit test here would then have
    watched nothing and the guard would have looked healthy while inert.

    The rule describes the structure *inside* the watched tree, not where the
    tree sits on disk. Pytest's `tmp_path` lands somewhere different under
    `--basetemp`, so this must hold wherever the guard is pointed.
    """
    # Deliberately nested under a directory whose name is on the exclusion list,
    # exactly as .pytest-tmp is.
    nested = tmp_path / "__pycache__" / "checkout"
    (nested / "backend").mkdir(parents=True)
    write(nested / "backend" / "pkg.py", "VALUE = 1\n")
    write(nested / "pytest.ini")

    snap = ss.snapshot(nested)
    assert set(snap) == {"backend/pkg.py"}, (
        "a tree parked under an excluded directory name watched nothing; the "
        "guard would report a clean run for a tree it never looked at")


def test_an_excluded_directory_inside_the_tree_is_still_excluded(repo):
    """The other half: pruning still has to work inside the watched tree.

    Without this, the fix above could be right by accident -- e.g. by excluding
    nothing at all.
    """
    (repo / "backend" / "__pycache__").mkdir()
    write(repo / "backend" / "__pycache__" / "pkg.cpython-311.pyc")
    assert set(ss.snapshot(repo)) == {"backend/pkg.py"}


# ── the diff ────────────────────────────────────────────────────────────

def test_two_snapshots_of_a_still_tree_differ_on_nothing(repo):
    assert ss.diff(ss.snapshot(repo), ss.snapshot(repo)) == []


def test_a_new_mtime_is_a_change(repo):
    before = ss.snapshot(repo)
    retime(repo / "backend" / "pkg.py")
    changes = ss.diff(before, ss.snapshot(repo))
    assert [c.path for c in changes] == ["backend/pkg.py"]
    assert changes[0].kind == "modified"


def test_a_same_length_rewrite_is_still_a_change(repo):
    """The editor-saves-in-place case: fresh mtime, identical size.

    This is the shape of a half-written file, and size alone would miss it.
    """
    target = repo / "backend" / "pkg.py"
    write(target, "VALUE = 2\n")
    before = ss.snapshot(repo)
    write(target, "VALUE = 3\n")                      # same length, new mtime
    after = ss.snapshot(repo)

    assert before["backend/pkg.py"].size == after["backend/pkg.py"].size, (
        "the fixture no longer reproduces a same-length rewrite")
    assert [c.path for c in ss.diff(before, after)] == ["backend/pkg.py"]


def test_an_added_file_is_a_change(repo):
    before = ss.snapshot(repo)
    write(repo / "backend" / "new.py", "#\n")
    changes = ss.diff(before, ss.snapshot(repo))
    assert [c.path for c in changes] == ["backend/new.py"]
    assert changes[0].kind == "added"
    assert changes[0].before is None


def test_a_removed_file_is_a_change(repo):
    write(repo / "backend" / "gone.py", "#\n")
    before = ss.snapshot(repo)
    (repo / "backend" / "gone.py").unlink()
    changes = ss.diff(before, ss.snapshot(repo))
    assert [c.path for c in changes] == ["backend/gone.py"]
    assert changes[0].kind == "deleted"
    assert changes[0].after is None


def test_a_change_is_reported_once_with_both_stamps(repo):
    before = ss.snapshot(repo)
    write(repo / "backend" / "pkg.py", "VALUE = 40\n")
    after = ss.snapshot(repo)
    changes = ss.diff(before, after)
    assert len(changes) == 1
    rendered = changes[0].render()
    assert "at session start" in rendered and "at session end" in rendered, (
        "a report with one timestamp cannot tell a reader what moved when")


# ── the two change signals ──────────────────────────────────────────────

def test_a_content_change_behind_a_coarse_timestamp_is_still_a_change():
    """The bug that killed the timestamp-only version.

    Measured on this machine, five writes to one file produced two distinct
    mtimes -- Windows advances the last-write-time on a coarse timer, so two
    saves close enough together share a timestamp::

        write 0  mtime_ns=1791555323115779500
        write 1  mtime_ns=1791555323115779500
        write 2  mtime_ns=1791555323117793100
        write 3  mtime_ns=1791555323117793100
        write 4  mtime_ns=1791555323117793100

    An unchanged mtime with genuinely different bytes is therefore reachable
    without any contrivance, and a guard that trusted the timestamp alone would
    report a clean run over a changed tree.
    """
    before_stamp = ss.FileStamp(1791555323117793100, 11, "a" * 64)
    after_stamp = ss.FileStamp(1791555323117793100, 11, "b" * 64)
    changes = ss.diff({"backend/pkg.py": before_stamp},
                      {"backend/pkg.py": after_stamp})
    assert [c.path for c in changes] == ["backend/pkg.py"]


def test_a_timestamp_change_with_identical_content_is_still_a_change():
    """The target failure needs the timestamp, not the digest.

    A file half-written during the run and *restored* by the end has matching
    bytes, so a digest-only guard reports it unchanged -- while a subprocess in
    the middle of that window was importing a truncated module. This is the
    ImportError that started the whole investigation.
    """
    before = ss.FileStamp(1000, 11, "c" * 64)
    after = ss.FileStamp(2000, 11, "c" * 64)
    assert ss.diff({"backend/pkg.py": before},
                   {"backend/pkg.py": after})


def test_identical_stamps_are_not_a_change():
    stamp = ss.FileStamp(1000, 11, "d" * 64)
    assert ss.diff({"backend/pkg.py": stamp}, {"backend/pkg.py": stamp}) == []


def test_a_size_change_alone_is_a_change():
    """Neither signal subsumes the other; all three are compared."""
    before = ss.FileStamp(1000, 11, "e" * 64)
    after = ss.FileStamp(1000, 12, "e" * 64)
    assert ss.diff({"backend/pkg.py": before},
                   {"backend/pkg.py": after})


def test_the_digest_is_a_real_sha256(repo):
    """A digest that recurs for every file would make the signal decorative."""
    write(repo / "backend" / "a.py", "import os\n")
    write(repo / "backend" / "b.py", "import sys\n")
    snap = ss.snapshot(repo)

    digests = {snap[p].digest for p in ("backend/a.py", "backend/b.py")}
    assert digests, "no digests were recorded"
    assert len(digests) == 2, "different files produced the same digest"
    for d in digests:
        assert len(d) == 64 and all(c in "0123456789abcdef" for c in d)


def test_the_snapshot_still_reports_a_digest_for_the_real_repo():
    snap = ss.snapshot(REPO_ROOT)
    assert all(len(s.digest) == 64 for s in snap.values()), (
        "some watched file recorded no digest; it would be invisible to the "
        "signal that catches a coarse timestamp")


# ── the report ──────────────────────────────────────────────────────────

def _unstable_report(repo) -> ss.Report:
    before = ss.snapshot(repo)
    write(repo / "backend" / "pkg.py", "VALUE = 99\n")
    after = ss.snapshot(repo)
    return ss.Report(before, after, ss.diff(before, after), stable=False,
                     bypassed=False)


def test_the_report_says_the_run_is_not_trustworthy(repo):
    text = ss.render_report(_unstable_report(repo))
    assert "RUN NOT TRUSTWORTHY" in text
    assert "NOT a test failure" in text, (
        "a reader must not have to work out whether this is a real regression")
    assert "backend/pkg.py" in text
    assert ss.BYPASS_ENV + "=1" in text, "the message has to say how to proceed"


def test_a_stable_report_says_so(repo):
    before = ss.snapshot(repo)
    report = ss.Report(before, ss.snapshot(repo), [], stable=True, bypassed=False)
    assert "attestation passed" in ss.render_report(report)


def test_a_bypassed_report_makes_no_claim(repo):
    snap = ss.snapshot(repo)
    text = ss.render_report(ss.Report(snap, snap, [], stable=True, bypassed=True))
    assert "skipped" in text
    assert "no claim" in text, "an opt-out must not read as a pass"


# ── the escape hatch ────────────────────────────────────────────────────

def test_the_bypass_is_honoured(repo, monkeypatch):
    monkeypatch.setenv(ss.BYPASS_ENV, "1")
    guard = ss.SessionGuard(repo)
    guard.start()
    write(repo / "backend" / "pkg.py", "VALUE = 7\n")
    report = guard.finish()

    assert report.changes, "the guard still saw the change; bypass must not hide it"
    assert report.bypassed is True
    assert report.stable is False


def test_a_blank_bypass_does_not_bypass(repo, monkeypatch):
    """An exported-but-empty variable is not an opt-out."""
    monkeypatch.setenv(ss.BYPASS_ENV, "   ")
    assert ss.SessionGuard(repo).bypassed is False


# ── the session guard itself ────────────────────────────────────────────

def test_a_guard_over_a_still_span_reports_stable(repo):
    guard = ss.SessionGuard(repo)
    assert guard.start()
    assert guard.finish().stable is True


def test_a_guard_over_an_edited_span_reports_the_change(repo):
    guard = ss.SessionGuard(repo)
    guard.start()
    write(repo / "backend" / "pkg.py", "VALUE = 100\n")
    report = guard.finish()
    assert report.stable is False
    assert [c.path for c in report.changes] == ["backend/pkg.py"]


def test_verify_starts_if_no_one_did(repo):
    assert ss.SessionGuard(repo).verify().stable is True


def test_guards_do_not_share_state_through_the_root(repo):
    a, b = ss.SessionGuard(repo), ss.SessionGuard(repo)
    a.start()
    b.start()
    write(repo / "backend" / "pkg.py", "VALUE = 5\n")
    assert a.finish().stable is False
    assert b.finish().stable is False


# ── the advisory ────────────────────────────────────────────────────────

def test_recent_modifications_reports_a_just_written_file(repo):
    write(repo / "backend" / "fresh.py", "#\n")
    paths = [p for p, _ in ss.recent_modifications(600, repo)]
    assert "backend/fresh.py" in paths


def test_recent_modifications_excludes_an_aged_file(tmp_path):
    """The advisory must be about *now*, not about anything ever edited.

    Its own tree rather than the `repo` fixture: that fixture writes a file
    moments earlier, which is recent under any window and would make this pass
    for the wrong reason.
    """
    (tmp_path / "backend").mkdir()
    only = tmp_path / "backend" / "old.py"
    write(only, "#\n")
    long_ago = int((time.time() - 3600) * 1_000_000_000)
    os.utime(only, ns=(long_ago, long_ago))

    assert ss.recent_modifications(600, tmp_path) == []
    assert [p for p, _ in ss.recent_modifications(3600 * 2, tmp_path)] == ["backend/old.py"]


# ── end to end: a real pytest process ───────────────────────────────────

#: Conftest copied verbatim, so this exercises the guard as installed. The only
#: substitution is `scripts/`, supplied through PYTHONPATH: a synthetic tree has
#: no reason to carry a second copy of the guard it is meant to be testing.
def _build_tree(tmp_path: Path, rewrite_during_run: bool) -> Path:
    (tmp_path / "backend").mkdir()
    write(tmp_path / "backend" / "pkg.py", "VALUE = 1\n")
    write(tmp_path / "pytest.ini", "[pytest]\naddopts =\n")
    (tmp_path / "tests").mkdir()
    write(tmp_path / "tests" / "conftest.py", CONFTEST_SOURCE)
    body = "from pathlib import Path\n\n\ndef test_one():\n    pass\n"
    if rewrite_during_run:
        body += (
            "\n\ndef test_two_rewrites_a_watched_source():\n"
            "    pkg = Path(__file__).resolve().parents[1] / 'backend' / 'pkg.py'\n"
            "    pkg.write_text('VALUE = 2\\n', encoding='utf-8')\n")
    write(tmp_path / "tests" / "test_ok.py", body)
    return tmp_path


def _run_pytest_in(tree: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(SCRIPTS_DIR), env.get("PYTHONPATH", "")) if p)
    env.pop(ss.BYPASS_ENV, None)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=str(tree), env=env, capture_output=True, text=True, timeout=600)


def test_a_passing_run_over_a_still_tree_exits_zero(tmp_path):
    """The guard must not manufacture failures out of a clean run."""
    done = _run_pytest_in(_build_tree(tmp_path, rewrite_during_run=False))

    assert "attestation passed" in done.stdout + done.stderr
    assert done.returncode == 0, done.stdout + done.stderr
    assert "1 passed" in done.stdout + done.stderr


def test_a_run_that_rewrites_a_source_mid_session_exits_nonzero(tmp_path):
    """The acceptance criterion.

    The test file itself passes and the rewrite is a one-line no-op. Without
    the guard this exits 0, which is precisely the reported defect.
    """
    done = _run_pytest_in(_build_tree(tmp_path, rewrite_during_run=True))
    output = done.stdout + done.stderr

    assert "RUN NOT TRUSTWORTHY" in output
    assert "backend/pkg.py" in output
    assert done.returncode != 0, (
        "a run whose source tree changed mid-session reported success")
    assert "2 passed" in output, (
        "the guard must fail the run, not the tests -- the tests did pass")


def test_the_guard_fails_the_run_without_claiming_tests_failed(tmp_path):
    """`FAILED` must never appear for a blocked run."""
    done = _run_pytest_in(_build_tree(tmp_path, rewrite_during_run=True))
    output = done.stdout + done.stderr

    assert "FAILED" not in output, (
        "a blocked run must not be reported as a test failure; that is the "
        "exact failure this guard exists to prevent")
    assert "NOT a test failure" in output


def test_the_bypass_lets_the_run_through(tmp_path):
    tree = _build_tree(tmp_path, rewrite_during_run=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(SCRIPTS_DIR), env.get("PYTHONPATH", "")) if p)
    env[ss.BYPASS_ENV] = "1"
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=str(tree), env=env, capture_output=True, text=True, timeout=600)

    assert done.returncode == 0, done.stdout + done.stderr
    assert "attestation skipped" in done.stdout + done.stderr


def test_a_guard_that_cannot_start_is_reported_not_silent(tmp_path):
    done = _run_pytest_in(_build_tree(tmp_path, rewrite_during_run=False))
    assert "RUN NOT TRUSTWORTHY" not in (done.stdout + done.stderr)
    assert "attestation passed" in (done.stdout + done.stderr)


# ── the guard must not be able to break the suite ───────────────────────

def _tree_without_guard(tmp_path: Path, extra: str = "") -> Path:
    """A runnable tree whose conftest cannot import the guard.

    Reproduces the worst case this safety net has: `scripts/source_stability.py`
    absent and `scripts/` not on `PYTHONPATH`. The conftest is the real one,
    verbatim -- a point this test got wrong the first time, when it wrote an
    empty conftest and then assert an absence that proved nothing.
    """
    (tmp_path / "backend").mkdir()
    write(tmp_path / "backend" / "pkg.py", "V = 1\n")
    write(tmp_path / "pytest.ini")
    (tmp_path / "tests").mkdir()
    write(tmp_path / "tests" / "conftest.py", CONFTEST_SOURCE)
    write(tmp_path / "tests" / "test_ok.py", "def test_ok():\n    pass\n")
    if extra:
        write(tmp_path / "tests" / "test_bad.py", extra)
    return tmp_path


def _run_guardless(tmp_path: Path, extra: str = "") -> subprocess.CompletedProcess:
    _tree_without_guard(tmp_path, extra)
    env = dict(os.environ)
    env["PYTHONPATH"] = ""
    env.pop(ss.BYPASS_ENV, None)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=600)


def test_a_missing_guard_does_not_take_the_suite_down(tmp_path):
    """A safety net must not be the thing that breaks the suite.

    A top-level `import source_stability` raises ModuleNotFoundError out of
    conftest itself: pytest reports "ImportError while loading conftest",
    collects nothing, and the run exits 4 with **zero tests executed**. The guard
    is a new file in a directory that is only on sys.path by conftest's own
    arrangement, so this is reachable by a rename or an uncommitted new file --
    and it would turn the suite red for a *missing* safety net, the opposite of
    the intent. So the tests must still RUN.

    This is the fail-soft half, and it is deliberately paired with
    `test_a_missing_guard_fails_the_run_it_cannot_attest`, which is the
    fail-closed half. Together they say: the suite still runs and reports its
    real results, but the run is not green.
    """
    done = _run_guardless(tmp_path)
    output = done.stdout + done.stderr

    assert "1 passed" in output, output
    assert "GUARD UNAVAILABLE" in output, (
        "an unattested run must say so, or it reads as an attested one")
    assert "unattested" in output


def test_a_missing_guard_reports_no_attestation_claim(tmp_path):
    """The warning must not look like a pass."""
    done = _run_guardless(tmp_path)
    output = done.stdout + done.stderr

    assert "attestation passed" not in output, (
        "an unattested run must not claim an attestation it did not perform")
    assert "RUN NOT TRUSTWORTHY" not in output

    assert "GUARD UNAVAILABLE" in output
    assert "exiting non-zero" in output, (
        "an unattested run must say that it is failing because it cannot attest, "
        "not merely that the guard is missing")
    assert done.returncode != 0, (
        "a run whose guard could not be loaded reported success; the guard fails "
        "OPEN and can be removed to make an unattested run look green")


def test_a_missing_guard_degrades_but_still_reports_test_failures(tmp_path):
    """Degrading must not mask genuine test failures -- and must not hide them.

    Both properties at once, because the combination is what a fail-closed guard
    needs: absent guard means the run goes red, but a *real* test failure is
    still reported as a failure rather than being replaced by the guard's own
    complaint.
    """
    done = _run_guardless(tmp_path, extra="def test_fails():\n    assert False\n")
    output = done.stdout + done.stderr

    assert "1 failed" in output, output
    assert "GUARD UNAVAILABLE" in output
    assert done.returncode != 0


def test_a_missing_guard_fails_the_run_it_cannot_attest(tmp_path):
    """A safety net must fail CLOSED, not open.

    An independent review found this was not true. `pytest_sessionfinish` printed
    a notice and returned without touching `session.exitstatus`, so deleting,
    renaming or breaking `scripts/source_stability.py` disabled the guard and the
    run reported green -- while the module docstring said "fail-closed by
    default". A guard that can be removed to make the thing it covers look fine
    is not a guard.

    Before the fix the exit status was 0. The test file itself still passes, so
    this pins the exit code rather than any assertion.
    """
    done = _run_guardless(tmp_path)
    output = done.stdout + done.stderr

    assert "GUARD UNAVAILABLE" in output
    assert "exiting non-zero" in output, (
        "an unattested run must say that it is failing because it cannot attest, "
        "not merely that the guard is missing")
    assert done.returncode != 0, (
        "a run whose guard could not be loaded reported success; the guard fails "
        "OPEN and can be removed to make an unattested run look green")


def test_the_bypass_variable_ignores_values_that_are_not_enablers():
    """`=0` and `=false` must not disable the guard.

    `os.environ.get(...)` truthiness made every non-empty string an opt-out, so
    an operator setting `MECH_SKIP_SOURCE_STABILITY_CHECK=0` believing they were
    re-enabling it had disabled it. Found by the same independent review.

    Checked on the property alone: constructing a `SessionGuard` would resolve a
    real repository root, which is not what this test is about.
    """
    guard = ss.SessionGuard.__new__(ss.SessionGuard)

    for disabled in ("0", "false", "no", "off", "n", "F", "OFF", " ",
                     "disabled", "none"):
        with mock.patch.dict(os.environ, {ss.BYPASS_ENV: disabled}):
            assert guard.bypassed is False, (
                f"{ss.BYPASS_ENV}={disabled!r} was treated as an opt-out")

    for enabled in ("1", "true", "TRUE", "yes", "Y", " true "):
        with mock.patch.dict(os.environ, {ss.BYPASS_ENV: enabled}):
            assert guard.bypassed is True
