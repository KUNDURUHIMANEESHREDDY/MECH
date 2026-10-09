"""Pytest configuration. Ensures the ``backend/`` directory is on sys.path so
``from api.dispatcher import build_dispatcher`` resolves correctly.

Also installs the source-stability guard -- see ``scripts/source_stability.py``
for the defect it exists for. In short: a nine-minute suite can straddle a
source-file write and report the resulting ImportError as a test failure, with
no way to tell it apart from a real regression. The guard snapshots the watched
sources at session start and again at session end, and refuses to let an
untrustworthy run look green.

The guard is a safety net, so it is not allowed to be the thing that breaks the
suite: if it cannot be imported or cannot run, this file says so once and lets
the tests run unattested, rather than raising out of conftest and collecting
nothing.

All reporting is done in ``pytest_sessionfinish``, never at session start. Pytest
runs ``pytest_sessionstart`` under its global output capture, so anything printed
there is swallowed -- and a warning nobody can read is the same as no warning.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "backend"))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# The guard lives in scripts/ so it is usable outside pytest (see its CLI), and
# scripts/ is not a package. Adding it to sys.path is what conftest already did
# for backend/, for the same reason.
SCRIPTS_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# The guard is a *safety net*, so it must not be able to take the suite down with
# it. A top-level `import source_stability` raises ModuleNotFoundError out of
# conftest itself when the file is missing or renamed: pytest reports "ImportError
# while loading conftest", collects nothing, and the run exits 4 with zero tests
# executed. That turns a missing safety net into an outage, the opposite of the
# intent, so degrade to "no attestation" and let the suite run.
try:
    import source_stability
except Exception as _exc:  # pragma: no cover - exercised via a synthetic tree
    source_stability = None  # type: ignore[assignment]
    _GUARD_IMPORT_ERROR = "%s: %s" % (type(_exc).__name__, _exc)

# Shared bearer token for HTTP-level tests. The control plane (backend/main.py)
# requires Authorization on /api/*, /mcp and /health/subsystems; without a
# default here every TestClient test would 401. test_auth_control_plane.py
# overrides per-test via monkeypatch + auth_mod.reset_cache().
os.environ.setdefault("MECH_API_TOKEN", "pytest-shared-token")

#: How many watched files edited in the last two minutes before it is worth
#: mentioning at all. A single file is an ordinary save; a handful is a peer
#: session working the same tree.
RECENT_EDIT_WARN_THRESHOLD = 3

# State written by pytest_sessionstart and read at pytest_sessionfinish.
_SOURCE_GUARD = None
_UNAVAILABLE_REASON = None
_RECENT_EDIT_NOTE = None


def pytest_sessionstart(session):
    """Snapshot the watched sources before any test module is imported.

    Collects state only -- see the module docstring for why nothing is printed
    here.
    """
    global _SOURCE_GUARD, _UNAVAILABLE_REASON, _RECENT_EDIT_NOTE
    _SOURCE_GUARD = _UNAVAILABLE_REASON = _RECENT_EDIT_NOTE = None

    if source_stability is None:
        _UNAVAILABLE_REASON = _GUARD_IMPORT_ERROR
        return

    try:
        guard = source_stability.SessionGuard()
        guard.start()
        _SOURCE_GUARD = guard
    except Exception as exc:
        # Refusing to run because the guard itself broke would be worse than
        # running unattested, but it must never be silent.
        _UNAVAILABLE_REASON = "could not start: %s: %s" % (
            type(exc).__name__, exc)
        return

    # Advisory only, deliberately: an edit that landed *before* the run began
    # cannot corrupt it, so this must not block anything. It is worth reporting
    # at all because the dangerous case -- a peer saving every few minutes -- is
    # otherwise indistinguishable from an ordinary recent edit, and finding out
    # after ten minutes is ten minutes wasted.
    try:
        fresh = source_stability.recent_modifications(120, guard.root)
    except Exception:
        return
    if len(fresh) >= RECENT_EDIT_WARN_THRESHOLD:
        _RECENT_EDIT_NOTE = (
            "%d watched source file(s) were modified in the last two minutes, "
            "which suggests another session is editing this tree. If that "
            "happens again while the tests run, this run will be reported as "
            "untrustworthy." % len(fresh))


def pytest_sessionfinish(session, exitstatus):
    """Attest the span the session covered.

    A run that began and ended on different sources is reported as a failure,
    not a success. This is the whole point of the guard: an ImportError from a
    half-written file and an ImportError from a real defect are otherwise
    indistinguishable, and only the latter is worth anybody's afternoon.
    """
    if _RECENT_EDIT_NOTE:
        print("\n[source-stability] note: %s" % _RECENT_EDIT_NOTE)

    if _UNAVAILABLE_REASON is not None:
        # Said here rather than at session start so it actually reaches the
        # terminal. The session's real test results are still reported normally;
        # only the attestation is absent.
        print("\n[source-stability] GUARD UNAVAILABLE -- this run is unattested, "
              "so a source change during it cannot be detected and its result "
              "may describe a half-written file.")
        print("  (%s)" % _UNAVAILABLE_REASON)
        return

    guard = _SOURCE_GUARD
    if guard is None:
        return

    try:
        report = guard.finish()
    except Exception as exc:
        # Could not attest. The honest outcome is that no claim is made, so the
        # run cannot be reported green.
        print("\n[source-stability] could not attest this run: %s: %s"
              % (type(exc).__name__, exc))
        session.exitstatus = 1
        return

    print(source_stability.render_report(report))

    if not (report.stable or report.bypassed):
        session.exitstatus = 1
