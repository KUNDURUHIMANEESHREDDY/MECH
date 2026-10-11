"""Root pytest configuration.

Session-scoped temporary directories
------------------------------------
``pytest.ini`` pins ``addopts = --basetemp=.pytest-tmp`` -- a single fixed
path shared by every pytest session in this checkout. That is unsafe here for
a reason specific to how this project is worked: **two pytest sessions
routinely run at once**, because more than one agent session works this tree
concurrently.

``TempPathFactory.getbasetemp()`` does this on its *first* call in a session
(``_pytest/tmpdir.py``)::

    if self._basetemp is not None:
        return self._basetemp
    if self._given_basetemp is not None:
        basetemp = self._given_basetemp
        if basetemp.exists():
            rm_rf(basetemp)          # <-- wipes the whole tree
        basetemp.mkdir(mode=0o700)

So the second session to request a ``tmp_path`` deletes the entire
``.pytest-tmp`` -- including temp trees a session that is *still running*
owns. Reproduced directly: a tree planted under a shared basetemp survived a
session that never used ``tmp_path``, and was destroyed by one that did.

The damage lands on exactly the tests that depend on a temp tree surviving
between writing it and using it -- and this suite has several, because they
spawn a real pytest subprocess against a tree they just built:

    test_source_stability.py::test_a_guard_that_cannot_start_is_reported_not_silent
    test_source_stability.py::test_a_missing_guard_does_not_take_the_suite_down
    test_provenance_signing.py::test_promotion_refuses_without_a_key

Those failed in a full-suite run with ``returncode=5`` and
``no tests ran in 0.00s`` -- an empty collection in a tree that had just been
written and did contain a passing test. They pass in isolation, because in
isolation nothing else is running.

The fix is to make the basetemp unique per session rather than to change the
tests. The alternative -- making those three tests resilient -- would leave
every other ``tmp_path`` consumer exposed to the same race, and would paper
over a configuration defect in the one place that can actually fix it.

Why the leaf, and not ``pytest.ini``
-----------------------------------
An earlier draft of this file justified keeping scratch under ``.pytest-tmp/``
as "the deliberate choice to keep scratch inside the checkout". That was an
assumption, not a finding. Checking it afterwards: the setting arrived in
``32944e5`` with a bulk consolidation whose message says nothing about
basetemp, so there was no recorded rationale to appeal to.

The real reason the path cannot move is sharper, and is documented at
``scripts/source_stability.py:172``. ``.pytest-tmp`` is in that module's
``EXCLUDED_DIRS``, so a tree parked underneath it would be excluded wholesale
and the guard would silently watch nothing. ``_is_watched`` was changed to
test the path *relative to the watched root* precisely so this module's own
unit tests -- which ``pytest.ini`` sends to ``.pytest-tmp`` -- still exercise
a real watch. Repointing the basetemp outside the checkout would have undone
that and quietly reduced the guard's coverage to zero.

So the pinned path stays. Adding a leaf below it leaves ``_is_watched``
reading the same relative path as before, which is what that module's tests
assert; they pass unchanged, and a full session still reports 596 watched
files rather than none.

Stale leaves are pruned once per session. Without that, the fix would trade a
collision for unbounded accumulation: pytest reclaims its own numbered
directories but knows nothing about leaves it did not create. A leaf is
removed only when it is more than a day old, so a session that is genuinely
still running is never touched -- pruning a live session would reintroduce the
very failure this file exists to prevent.
"""

from __future__ import annotations

import os
import shutil
import time

#: Set by ``pytest_configure`` so a test can name the session it ran in.
SESSION_BASETEMP: str = ""

#: Scratch leaves older than this are assumed dead and removed. A full suite
#: takes roughly half an hour, so a day-old leaf cannot belong to a running
#: session. Generous on purpose: pruning a live session's directory would
#: reintroduce exactly the failure this file exists to prevent.
_STALE_LEAF_SECONDS = 24 * 60 * 60

#: Prefix identifying a leaf this file created.
_LEAF_PREFIX = "session-"


def _prune_stale_leaves(parent: str, keep: str) -> list[str]:
    """Remove dead session leaves from a previous run. Returns what was removed.

    Only ever called when ``--basetemp`` was pinned, so it cannot touch a
    tree the caller pointed pytest at some other way.
    """
    removed: list[str] = []
    now = time.time()
    try:
        entries = list(os.scandir(parent))
    except (FileNotFoundError, NotADirectoryError):
        return removed

    for entry in entries:
        if not entry.is_dir(follow_symlinks=False):
            continue
        if not entry.name.startswith(_LEAF_PREFIX):
            continue
        if os.path.abspath(entry.path) == os.path.abspath(keep):
            continue
        try:
            age = now - entry.stat(follow_symlinks=False).st_mtime
        except OSError:
            continue
        if age < _STALE_LEAF_SECONDS:
            continue
        shutil.rmtree(entry.path, ignore_errors=True)
        removed.append(entry.name)
    return removed


def pytest_configure(config) -> None:
    """Give this session its own basetemp leaf.

    A no-op when ``--basetemp`` was not pinned by ``pytest.ini`` -- in that
    case pytest already allocates a numbered, session-unique directory and
    there is nothing to fix.
    """
    global SESSION_BASETEMP

    given = getattr(config.option, "basetemp", None)
    if not given:
        # Unpinned: pytest's own default (pytest-of-<user>/pytest-N) is
        # already unique per session.
        SESSION_BASETEMP = ""
        return

    # Keyed on the process id, which is unique per running pytest session.
    # The start time is included too: pids recycle on Windows, so a stale
    # directory from a dead session must not be adopted by a new one.
    unique = f"session-{os.getpid()}-{int(time.time())}"
    leaf = os.path.join(str(given), unique)
    os.makedirs(leaf, exist_ok=True)
    _prune_stale_leaves(str(given), keep=leaf)
    config.option.basetemp = leaf
    SESSION_BASETEMP = leaf