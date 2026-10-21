"""Keep the packaged backend honest: no secrets, no runtime state, no drift.

The Electron installer ships a copy of ``backend/``. The copy is described by
``build.extraResources`` in ``frontend/package.json``, and until this was
written that filter said only "everything except ``__pycache__`` and
``*.pyc``". It did not say "except the control-plane bearer token" and it did
not say "except the research database", so
``backend/storage/.mech_api_token`` and ``backend/storage/mech.db`` went into
every installer. ``frontend/electron/main.js`` then read the *packaged*
token and injected it into the renderer, so every install of the app shared
one credential with every other install, and the developer's research data
was redistributed with the application.

Two guards, because those are two different failures.

*Part one -- the filter.* ``frontend/package.json`` now excludes the token,
the database and the runtime-state directories. Note the sibling build,
``scripts/build_standalone.py``, produces ``MECH-standalone/`` by copying
``backend/`` with ``shutil.ignore_patterns``; it needs the same exclusions
and does not have them yet, so a fresh standalone build still carries the
token. It is outside this change's scope and is called out in the report.

*Part two -- the drift.* A packaged backend can differ from the source tree
without anyone noticing: the artifact is a build output, it is gitignored,
and nothing compared it to the source it claims to be. A stale copy ships
silently, and the app runs code that no longer exists in the repository.
``compare()`` answers "does the packaged backend still match the source?".

What the source set is
----------------------

The source set is derived from **git**: every tracked file under
``backend/``, plus every untracked file that git does not ignore, minus the
build filter's exclusions and minus the runtime state that must never ship.
Git is the repository's own definition of source, and it is the only one
that can tell the two apart:

* a ``.dsh-ops-update-*.lock`` file left behind by a tool run is untracked
  noise, not source, and must not make every comparison fail;
* ``mech.db``, the bearer token, the evidence directory and the SAE
  checkpoints are deliberately *untracked*, and must not be compared --
  except on the packaged side, where their presence is exactly the failure
  this guard exists to catch.

The comparison is two-way. A file in the package that the source does not
have is reported as extra, because a one-way comparison would miss a file
that was put into the package and nothing else. That is the difference
between "the package is a copy of the source" and "the package contains the
source", and only the second is worth shipping.

Rules of the comparison
-----------------------

*Paths* are compared case-insensitively, because the filesystem this runs on
may be case-insensitive: on Windows, ``core/Auth.py`` and ``core/auth.py``
are one file, and reporting it as both missing and extra would be noise
rather than a finding. The path is printed as it was found.

*Symlinks and junctions* are never followed. A symlinked file is not
compared, and a symlinked or junctioned directory is not descended into: a
link points somewhere else, and following it would compare two different
trees against each other.

*Empty comparisons are failures.* If the source set is empty, nothing was
verified, and the check says so and exits non-zero. "Clean" with zero files
compared is precisely the false negative this guard exists to kill.

Usage
-----

    python scripts/packaged_backend_guard.py check
    python scripts/packaged_backend_guard.py check --target path/to/backend
    python scripts/packaged_backend_guard.py stage
    python scripts/packaged_backend_guard.py stage --dry-run

``check`` is strictly read-only -- it never writes into a packaged tree.
It exits 0 when the trees agree, 1 when they have drifted (or when the
comparison was not possible), and 2 when a packaged tree does not exist:
"not checked" must never be readable as a pass.

``stage`` copies the source set into the packaged tree, deletes what the
source does not have, and removes every token, database and runtime-state
file it finds. It never copies a secret in. Staging is an explicit,
separate command: ``check`` must never quietly repair the thing it checks.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# `scripts/` is not a package (see tests/pytest/conftest.py), so the sibling
# module is imported by path rather than by package name.
import source_stability  # noqa: E402

#: The repository root, resolved from this file's location and not from the
#: current working directory, so the guard works when it is run from
#: anywhere -- a CI step that starts in a temporary directory, a shell that
#: happens to be in ``C:\``.
REPO_ROOT = source_stability.find_repo_root(_SCRIPTS_DIR)

#: The backend as it exists in the source tree.
SOURCE_BACKEND = REPO_ROOT / "backend"

#: The packaged trees this guard knows about, with a label for the output.
#: Both are gitignored build outputs and both are regenerated by a build.
DEFAULT_TARGETS: Tuple[Tuple[str, Path], ...] = (
    ("electron", REPO_ROOT / "frontend" / "release" / "win-unpacked" / "resources" / "backend"),
    ("standalone", REPO_ROOT / "MECH-standalone" / "backend"),
)

#: The same targets without their labels.
DEFAULT_PACKAGED_TARGETS: Tuple[Path, ...] = tuple(path for _, path in DEFAULT_TARGETS)

#: Where electron-builder's copy filter is declared. It is read from there
#: rather than restated, so the guard stages exactly what the build copies:
#: one filter, not two that can disagree.
PACKAGE_JSON = REPO_ROOT / "frontend" / "package.json"

#: Directories never descended into, per scripts/source_stability.py. Used
#: for the filesystem fallback, when git cannot describe the tree.
EXCLUDED_DIRS = source_stability.EXCLUDED_DIRS

#: Directories pruned from both sides of a comparison. ``__pycache__`` holds
#: compiled bytecode, which is not part of a build's output.
NOISE_DIRS = frozenset({"__pycache__"})

#: Files ignored on both sides of a comparison: compiled bytecode, and the
#: lock files a tool run leaves behind. They are dev-machine noise, not
#: content, and comparing them would report a difference on every run.
NOISE_SUFFIXES = (".pyc", ".lock")

#: Secrets and runtime state that must never reach a packaged tree. The
#: first four mirror the exclusions added to ``frontend/package.json``; the
#: last two are the general form, restated here so the guard does not depend
#: on that file being well-formed. A typo in the build filter must not
#: become a shipped credential.
FORBIDDEN_PATTERNS: Tuple[str, ...] = (
    "storage/.mech_api_token",
    "storage/*.db*",
    "storage/evidence/**",
    "storage/v2_jobs/**",
    "*.db*",
    "*.lock",
)


class GuardError(RuntimeError):
    """Something that stops the guard from doing its job."""


# ── the build filter ─────────────────────────────────────────────────────

def _glob_to_regex(pattern: str) -> "re.Pattern[str]":
    """Translate one electron-builder filter glob into a regex.

    Supports the subset the build filter uses: ``**`` (any number of path
    segments), ``*`` (any run of characters inside one segment) and ``?``.
    ``fnmatch`` is not enough -- its ``*`` crosses ``/``, so ``**/*`` would
    fail to match a file at the root of the tree, which is exactly the
    shape of ``backend/main.py``.
    """
    out: List[str] = []
    i = 0
    n = len(pattern)
    while i < n:
        if pattern.startswith("**/", i):
            out.append("(?:[^/]+/)*")
            i += 3
        elif pattern.startswith("/**", i) and i + 3 == n:
            out.append("/.*")
            i += 3
        elif pattern.startswith("**", i) and i + 2 == n:
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def _pattern_matches(pattern: str, rel_path: str) -> bool:
    """Whether ``rel_path`` (POSIX, relative to the tree root) matches ``pattern``.

    A pattern without a ``/`` matches at any depth, which is how
    electron-builder reads a bare name: ``!__pycache__`` excludes every
    ``__pycache__`` directory in the tree, not only one at the root.
    """
    if "/" not in pattern:
        # `fnmatch` compares case-insensitively on Windows, which is the
        # safer direction for a rule whose whole job is to catch a secret.
        return any(fnmatch.fnmatch(part, pattern)
                   for part in rel_path.split("/"))
    return bool(_glob_to_regex(pattern).match(rel_path))


class BuildFilter:
    """The electron-builder ``extraResources`` filter, as a predicate."""

    def __init__(self, patterns: Sequence[str]) -> None:
        self.patterns = [str(p) for p in patterns]

    def excludes(self, rel_path: str) -> bool:
        """Whether ``rel_path`` is excluded by this filter.

        Last match wins, the way an ordered filter list behaves: ``**/*``
        includes everything and the ``!`` patterns after it take files back
        out.
        """
        excluded = False
        for pattern in self.patterns:
            if pattern.startswith("!"):
                if _pattern_matches(pattern[1:], rel_path):
                    excluded = True
            elif _pattern_matches(pattern, rel_path):
                excluded = False
        return excluded

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "BuildFilter(%r)" % (self.patterns,)


def read_build_filter(package_json: Path = PACKAGE_JSON) -> BuildFilter:
    """The filter electron-builder applies to the backend copy.

    The entry is found by its destination (``"to": "backend"``), not by its
    source: the source is now the staged build directory (``build/backend``),
    and the filter is the part that must not change. The second loop keeps an
    older package.json -- one that still says ``../backend`` -- working.
    """
    try:
        data = json.loads(Path(package_json).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GuardError("could not read %s: %s" % (package_json, exc)) from exc
    entries = data.get("build", {}).get("extraResources", [])
    for entry in entries:
        if str(entry.get("to", "")) == "backend":
            return BuildFilter(entry.get("filter", []))
    for entry in entries:
        origin = str(entry.get("from", "")).replace("\\", "/").rstrip("/")
        if origin in ("../backend", "backend", "build/backend"):
            return BuildFilter(entry.get("filter", []))
    raise GuardError("%s has no extraResources entry that copies the backend"
                     % package_json)


def is_forbidden(rel_path: str,
                 patterns: Optional[Sequence[str]] = None) -> bool:
    """Whether ``rel_path`` is a secret or runtime state that must not ship."""
    rules = tuple(patterns) if patterns is not None else FORBIDDEN_PATTERNS
    return any(_pattern_matches(pattern, rel_path) for pattern in rules)


# ── walking a tree ───────────────────────────────────────────────────────

def _is_link_or_junction(path: Path) -> bool:
    """Whether ``path`` is a symbolic link or a Windows junction.

    A junction is a reparse point rather than a symbolic link, so
    ``islink()`` alone does not catch it -- and a directory link would
    otherwise pull a second tree into the comparison.
    """
    try:
        st = os.lstat(path)
    except OSError:
        return True
    if stat.S_ISLNK(st.st_mode):
        return True
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(st, "st_file_attributes", 0) & reparse)


def _iter_files(root: Path,
                prune_dirs: Optional[Sequence[str]] = None) -> Iterator[Tuple[str, Path]]:
    """Yield ``(relative POSIX path, path)`` for every file under ``root``.

    Symlinks and junctions are never followed. A symlinked file is not
    listed at all, and a symlinked or junctioned directory is not descended
    into: a link points somewhere else, and following it would compare two
    different trees against each other.
    """
    prune = set(prune_dirs or ())
    stack: List[Tuple[Path, str]] = [(Path(root), "")]
    while stack:
        directory, prefix = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue
        for entry in entries:
            rel = "%s/%s" % (prefix, entry.name) if prefix else entry.name
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue
            if is_dir:
                if entry.name in prune:
                    continue
                if _is_link_or_junction(Path(entry.path)):
                    continue
                stack.append((Path(entry.path), rel))
            elif entry.is_file(follow_symlinks=False):
                yield rel, Path(entry.path)


def _sha256(path: Path) -> str:
    """sha256 of a file's bytes, or ``""`` when it cannot be read."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def _key(rel_path: str) -> str:
    """The comparison key for a relative path.

    Case-insensitive, because the filesystem may be: on Windows
    ``core/Auth.py`` and ``core/auth.py`` are one file, and reporting it as
    both missing and extra would be noise rather than a finding.
    """
    return os.path.normcase(rel_path)


def is_noise(rel_path: str) -> bool:
    """Whether a file is dev-machine noise rather than content."""
    return rel_path.endswith(NOISE_SUFFIXES)


def find_forbidden(root: Path,
                   patterns: Optional[Sequence[str]] = None) -> List[str]:
    """The forbidden files (token, database, runtime state) present under ``root``."""
    rules = tuple(patterns) if patterns is not None else FORBIDDEN_PATTERNS
    return sorted(rel for rel, _ in _iter_files(Path(root))
                  if is_forbidden(rel, rules))


# ── the source set ───────────────────────────────────────────────────────

def _git_files(root: Path) -> Optional[List[str]]:
    """The git-visible files under ``root``, or ``None`` when git cannot answer.

    ``git ls-files --cached --others --exclude-standard`` lists the tracked
    files and the untracked ones that git does not ignore.
    ``--exclude-standard`` is what keeps the untracked runtime state -- the
    database, the bearer token, the evidence directory -- out of the source
    set, while the tracked files stay in.
    """
    try:
        relative = Path(root).resolve().relative_to(REPO_ROOT)
    except ValueError:
        return None
    prefix = relative.as_posix()
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard",
             "--", prefix],
            cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    names = proc.stdout.decode("utf-8", "replace").split("\0")
    out: List[str] = []
    for name in names:
        if not name:
            continue
        name = name.replace("\\", "/")
        if prefix not in ("", "."):
            if not name.startswith(prefix + "/"):
                continue
            name = name[len(prefix) + 1:]
        out.append(name)
    return out


def source_set(source_root: Path,
               build_filter: Optional[BuildFilter] = None) -> List[str]:
    """The relative paths of the files that belong in a packaged backend.

    Git-derived where possible -- tracked files plus untracked files that
    git does not ignore -- and a filesystem walk otherwise, with the same
    exclusions either way. The build filter and the runtime-state
    exclusions are applied on top, so the result is "what a real build
    would stage, minus the state that must never ship".
    """
    root = Path(source_root)
    if build_filter is None:
        build_filter = read_build_filter()
    candidates = _git_files(root)
    if not candidates:
        # Git described nothing: the tree is not tracked (a synthetic tree in
        # a test, or a directory outside the repository). Fall back to the
        # filesystem with the same exclusions, rather than comparing against
        # an empty set -- an empty comparison verifies nothing.
        candidates = [rel for rel, _ in _iter_files(root, prune_dirs=EXCLUDED_DIRS)]
    out: List[str] = []
    for rel in sorted(candidates):
        if build_filter.excludes(rel):
            continue
        if is_forbidden(rel):
            continue
        if not (root / rel).is_file():
            # Tracked but deleted from the working tree: there is nothing to
            # compare, and nothing to ship.
            continue
        out.append(rel)
    return out


# ── comparing ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FileEntry:
    """One file, identified by its path relative to the tree root."""

    path: str
    sha256: str


@dataclass(frozen=True)
class ChangedFile:
    """One file that exists in both trees with different contents."""

    path: str
    source_sha256: str
    packaged_sha256: str


@dataclass(frozen=True)
class DriftReport:
    """The three ways two trees can disagree."""

    only_in_source: Tuple[FileEntry, ...] = ()
    only_in_packaged: Tuple[FileEntry, ...] = ()
    changed: Tuple[ChangedFile, ...] = ()
    compared: int = 0

    @property
    def ok(self) -> bool:
        """Whether the two trees agree."""
        return not (self.only_in_source or self.only_in_packaged or self.changed)

    def summary(self) -> str:
        return ("only_in_source=%d only_in_packaged=%d changed=%d compared=%d"
                % (len(self.only_in_source), len(self.only_in_packaged),
                   len(self.changed), self.compared))


def _index(root: Path, files: Sequence[str]) -> Dict[str, Tuple[str, str]]:
    """``{comparison key: (path as found, sha256)}`` for the given files."""
    index: Dict[str, Tuple[str, str]] = {}
    for rel in files:
        full = Path(root) / rel
        if not full.is_file():
            continue
        index[_key(rel)] = (rel, _sha256(full))
    return index


def compare(source_root: Path, packaged_root: Path,
            source_paths: Optional[Sequence[str]] = None) -> DriftReport:
    """Compare the source backend tree with a packaged one.

    The comparison is two-way: a file the source has and the package does
    not is reported, and a file the package has and the source does not is
    reported too. The second half is what catches a file put into the
    package by hand -- including the bearer token, which a one-way
    comparison would wave through.
    """
    source_root = Path(source_root)
    packaged_root = Path(packaged_root)
    if not packaged_root.is_dir():
        raise GuardError("packaged tree does not exist: %s" % packaged_root)
    if source_paths is None:
        source_paths = source_set(source_root)

    source = _index(source_root, source_paths)
    packaged = _index(packaged_root,
                       [rel for rel, _ in _iter_files(packaged_root,
                                                      prune_dirs=NOISE_DIRS)
                        if not is_noise(rel)])

    only_in_source = tuple(FileEntry(source[key][0], source[key][1])
                           for key in sorted(set(source) - set(packaged)))
    only_in_packaged = tuple(FileEntry(packaged[key][0], packaged[key][1])
                             for key in sorted(set(packaged) - set(source)))
    changed = tuple(ChangedFile(source[key][0], source[key][1],
                                packaged[key][1])
                    for key in sorted(set(source) & set(packaged))
                    if source[key][1] != packaged[key][1])
    return DriftReport(only_in_source=only_in_source,
                       only_in_packaged=only_in_packaged,
                       changed=changed,
                       compared=len(source))


# ── staging ──────────────────────────────────────────────────────────────

@dataclass
class StageResult:
    """What one staging run did."""

    copied: List[str] = field(default_factory=list)
    deleted: List[str] = field(default_factory=list)
    removed_secrets: List[str] = field(default_factory=list)
    unchanged: int = 0


def _prune_empty_dirs(root: Path) -> None:
    """Remove directories left empty by the deletions, deepest first."""
    for dirpath, _dirnames, _filenames in os.walk(root, topdown=False):
        directory = Path(dirpath)
        if directory == root:
            continue
        try:
            if not any(directory.iterdir()):
                directory.rmdir()
        except OSError:
            pass


def stage(source_root: Path, packaged_root: Path,
          build_filter: Optional[BuildFilter] = None,
          dry_run: bool = False) -> StageResult:
    """Copy the source backend into the packaged tree, and clean the rest out.

    Files that the source does not have are deleted, and every token,
    database and runtime-state file is removed from the packaged tree. No
    secret is ever copied in: the forbidden list is applied to the source
    before anything is copied.
    """
    source_root = Path(source_root)
    packaged_root = Path(packaged_root)
    if not source_root.is_dir():
        raise GuardError("source tree does not exist: %s" % source_root)
    if build_filter is None:
        build_filter = read_build_filter()
    wanted = source_set(source_root, build_filter)
    result = StageResult()

    if not dry_run:
        packaged_root.mkdir(parents=True, exist_ok=True)

    for rel in wanted:
        source_file = source_root / rel
        destination = packaged_root / rel
        if destination.is_file() and _sha256(destination) == _sha256(source_file):
            result.unchanged += 1
            continue
        result.copied.append(rel)
        if not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination)

    # Materialise the list before deleting anything: removing files while a
    # directory is being scanned is how a staging run deletes half a tree.
    for rel, full in list(_iter_files(packaged_root, prune_dirs=NOISE_DIRS)):
        if rel in wanted:
            continue
        result.deleted.append(rel)
        if is_forbidden(rel):
            result.removed_secrets.append(rel)
        if not dry_run:
            try:
                os.remove(full)
            except OSError:
                pass

    if not dry_run:
        _prune_empty_dirs(packaged_root)
    return result


# ── command line ─────────────────────────────────────────────────────────

def _print_report(report: DriftReport, packaged_count: int) -> None:
    print("  compared %d source file(s) against %d packaged file(s)"
          % (report.compared, packaged_count))
    print("  only_in_source (%d)" % len(report.only_in_source))
    for entry in report.only_in_source:
        print("    %s  sha256:%s" % (entry.path, entry.sha256))
    if not report.only_in_source:
        print("    (none)")
    print("  only_in_packaged (%d)" % len(report.only_in_packaged))
    for entry in report.only_in_packaged:
        print("    %s  sha256:%s" % (entry.path, entry.sha256))
    if not report.only_in_packaged:
        print("    (none)")
    print("  changed (%d)" % len(report.changed))
    for entry in report.changed:
        print("    %s  expected=sha256:%s  actual=sha256:%s"
              % (entry.path, entry.source_sha256, entry.packaged_sha256))
    if not report.changed:
        print("    (none)")


def _check(source: Path, targets: Sequence[Path]) -> int:
    print("[check] source backend: %s" % source)
    build_filter = read_build_filter()
    source_paths = source_set(source, build_filter)
    print("[check] source set: %d file(s)" % len(source_paths))
    if not source_paths:
        # An empty comparison verifies nothing. It is not a clean result.
        print("[check] FAILURE: the source set is empty, so nothing was "
              "compared. An empty comparison is not a pass.")
        return 1
    missing = False
    drift = False
    for target in targets:
        print()
        print("[check] packaged backend: %s" % target)
        if not target.is_dir():
            missing = True
            print("  NOT CHECKED: this packaged tree does not exist, so the "
                  "package cannot be verified.")
            print("  An absent tree is not a pass.")
            continue
        report = compare(source, target, source_paths)
        packaged_count = len([rel for rel, _ in _iter_files(target,
                                                            prune_dirs=NOISE_DIRS)
                              if not is_noise(rel)])
        _print_report(report, packaged_count)
        if not report.ok:
            drift = True
    print()
    if missing and drift:
        print("RESULT: NOT CHECKED and DRIFT -- a packaged tree is missing and "
              "another has drifted from the source.")
        return 2
    if missing:
        print("RESULT: NOT CHECKED -- a packaged tree is missing, so the "
              "package was not verified. An absent tree is not a pass.")
        return 2
    if drift:
        print("RESULT: DRIFT -- the packaged backend does not match the "
              "source backend.")
        return 1
    print("RESULT: OK -- every packaged backend matches the source backend.")
    return 0


def _stage(source: Path, targets: Sequence[Path], dry_run: bool = False) -> int:
    build_filter = read_build_filter()
    print("[stage] build filter: %s" % " ".join(build_filter.patterns))
    for target in targets:
        print()
        print("[stage] %s -> %s" % (source, target))
        result = stage(source, target, build_filter=build_filter, dry_run=dry_run)
        print("  copied or updated (%d)" % len(result.copied))
        for rel in result.copied:
            print("    + %s" % rel)
        print("  deleted (%d)" % len(result.deleted))
        for rel in result.deleted:
            print("    - %s" % rel)
        print("  removed secrets and runtime state (%d)" % len(result.removed_secrets))
        for rel in result.removed_secrets:
            print("    ! %s" % rel)
        print("  already current (%d)" % result.unchanged)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="packaged_backend_guard.py",
        description="Compare the packaged backend against the source backend, "
                    "or stage a fresh copy of the source into it.")
    parser.add_argument("mode", choices=("check", "stage"),
                        help="check: report drift and exit non-zero if there is "
                             "any; stage: copy the source into the package")
    parser.add_argument("--source", default=str(SOURCE_BACKEND),
                        help="the source backend tree (default: %(default)s)")
    parser.add_argument("--target", action="append", default=None,
                        help="a packaged backend tree; repeatable (default: the "
                             "Electron package and MECH-standalone)")
    parser.add_argument("--dry-run", action="store_true",
                        help="stage only: report what would change, change nothing")
    args = parser.parse_args(argv)

    source = Path(args.source).resolve()
    targets = ([Path(t).resolve() for t in args.target] if args.target
               else list(DEFAULT_PACKAGED_TARGETS))
    try:
        if args.mode == "check":
            return _check(source, targets)
        return _stage(source, targets, dry_run=args.dry_run)
    except GuardError as exc:
        print("packaged-backend-guard: ERROR: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
