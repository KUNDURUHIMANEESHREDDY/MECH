"""The packaged backend must match the source, and must not carry secrets.

Two separate claims, tested separately.

*The filter.* ``frontend/package.json`` declares which files the Electron
installer copies out of the staged backend. That declaration is the only
thing between the developer's bearer token and every installer, so it is
asserted directly, and asserted as behaviour rather than as a string: the
filter must exclude the token, the database and the runtime-state
directories, and it must not exclude the source files. This test needs no
build artifact and runs everywhere, including a clean CI checkout.

*The build path.* The Electron build stages the backend itself, from the
repository's source for ``backend/``, and the standalone build stages the
same set. Those are two implementations of one rule -- one in Node, one in
Python -- so the tests here drive both and assert that they agree, and
that what they stage holds no secret and nothing git ignores.

*The artifacts.* What is on disk must match the source backend, and must not
contain a token, a database or runtime state. Those tests need a build
artifact, so they skip when there is none -- and they say so. An absent
artifact is not a pass: it means nobody looked.

The tests in between exercise the guard itself on synthetic trees, so every
scenario is covered even when no artifact exists to compare against: an
absent artifact, an equal pair, a one-byte change, a file added to the
package, harness noise, and a changed non-Python file.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import packaged_backend_guard as pbg  # noqa: E402

#: The exclusions the backend copy filter must carry.
REQUIRED_EXCLUSIONS = (
    "!storage/.mech_api_token",
    "!storage/*.db*",
    "!storage/evidence/**",
    "!storage/v2_jobs/**",
)

#: The filter entries that must survive: the copy itself, and the two
#: exclusions that were already there.
REQUIRED_PATTERN_ENTRIES = ("**/*", "!__pycache__", "!**/*.pyc")

#: What must not be inside a packaged backend. "*.db*" subsumes
#: "storage/*.db*" and also catches a database that lives outside storage/,
#: such as backend/runtime.db.
FORBIDDEN_IN_PACKAGE = (
    "storage/.mech_api_token",
    "*.db*",
    "storage/evidence/**",
    "storage/v2_jobs/**",
)

#: The packaged trees this test knows about.
TARGETS = list(pbg.DEFAULT_TARGETS)
TARGET_IDS = [label for label, _ in TARGETS]

#: The Node helper the Electron build stages the backend with.
NODE_HELPER = "./frontend/scripts/backend_source_set"


def _tree(root: Path, files) -> Path:
    """Write ``{relative path: text}`` into ``root``."""
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _backend_entry():
    """The extraResources entry that copies the backend into the installer."""
    package_json = json.loads(
        (REPO_ROOT / "frontend" / "package.json").read_text(encoding="utf-8"))
    entries = package_json["build"]["extraResources"]
    backends = [entry for entry in entries if entry.get("to") == "backend"]
    assert len(backends) == 1, (
        "frontend/package.json must declare exactly one extraResources entry "
        "with \"to\": \"backend\"; it declares %d" % len(backends))
    return backends[0]


def _backend_filter():
    """The filter electron-builder applies to the backend copy."""
    return _backend_entry()["filter"]


def _skip_without_artifact(target: Path) -> None:
    """Skip -- loudly -- when there is no artifact to check."""
    if not target.is_dir():
        pytest.skip(
            "NOT VERIFIED: there is no packaged backend at %s, so this check "
            "did not happen. An absent artifact is not a pass." % target)


def _sample(paths, limit: int = 10) -> str:
    names = list(paths)
    if len(names) > limit:
        names = names[:limit] + ["..."]
    return ", ".join(names)


def _snapshot(root: Path):
    """``{relative path: (size, mtime_ns, sha256)}`` for a tree."""
    out = {}
    for rel, full in pbg._iter_files(root):
        stat_result = full.stat()
        out[rel] = (stat_result.st_size, stat_result.st_mtime_ns,
                    pbg._sha256(full))
    return out


def _node_helper(body: str):
    """Run a snippet against the Node build helper and return its JSON."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH, so the Node build helper cannot be "
                    "exercised here")
    script = ("const m = require('%s');" % NODE_HELPER) + body
    proc = subprocess.run([node, "-e", script], cwd=str(REPO_ROOT),
                          capture_output=True, text=True)
    assert proc.returncode == 0, (
        "the Node build helper failed: %s" % (proc.stderr or "").strip())
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _git_ignored_files():
    """The files under ``backend/`` that git ignores."""
    proc = subprocess.run(
        ["git", "ls-files", "-z", "--others", "--ignored", "--exclude-standard",
         "--", "backend"],
        cwd=str(REPO_ROOT), capture_output=True)
    assert proc.returncode == 0, "git ls-files --ignored failed"
    out = []
    for name in proc.stdout.decode("utf-8", "replace").split("\0"):
        if not name:
            continue
        out.append(name.replace("\\", "/")[len("backend/"):])
    return out


# ── the build filter (always runs) ───────────────────────────────────────

def test_packaging_filter_excludes_the_token_and_the_database():
    """The copy filter must exclude the token, the database and runtime state."""
    filt = _backend_filter()
    for pattern in REQUIRED_EXCLUSIONS:
        assert pattern in filt, (
            "the backend extraResources filter must exclude %r; it is currently %r"
            % (pattern, filt))
    for pattern in REQUIRED_PATTERN_ENTRIES:
        assert pattern in filt, (
            "the backend extraResources filter must keep %r; it is currently %r"
            % (pattern, filt))


def test_packaging_filter_behaves_as_declared():
    """The declared filter must exclude the secrets and keep the sources."""
    build_filter = pbg.read_build_filter()
    for path in ("storage/.mech_api_token", "storage/mech.db",
                 "storage/mech.db-wal", "storage/evidence/r0.json",
                 "storage/v2_jobs/j0.json", "runtime.db"):
        assert build_filter.excludes(path), (
            "the build filter does not exclude %r" % path)
    for path in ("main.py", "core/auth.py", "storage/database.py"):
        assert not build_filter.excludes(path), (
            "the build filter must not exclude the source file %r" % path)


def test_the_build_copies_the_staged_backend_not_the_raw_tree():
    """The build must package the staged source set, not whatever is on disk."""
    entry = _backend_entry()
    assert entry["from"] == "build/backend", (
        "the backend extraResource must copy the staged build directory; it "
        "copies %r" % entry["from"])

    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "frontend/build/" in gitignore, (
        "frontend/build/ holds a copy of the backend and must be gitignored")

    build_js = (REPO_ROOT / "frontend" / "scripts" / "build.js").read_text(
        encoding="utf-8")
    assert "backend_source_set" in build_js, (
        "the build must stage the backend from the source set")
    main_body = build_js[build_js.index("function main()"):]
    assert main_body.index("stageBackendForBuild()") < \
        main_body.index("resolveExtraResources()"), (
            "the backend must be staged before the extraResources are "
            "resolved, or the build packages the raw tree")


def test_the_repo_root_comes_from_the_script_not_the_cwd(tmp_path, monkeypatch):
    """The guard must work when it is run from outside the repository."""
    monkeypatch.chdir(tmp_path)
    assert (pbg.REPO_ROOT / "backend").is_dir()
    assert (pbg.REPO_ROOT / "pytest.ini").is_file()
    assert pbg.SOURCE_BACKEND.is_dir()
    assert pbg.PACKAGE_JSON.is_file()


# ── the two implementations of the source set (always run) ───────────────

def test_the_build_helper_and_the_guard_agree_on_the_source_set():
    """Node and Python must stage and compare the same files.

    The tree is edited by other sessions, so a single comparison can catch
    a file added or removed between the two calls; the comparison is
    retried, and a persistent difference is a real disagreement.
    """
    data = {}
    python_files = []
    for _ in range(3):
        data = _node_helper(
            "console.log(JSON.stringify({files: m.sourceSet(), "
            "forbidden: m.FORBIDDEN}));")
        python_files = pbg.source_set(pbg.SOURCE_BACKEND)
        if list(data["files"]) == list(python_files):
            break
    else:
        pytest.fail(
            "the Node build helper and the Python guard disagree about the "
            "source set:\n  only in node: %s\n  only in python: %s"
            % (_sample(sorted(set(data["files"]) - set(python_files))),
               _sample(sorted(set(python_files) - set(data["files"])))))

    assert list(data["forbidden"]) == list(pbg.FORBIDDEN_PATTERNS), (
        "the Node denylist and the Python denylist must be the same list")


def test_the_staged_build_directory_carries_no_secret_and_no_ignored_file(tmp_path):
    """What the build stages must hold no secret and nothing git ignores."""
    destination = tmp_path / "build" / "backend"
    _node_helper("console.log(JSON.stringify({count: m.stageBackend(%s).length}));"
                % json.dumps(str(destination)))

    staged = sorted(path.relative_to(destination).as_posix()
                    for path in destination.rglob("*") if path.is_file())
    assert staged, "the staged backend must not be empty"

    for offender in pbg.find_forbidden(destination):
        pytest.fail("the staged backend carries %r, which must never ship"
                    % offender)

    for rel in _git_ignored_files():
        assert rel not in staged, (
            "the staged backend carries %r, which git ignores; only source "
            "belongs in a package" % rel)


# ── the guard itself (always runs) ───────────────────────────────────────

def test_equal_trees_have_no_drift(tmp_path):
    """A package that is a copy of the source has nothing to report."""
    files = {
        "core/auth.py": "TOKEN = None\n",
        "core/__init__.py": "",
        "storage/database.py": "SCHEMA = 1\n",
    }
    source = _tree(tmp_path / "backend", files)
    packaged = _tree(tmp_path / "packaged", files)

    report = pbg.compare(source, packaged)

    assert report.ok is True
    assert report.compared == 3
    assert report.only_in_source == ()
    assert report.only_in_packaged == ()
    assert report.changed == ()


def test_a_one_byte_change_is_reported_with_both_hashes(tmp_path):
    """A package that differs from the source names the file and the hashes."""
    source = _tree(tmp_path / "backend", {"core/auth.py": "TOKEN = None\n"})
    packaged = _tree(tmp_path / "packaged", {"core/auth.py": "TOKEN = NoNe\n"})

    report = pbg.compare(source, packaged)

    assert not report.ok
    assert [e.path for e in report.changed] == ["core/auth.py"]
    changed = report.changed[0]
    assert changed.source_sha256 == pbg._sha256(source / "core" / "auth.py")
    assert changed.packaged_sha256 == pbg._sha256(packaged / "core" / "auth.py")
    assert changed.source_sha256 != changed.packaged_sha256


def test_a_file_added_to_the_package_is_reported_as_extra(tmp_path):
    """A file the source does not have is drift, even when nothing is missing."""
    source = _tree(tmp_path / "backend", {"core/auth.py": "TOKEN = None\n"})
    packaged = _tree(tmp_path / "packaged", {
        "core/auth.py": "TOKEN = None\n",
        "core/backdoor.py": "TOKEN = 'stolen'\n",
    })

    report = pbg.compare(source, packaged)

    assert not report.ok
    assert [e.path for e in report.only_in_packaged] == ["core/backdoor.py"]
    assert report.only_in_source == ()
    assert report.changed == ()


def test_a_changed_non_python_file_is_reported(tmp_path):
    """A changed .json or .pt inside the package is drift too."""
    source = _tree(tmp_path / "backend", {
        "core/auth.py": "TOKEN = None\n",
        "data/config.json": '{"a": 1}\n',
    })
    packaged = _tree(tmp_path / "packaged", {
        "core/auth.py": "TOKEN = None\n",
        "data/config.json": '{"a": 2}\n',
    })

    report = pbg.compare(source, packaged)

    assert not report.ok
    assert [e.path for e in report.changed] == ["data/config.json"]


def test_lock_file_noise_is_not_drift(tmp_path):
    """A lock file left behind by a tool run is not a difference."""
    source = _tree(tmp_path / "backend", {
        "core/auth.py": "TOKEN = None\n",
        "core/.dsh-ops-update-abc123.lock": "lock\n",
    })
    packaged = _tree(tmp_path / "packaged", {
        "core/auth.py": "TOKEN = None\n",
        "core/.dsh-ops-update-abc123.lock": "lock\n",
    })

    report = pbg.compare(source, packaged)

    assert report.ok is True


def test_compare_ignores_caches_and_runtime_state(tmp_path):
    """A ``.pyc`` or a database is not drift; it is the other failure."""
    source = _tree(tmp_path / "backend", {"core/auth.py": "TOKEN = None\n"})
    packaged = _tree(tmp_path / "packaged", {"core/auth.py": "TOKEN = None\n"})
    (packaged / "core" / "__pycache__").mkdir(parents=True)
    (packaged / "core" / "__pycache__" / "auth.cpython-311.pyc").write_bytes(b"\x00")
    (packaged / "storage").mkdir(parents=True)
    (packaged / "storage" / ".mech_api_token").write_text("secret\n", encoding="utf-8")
    (packaged / "storage" / "mech.db").write_bytes(b"\x00")

    report = pbg.compare(source, packaged)

    # The token and the database are not "drift": they are the separate
    # failure the secret scan looks for.
    assert [e.path for e in report.only_in_packaged] == [
        "storage/.mech_api_token", "storage/mech.db"]
    assert report.changed == ()


def test_stage_never_copies_a_token_or_a_database(tmp_path):
    """Staging copies the source and refuses to copy a secret in."""
    source = tmp_path / "backend"
    packaged = tmp_path / "packaged"
    _tree(source, {
        "core/auth.py": "TOKEN = None\n",
        "storage/evidence/r0.json": "{}\n",
        "storage/v2_jobs/j0.json": "{}\n",
        "storage/.mech_api_token": "dev-token\n",
        "runtime.db": "sqlite",
    })
    (source / "storage" / "mech.db").write_bytes(b"sqlite")
    # A packaged tree that already carries a token from an earlier build.
    _tree(packaged, {
        "core/auth.py": "TOKEN = None\n",
        "storage/.mech_api_token": "old-token\n",
        "storage/mech.db": "old sqlite",
    })

    result = pbg.stage(source, packaged)

    assert not (packaged / "storage" / ".mech_api_token").exists()
    assert not (packaged / "storage" / "mech.db").exists()
    assert not (packaged / "storage" / "evidence").exists()
    assert not (packaged / "storage" / "v2_jobs").exists()
    assert not (packaged / "runtime.db").exists()
    assert (packaged / "core" / "auth.py").is_file()
    assert "storage/.mech_api_token" in result.removed_secrets
    assert "storage/mech.db" in result.removed_secrets


def test_check_never_writes_into_the_packaged_tree(tmp_path, capsys):
    """`check` is strictly read-only: it must not repair what it checks."""
    source = _tree(tmp_path / "backend", {"core/auth.py": "TOKEN = None\n"})
    packaged = _tree(tmp_path / "packaged", {
        "core/auth.py": "TOKEN = NoNe\n",
        "storage/.mech_api_token": "secret\n",
    })
    before = _snapshot(packaged)

    status = pbg.main(["check", "--source", str(source), "--target", str(packaged)])

    after = _snapshot(packaged)
    assert status == 1, "drift must be reported as a failure"
    assert after == before, "check must not write into the packaged tree"
    assert "NOT CHECKED" not in capsys.readouterr().out


def test_check_reports_a_missing_artifact_as_not_checked(tmp_path, capsys):
    """An absent package is reported as not checked, and never as a pass."""
    source = _tree(tmp_path / "backend", {"core/auth.py": "TOKEN = None\n"})
    missing = tmp_path / "not-built" / "backend"

    status = pbg.main(["check", "--source", str(source), "--target", str(missing)])

    output = capsys.readouterr().out
    assert status == 2, "a missing packaged tree must not exit 0"
    assert "NOT CHECKED" in output
    assert "An absent tree is not a pass" in output


# ── the artifacts (only when there is one) ───────────────────────────────

@pytest.mark.parametrize("label,target", TARGETS, ids=TARGET_IDS)
def test_packaged_backend_matches_the_source(label, target):
    """A packaged backend that differs from the source is a stale package."""
    _skip_without_artifact(target)
    report = pbg.compare(pbg.SOURCE_BACKEND, target)
    problems = []
    if report.only_in_source:
        problems.append(
            "only in source (%d): %s"
            % (len(report.only_in_source),
               _sample(e.path for e in report.only_in_source)))
    if report.only_in_packaged:
        problems.append(
            "only in packaged (%d): %s"
            % (len(report.only_in_packaged),
               _sample(e.path for e in report.only_in_packaged)))
    if report.changed:
        problems.append(
            "changed (%d): %s"
            % (len(report.changed), _sample(e.path for e in report.changed)))
    assert report.compared > 0, (
        "the source set was empty, so nothing was compared; an empty "
        "comparison is not a pass")
    assert not problems, (
        "the packaged backend at %s has drifted from the source backend:\n  %s\n"
        "Run: python scripts/packaged_backend_guard.py stage"
        % (target, "\n  ".join(problems)))


@pytest.mark.parametrize("label,target", TARGETS, ids=TARGET_IDS)
def test_packaged_backend_carries_no_token_or_database(label, target):
    """A packaged backend must not carry the token, the database or the state."""
    _skip_without_artifact(target)
    offenders = pbg.find_forbidden(target, patterns=FORBIDDEN_IN_PACKAGE)
    assert not offenders, (
        "the packaged backend at %s carries %d file(s) that must never ship: %s"
        % (target, len(offenders), _sample(offenders)))
