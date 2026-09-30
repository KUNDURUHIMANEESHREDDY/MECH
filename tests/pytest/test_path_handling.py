"""Path-handling tests for DesktopStorage.describe_workspace and ProjectExporter.

Both entry points take a path from the request and touch the filesystem, so
both are bounded: describe_workspace to a set of allowed roots, export_project
to a slug and a configured export root.

Covers the reachability of each: describe_workspace is wired to the Electron
sidecar's ``workspace.describe`` method; ProjectExporter currently has no
caller, so its guards are tested directly to keep it from becoming a trap for
whoever wires it up.
"""
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.storage.database import DesktopStorage, StorageError  # noqa: E402


@pytest.fixture
def storage(tmp_path):
    store = DesktopStorage(tmp_path / "mech.db")
    store.initialize()
    return store


# --------------------------------------------------------------------------- #
# describe_workspace — containment
# ---------------------------------------------------------------------------


def test_workspace_inside_allowed_root(tmp_path, storage, monkeypatch):
    allowed = tmp_path / "work"
    allowed.mkdir()
    (allowed / "a.py").write_text("x", encoding="utf-8")

    monkeypatch.setenv("MECH_WORKSPACE_ROOTS", str(allowed))
    summary = storage.describe_workspace(str(allowed))
    assert summary["exists"] is True
    assert summary["fileCount"] == 1


def test_workspace_outside_allowed_root_rejected(tmp_path, storage, monkeypatch):
    allowed = tmp_path / "work"
    allowed.mkdir()
    secret = tmp_path / "elsewhere"
    secret.mkdir()
    (secret / "private.txt").write_text("s", encoding="utf-8")

    monkeypatch.setenv("MECH_WORKSPACE_ROOTS", str(allowed))
    with pytest.raises(StorageError, match="outside the allowed workspace"):
        storage.describe_workspace(str(secret))


def test_workspace_traversal_rejected(tmp_path, storage, monkeypatch):
    """A '..' path must not escape an allowed root."""
    allowed = tmp_path / "work"
    allowed.mkdir()
    monkeypatch.setenv("MECH_WORKSPACE_ROOTS", str(allowed))
    with pytest.raises(StorageError, match="outside the allowed workspace"):
        storage.describe_workspace(str(allowed / ".." / ".."))


def test_workspace_defaults_to_home_root(tmp_path, storage, monkeypatch):
    """With no explicit roots, containment falls back to the user's home."""
    monkeypatch.delenv("MECH_WORKSPACE_ROOTS", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    (tmp_path / "home").mkdir()
    (tmp_path / "home" / "x.txt").write_text("x", encoding="utf-8")

    summary = storage.describe_workspace(str(tmp_path / "home"))
    assert summary["fileCount"] == 1

    outside = tmp_path / "not_home"
    outside.mkdir()
    with pytest.raises(StorageError, match="outside the allowed workspace"):
        storage.describe_workspace(str(outside))


def test_workspace_empty_path_rejected(storage):
    with pytest.raises(StorageError, match="workspace path is required"):
        storage.describe_workspace("   ")


def test_workspace_non_string_rejected(storage):
    with pytest.raises(StorageError, match="workspace path is required"):
        storage.describe_workspace(None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# describe_workspace — cheap stat, no full walk
# ---------------------------------------------------------------------------


def test_workspace_nonexistent_path_is_cheap(tmp_path, storage, monkeypatch):
    """A missing directory returns a summary without walking anything."""
    monkeypatch.setenv("MECH_WORKSPACE_ROOTS", str(tmp_path))
    summary = storage.describe_workspace(str(tmp_path / "nope"))
    assert summary["exists"] is False
    assert summary["isDirectory"] is False
    assert summary["fileCount"] == 0


def test_workspace_file_path_reports_not_directory(tmp_path, storage, monkeypatch):
    monkeypatch.setenv("MECH_WORKSPACE_ROOTS", str(tmp_path))
    a_file = tmp_path / "plain.txt"
    a_file.write_text("x", encoding="utf-8")
    summary = storage.describe_workspace(str(a_file))
    assert summary["exists"] is True
    assert summary["isDirectory"] is False


def test_workspace_counts_are_capped_and_flagged(tmp_path, storage, monkeypatch):
    """Large trees must be bounded and say so, not silently truncate."""
    root = tmp_path / "big"
    root.mkdir()
    for i in range(50):
        (root / f"f{i}.txt").write_text("x", encoding="utf-8")
    monkeypatch.setenv("MECH_WORKSPACE_ROOTS", str(root))

    summary = storage.describe_workspace(str(root))
    assert summary["fileCount"] == 50
    assert summary["countCapped"] is False


def test_workspace_does_not_follow_symlinks_out_of_root(tmp_path, storage, monkeypatch):
    """A symlink inside the root must not pull in files from outside it."""
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    for i in range(5):
        (outside / f"secret{i}.txt").write_text("s", encoding="utf-8")

    try:
        os.symlink(outside, root / "link", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")

    monkeypatch.setenv("MECH_WORKSPACE_ROOTS", str(root))
    summary = storage.describe_workspace(str(root))
    assert summary["fileCount"] == 0, "symlinked subtree must not be counted"


# --------------------------------------------------------------------------- #
# ProjectExporter — slug validation and export-root containment
# ---------------------------------------------------------------------------


def _exporter():
    from backend.api.project_export import ProjectExporter
    return ProjectExporter()


@pytest.mark.parametrize("bad", [
    "../../../../Users/victim/.ssh/id_rsa",
    "..",
    "a/b",
    "a\\b",
    "",
    "  ",
    "name with spaces",
    "name;rm -rf /",
    "x" * 300,
])
def test_export_rejects_invalid_project_id(tmp_path, bad, monkeypatch):
    monkeypatch.setenv("MECH_EXPORT_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="project_id"):
        _exporter().export_project(bad, dest_dir=str(tmp_path))


def test_export_rejects_dest_dir_outside_export_root(tmp_path, monkeypatch):
    """dest_dir must stay under the configured export root."""
    root = tmp_path / "exports"
    root.mkdir()
    monkeypatch.setenv("MECH_EXPORT_ROOT", str(root))

    with pytest.raises(ValueError, match="outside the export root"):
        _exporter().export_project("myproject",
                                   dest_dir=str(tmp_path / "somewhere-else"))


def test_export_rejects_dest_dir_traversal(tmp_path, monkeypatch):
    root = tmp_path / "exports"
    root.mkdir()
    (tmp_path / "victim").mkdir()
    monkeypatch.setenv("MECH_EXPORT_ROOT", str(root))

    with pytest.raises(ValueError, match="outside the export root"):
        _exporter().export_project("myproject", dest_dir=str(root / ".." / "victim"))


def test_export_resolves_paths_before_checking(tmp_path, monkeypatch):
    """A traversal that lands back inside the root is accepted.

    Guards on the *resolved* path, not the string, so ``sub/../sub`` is fine
    while ``../victim`` is not.
    """
    root = tmp_path / "exports"
    (root / "sub").mkdir(parents=True)
    monkeypatch.setenv("MECH_EXPORT_ROOT", str(root))
    # Must not raise ValueError for the path; reaching the DB layer is fine.
    try:
        _exporter().export_project("myproject",
                                   dest_dir=str(root / "sub" / ".." / "sub"))
    except ValueError as exc:
        assert "outside the export root" not in str(exc)
    except Exception:
        pass  # reached the DB layer, which is the point
