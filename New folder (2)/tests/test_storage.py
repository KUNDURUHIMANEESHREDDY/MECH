from __future__ import annotations

from pathlib import Path

import pytest

from storage.database import DesktopStorage, StorageError


def test_settings_are_saved_to_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "desktop.sqlite3"
    storage = DesktopStorage(db_path)
    storage.initialize()

    saved = storage.update_settings(
        {
            "theme": "dark",
            "gpuEnabled": True,
            "cachePath": str(tmp_path / "cache"),
            "modelPath": "gpt2",
            "workspacePath": str(tmp_path),
        }
    )

    reloaded = DesktopStorage(db_path)
    reloaded.initialize()

    assert saved["theme"] == "dark"
    assert reloaded.get_settings()["gpuEnabled"] is True
    assert reloaded.get_settings()["cachePath"] == str(tmp_path / "cache")


def test_invalid_theme_is_rejected(tmp_path: Path) -> None:
    storage = DesktopStorage(tmp_path / "desktop.sqlite3")
    storage.initialize()

    with pytest.raises(StorageError):
        storage.update_settings({"theme": "high-contrast"})


def test_recent_projects_are_deduplicated_and_ordered(tmp_path: Path) -> None:
    storage = DesktopStorage(tmp_path / "desktop.sqlite3")
    storage.initialize()

    first = storage.add_recent_project(str(tmp_path / "one"), "One")
    storage.add_recent_project(str(tmp_path / "two"), "Two")
    updated = storage.add_recent_project(first["path"], "One Updated")

    projects = storage.list_recent_projects()

    assert len(projects) == 2
    assert projects[0]["name"] == "One Updated"
    assert projects[0]["id"] == updated["id"]


def test_recent_files_are_saved(tmp_path: Path) -> None:
    storage = DesktopStorage(tmp_path / "desktop.sqlite3")
    storage.initialize()

    saved = storage.add_recent_file(
        str(tmp_path / "model.py"),
        project_path=str(tmp_path),
    )

    assert saved["path"].endswith("model.py")
    assert storage.list_recent_files()[0]["projectPath"] == str(tmp_path)
