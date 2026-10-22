"""Inventory of the SAE checkpoints that exist on disk.

The SAE Lab has to talk about what is actually present: which checkpoint
files live in the configured directory, how large they are and when they
were written. This module is that inventory and nothing else. It opens no
tensors, it fetches nothing, and it never reports a checkpoint that is not
a file sitting in the directory.

Two states are kept apart on purpose:

* a directory that does not exist is an **empty list** -- there is nothing
  there;
* a field the backend did not return is **unknown** -- the caller decides
  what to do with that. Neither is ever turned into a placeholder entry or
  a made-up name.

The directory is the one the API route uses: ``MECH_SAE_DIR`` when set,
otherwise ``backend/storage/sae_checkpoints``. This module never creates
it; a missing directory is a legitimate answer, not an error to fix up
behind the caller's back.
"""

from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

#: Environment variable that relocates the checkpoint directory.
DIR_ENV = "MECH_SAE_DIR"

#: Where checkpoints live when ``MECH_SAE_DIR`` is not set.
DEFAULT_CHECKPOINT_DIR = (
    Path(__file__).resolve().parents[2] / "storage" / "sae_checkpoints"
)

#: Only PyTorch checkpoint files count as checkpoints.
CHECKPOINT_SUFFIX = ".pt"

#: The API contract caps the listing; the cap is applied here so every
#: caller reports the same inventory.
MAX_LISTED = 50


@dataclass(frozen=True)
class LocalCheckpoint:
    """One real checkpoint file found in the directory."""

    name: str
    size_bytes: int
    modified_at: float
    modified_at_iso: str

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "size_bytes": self.size_bytes,
            "modified_at": self.modified_at,
            "modified_at_iso": self.modified_at_iso,
        }


def checkpoint_dir() -> Path:
    """The configured checkpoint directory, resolved.

    Never creates it: a missing directory yields an empty inventory rather
    than a directory that only exists because something listed it.
    """
    configured = os.environ.get(DIR_ENV, "").strip()
    target = Path(configured) if configured else DEFAULT_CHECKPOINT_DIR
    try:
        return target.resolve()
    except OSError:
        return target


def list_checkpoints(
    directory: Optional[Path] = None,
    limit: int = MAX_LISTED,
) -> List[LocalCheckpoint]:
    """Real ``.pt`` checkpoint files in ``directory``, newest first.

    A missing or unreadable directory is an empty list. Files that are not
    checkpoints (other suffixes, directories, broken symlinks) are left out
    rather than listed as placeholders.
    """
    root = Path(directory) if directory is not None else checkpoint_dir()
    try:
        entries = list(root.iterdir())
    except (OSError, ValueError):
        return []

    found: List[LocalCheckpoint] = []
    for entry in entries:
        if entry.suffix.lower() != CHECKPOINT_SUFFIX:
            continue
        try:
            if not entry.is_file():
                continue
            stat = entry.stat()
        except OSError:
            continue
        found.append(
            LocalCheckpoint(
                name=entry.name,
                size_bytes=int(stat.st_size),
                modified_at=float(stat.st_mtime),
                modified_at_iso=_dt.datetime.fromtimestamp(
                    stat.st_mtime, _dt.timezone.utc
                ).isoformat(),
            )
        )

    found.sort(key=lambda item: (item.modified_at, item.name), reverse=True)
    if limit is not None and limit >= 0:
        return found[:limit]
    return found


def checkpoint_names(
    directory: Optional[Path] = None,
    limit: int = MAX_LISTED,
) -> List[str]:
    """The checkpoint file names, newest first -- what the status route lists."""
    return [item.name for item in list_checkpoints(directory, limit)]


def _self_check() -> None:
    """Assert the inventory only ever reports files that are really there."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        assert list_checkpoints(root) == [], "an empty directory must list nothing"

        older = root / "older.pt"
        newer = root / "newer.pt"
        older.write_bytes(b"x")
        newer.write_bytes(b"yy")
        os.utime(older, (1_000_000, 1_000_000))
        os.utime(newer, (2_000_000, 2_000_000))
        (root / "notes.txt").write_text("not a checkpoint", encoding="utf-8")
        (root / "nested.pt").mkdir()

        names = checkpoint_names(root)
        assert names == ["newer.pt", "older.pt"], names

        listed = list_checkpoints(root)
        assert listed[0].name == "newer.pt"
        assert listed[0].size_bytes == 2
        assert listed[1].size_bytes == 1

        assert checkpoint_names(root, limit=1) == ["newer.pt"]

    missing = Path(__file__).resolve().parent / "no-such-checkpoint-dir"
    assert checkpoint_names(missing) == [], "a missing directory must list nothing"


if __name__ == "__main__":
    _self_check()
    print("local_checkpoints self-check passed")
    print(f"directory: {checkpoint_dir()}")
    for item in list_checkpoints()[:5]:
        print(f"  {item.name}  {item.size_bytes} bytes  {item.modified_at_iso}")
