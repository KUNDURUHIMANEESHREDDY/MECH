"""SQLite-backed desktop application storage."""

from storage.database import DesktopStorage, StorageError

__all__ = ["DesktopStorage", "StorageError"]
