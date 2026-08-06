"""SQLite-backed desktop application storage."""

from .database import DesktopStorage, StorageError

__all__ = ["DesktopStorage", "StorageError"]
