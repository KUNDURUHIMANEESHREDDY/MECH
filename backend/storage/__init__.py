"""SQLite-backed desktop application storage.

This is the project's single storage authority. `backend/core/database.py`
used to be a second one -- its own engine on a CWD-relative
`sqlite:///./interp_research.db`, tables named `experiments` and `sessions`
that shared no column with the live ones here, and an `init_db()` with zero
callers, so the file it opened was always empty with no schema.
"""

from .database import DesktopStorage, StorageError, get_default_db_path

__all__ = ["DesktopStorage", "StorageError", "get_default_db_path"]
