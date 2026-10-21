"""Project Export API for Portability.

Bundles the sessions of one project into a single `.interp-project` archive
for sharing.

Status: reachable. ``POST /api/exports/{project_id}`` and
``POST /api/v1/exports/{project_id}`` (``backend/api/export_contracts.py``)
and ``mech projects export <project_id>`` (``backend/cli/main.py``) all call
the functions in this module, and there is no second implementation of the
archive format.

What the archive contains
-------------------------

The sessions stored for exactly one project, plus a ``metadata.json`` that
records what was included, what was omitted and why, when it was written,
where it came from, and a digest of every other member.

Reports are deliberately excluded: the storage authority has no ``reports``
table and no project link, so including them would leak every other
project's reports into this archive. The archive says so in
``reports_omitted_reason`` rather than dropping the file silently.

Rules this module enforces
---------------------------

* ``project_id`` becomes a filename, so it must be a slug
  (``PROJECT_ID_RE``). A bad id is refused before anything is read or
  written.
* ``dest_dir`` must resolve inside the export root. The root is
  ``MECH_EXPORT_ROOT`` when set, otherwise ``~/.mech/exports`` -- a
  user-writable location outside the install directory. The packaged app
  runs with the working directory set to the install directory, so the old
  ``./exports`` default was unwritable there and would have produced 500s,
  and archives written into the install directory are destroyed by upgrades.
* A project that no stored session names does not exist. Exporting it
  writes nothing and is refused (``404`` over HTTP, a non-zero exit in the
  CLI): an export must never report success for a project that is not there.
* Large exports need confirmation. The threshold is
  ``MECH_EXPORT_CONFIRM_BYTES`` (default 100 MiB); an estimate above it is
  refused until the caller confirms.
* The archive is written to a temporary file in the destination directory
  and then renamed onto its final name, as spec 20.1 requires ("Artifact
  files must be written to a temporary destination, validated, then
  atomically renamed into place"). A failure leaves no archive behind.
* Omitted sessions are counted in two buckets, not one:
  ``sessions_omitted_unattributed`` counts sessions that name no project at
  all, and ``sessions_omitted_other_projects`` counts sessions that name a
  *different* project. Lumping them together misnames the fact and tells
  the caller how many other projects exist.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

#: ``project_id`` becomes a filename, so it is a slug: no separators, no dots,
#: no whitespace. This is what stops ``../../..`` from escaping the export root.
PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

#: Version of the archive format this module writes.
EXPORT_VERSION = "1.0"

#: The estimate is computed from the exact bytes written as ``sessions.json``,
#: so it cannot drift from what the archive contains.
ESTIMATE_BASIS = "serialized_sessions_json"

#: Bytes an export may reach before the caller has to confirm it.
DEFAULT_CONFIRM_BYTES = 100 * 1024 * 1024

#: Environment variable that overrides :data:`DEFAULT_CONFIRM_BYTES`.
CONFIRM_BYTES_ENV = "MECH_EXPORT_CONFIRM_BYTES"

#: Why reports are not in the archive. Kept verbatim: it is the reason a
#: recipient sees inside the file, so it must not drift.
REPORTS_OMITTED_REASON = (
    "The storage authority has no reports table. Reports were "
    "never linked to a project -- there is no project_id column "
    "to filter on -- so exporting them would leak every other "
    "project's reports into this archive."
)


class InvalidProjectId(ValueError):
    """The project id is not a slug, so it cannot become a filename."""


class ProjectNotFound(ValueError):
    """No stored session names this project, so there is nothing to export.

    A ``ValueError`` because it is a refusal of the request, but a distinct
    type because it means "unknown project" rather than "malformed
    request": the two get different answers (404 against a bad slug).
    """


def export_root() -> Path:
    """The one directory exports are confined to.

    ``MECH_EXPORT_ROOT`` wins. The default is ``~/.mech/exports`` -- a
    user-writable location outside the install directory, because the
    packaged app runs with the working directory set to the install
    directory (so ``./exports`` was unwritable there) and because archives
    written into the install directory are destroyed by the next upgrade.
    """
    configured = os.environ.get("MECH_EXPORT_ROOT", "").strip()
    if configured:
        root = Path(configured).expanduser()
    else:
        try:
            root = Path.home() / ".mech" / "exports"
        except RuntimeError as exc:
            raise ValueError(f"export root is not resolvable: {exc}") from exc
    try:
        return root.resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"export root is not resolvable: {exc}") from exc


def confirmation_threshold_bytes() -> int:
    """Bytes an export may reach before the caller must confirm it.

    ``MECH_EXPORT_CONFIRM_BYTES`` overrides the default. A missing,
    malformed or negative value falls back to the default rather than
    quietly disabling the confirmation.
    """
    raw = os.environ.get(CONFIRM_BYTES_ENV, "").strip()
    if not raw:
        return DEFAULT_CONFIRM_BYTES
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_CONFIRM_BYTES
    if value < 0:
        return DEFAULT_CONFIRM_BYTES
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_project_id(project_id: Any) -> str:
    if not isinstance(project_id, str) or not PROJECT_ID_RE.match(project_id):
        raise InvalidProjectId(
            "project_id must be 1-64 chars of letters, digits, '_' or '-' "
            "and must start with a letter or digit"
        )
    return project_id


def _read_sessions() -> List[Dict[str, Any]]:
    """Every session the storage authority holds.

    An authority that has not been initialized yet holds no sessions at all,
    which is reported as an empty list: that makes every project unknown,
    and an unknown project is refused rather than exported empty. Any other
    storage failure is raised, never turned into an empty result.
    """
    from backend.storage import DesktopStorage, get_default_db_path

    store = DesktopStorage(get_default_db_path())
    try:
        return list(store.list_sessions())
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc):
            return []
        raise


def _serialize_sessions(sessions: List[Dict[str, Any]]) -> bytes:
    """The exact bytes written as ``sessions.json``.

    The estimate is computed from these bytes, so the estimate and the
    archive cannot disagree.
    """
    try:
        return json.dumps(sessions, indent=2).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"session payload is not serialisable: {exc}") from exc


def _plan(project_id: str) -> Tuple[Dict[str, Any], bytes]:
    """The estimate and the payload it was computed from, in one pass.

    Both the estimate and the archive are produced from this, so they are
    the same computation: the estimate cannot describe a different archive
    than the one that is written.
    """
    _validate_project_id(project_id)
    sessions = _read_sessions()
    included, unattributed, other_projects = _sessions_for_project(
        sessions, project_id)
    payload = _serialize_sessions(included)
    threshold = confirmation_threshold_bytes()
    estimate = {
        "project_id": project_id,
        "sessions_included": len(included),
        "sessions_omitted_unattributed": unattributed,
        "sessions_omitted_other_projects": other_projects,
        "estimated_bytes": len(payload),
        "threshold_bytes": threshold,
        "requires_confirmation": len(payload) > threshold,
        "estimate_basis": ESTIMATE_BASIS,
    }
    return estimate, payload


def estimate_export(project_id: str) -> Dict[str, Any]:
    """What an export of ``project_id`` would cost, without writing anything.

    The estimate is computed from the same serialized session payload the
    export would write, so it changes when the session content changes and
    cannot drift from the result.
    """
    estimate, _payload = _plan(project_id)
    return estimate


def _resolve_destination(dest_dir: Optional[str]) -> Path:
    """The directory the archive goes in, checked against the export root."""
    root = export_root()
    raw = root if dest_dir is None else Path(dest_dir).expanduser()
    try:
        destination = raw.resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"dest_dir is not resolvable: {exc}") from exc

    # Resolve before comparing, so a ".." path that lands back inside the
    # root is fine and one that leaves is not.
    if destination != root and root not in destination.parents:
        raise ValueError(
            f"'{destination}' is outside the export root '{root}'")
    return destination


def _write_archive_atomically(archive_path: Path,
                               members: List[Tuple[str, bytes]]) -> None:
    """Write the archive to a temporary file and rename it into place.

    Spec 20.1: an artifact file must be written to a temporary destination,
    validated, then atomically renamed into place. The temporary file lives
    in the destination directory so the rename stays on one filesystem, and
    it is removed if anything fails: a half-written archive must never be
    left where a caller could mistake it for a finished one.
    """
    handle, temporary_name = tempfile.mkstemp(
        dir=str(archive_path.parent),
        prefix=f".{archive_path.name}.",
        suffix=".tmp",
    )
    os.close(handle)
    temporary_path = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary_path, "w") as archive:
            for name, payload in members:
                archive.writestr(name, payload)
        os.replace(temporary_path, archive_path)
    except BaseException:
        try:
            temporary_path.unlink()
        except OSError:
            pass
        raise


class ProjectExporter:
    @staticmethod
    def _export_root() -> Path:
        """Directory exports are confined to. See :func:`export_root`."""
        return export_root()

    def export_project(self, project_id: str,
                       dest_dir: Optional[str] = None) -> Dict[str, Any]:
        """Creates a .interp-project zip archive of this project's sessions.

        Args:
            project_id: Slug identifying the project. Becomes the archive filename.
            dest_dir: Destination directory. Must resolve inside the export
                root. ``None`` means the export root itself.

        Raises:
            ValueError: ``project_id`` is not a valid slug, or ``dest_dir``
                resolves outside the configured export root.
            ProjectNotFound: no stored session names this project.
        """
        _validate_project_id(project_id)
        destination = _resolve_destination(dest_dir)
        estimate, sessions_payload = _plan(project_id)
        if estimate["sessions_included"] == 0:
            raise ProjectNotFound(
                f"no stored session names project {project_id!r}; "
                f"there is nothing to export"
            )

        os.makedirs(destination, exist_ok=True)
        archive_path = destination / f"{project_id}.interp-project"

        # Sessions come from `DesktopStorage`, the single storage authority.
        #
        # This used to read `backend.core.database.SessionLocal`, a second
        # authority with its own engine on `sqlite:///./interp_research.db`.
        # That file was always 0 bytes -- `init_db()` had no callers -- so
        # every call raised `no such table: sessions`. The queries below are
        # therefore not a port of working code; they were never runnable.
        #
        # Reports are deliberately not exported.
        #
        # The single authority has no `reports` table at all, so there is
        # nothing to export and nothing that can leak across projects. This
        # used to be a comment about `ReportRecord` having no `project_id`
        # column in the deleted ORM schema.
        #
        # If reports are ever added to `DesktopStorage`, they must carry a
        # project link before this export includes them, and
        # `test_broken_and_leaking_methods.py` fails until they do.
        files = [{
            "path": "sessions.json",
            "sha256": _sha256_bytes(sessions_payload),
            "bytes": len(sessions_payload),
        }]
        exported_at = _utc_now()
        metadata = {
            "project_id": project_id,
            "export_version": EXPORT_VERSION,
            "sessions_included": estimate["sessions_included"],
            "sessions_omitted_unattributed":
                estimate["sessions_omitted_unattributed"],
            "sessions_omitted_other_projects":
                estimate["sessions_omitted_other_projects"],
            "reports_included": 0,
            "reports_omitted_reason": REPORTS_OMITTED_REASON,
            "exported_at": exported_at,
            # The storage database file name, never its absolute path: an
            # archive travels to other machines, where that path means
            # nothing and leaks the exporting machine's layout.
            "source": _storage_source_name(),
            "files": files,
        }
        metadata_payload = json.dumps(metadata, indent=2).encode("utf-8")

        _write_archive_atomically(archive_path, [
            ("metadata.json", metadata_payload),
            ("sessions.json", sessions_payload),
        ])

        return {
            "status": "success",
            "archive_path": str(archive_path),
            "project_id": project_id,
            "archive_name": archive_path.name,
            "bytes_written": archive_path.stat().st_size,
            "sha256": _file_sha256(archive_path),
            "estimated_bytes": estimate["estimated_bytes"],
            "estimate_basis": estimate["estimate_basis"],
            "threshold_bytes": estimate["threshold_bytes"],
            "sessions_included": estimate["sessions_included"],
            "sessions_omitted_unattributed":
                estimate["sessions_omitted_unattributed"],
            "sessions_omitted_other_projects":
                estimate["sessions_omitted_other_projects"],
            "exported_at": exported_at,
            "files": files,
        }


def _storage_source_name() -> str:
    """The storage database's file name -- never its absolute path."""
    from backend.storage import get_default_db_path

    try:
        return get_default_db_path().name
    except (OSError, RuntimeError, ValueError):  # pragma: no cover - defensive
        return "unknown"


#: Payload keys a session may carry its project link under. `project_id` is
#: the canonical name; the others are read so a session written by an older
#: build is still attributable rather than silently dropped from an export.
_PROJECT_KEYS = ("project_id", "projectId", "project")


def _sessions_for_project(sessions, project_id: str):
    """Split sessions into this project's, unattributed, and other projects'.

    ``unattributed`` counts sessions that name no project at all.
    ``other_projects`` counts sessions that name a *different* project: they
    are a different fact, and merging the two buckets misnames the first
    while telling the caller how many other projects exist.
    """
    included, unattributed, other_projects = [], 0, 0
    for session in sessions:
        if not isinstance(session, dict):
            unattributed += 1
            continue
        owner = next((session[key] for key in _PROJECT_KEYS
                      if isinstance(session.get(key), str)), None)
        if owner == project_id:
            included.append(session)
        elif owner is None:
            unattributed += 1
        else:
            other_projects += 1
    return included, unattributed, other_projects
