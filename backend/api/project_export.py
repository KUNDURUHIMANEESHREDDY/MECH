"""Project Export API for Portability.

Bundles Notebooks, Models, Experiments, Reports, Figures, and the Knowledge Graph
into a single `.interp-project` archive for seamless sharing.

Status: not yet reachable from any route. The guards below exist so that
``project_id`` and ``dest_dir`` are safe the moment someone wires this up --
both are caller-supplied and both reach the filesystem.
"""

from typing import Any, Dict
import json
import os
import re
import zipfile
from pathlib import Path

#: ``project_id`` becomes a filename, so it is a slug: no separators, no dots,
#: no whitespace. This is what stops ``../../..`` from escaping the export root.
PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class ProjectExporter:
    @staticmethod
    def _export_root() -> Path:
        """Directory exports are confined to.

        Defaults to ``./exports`` under the current working directory. Override
        with ``MECH_EXPORT_ROOT``.
        """
        configured = os.environ.get("MECH_EXPORT_ROOT", "").strip()
        root = Path(configured).expanduser() if configured else Path.cwd() / "exports"
        try:
            return root.resolve()
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"export root is not resolvable: {exc}") from exc

    def export_project(self, project_id: str, dest_dir: str = "./exports") -> Dict[str, Any]:
        """Creates a .interp-project zip archive containing all research artifacts.

        Args:
            project_id: Slug identifying the project. Becomes the archive filename.
            dest_dir: Destination directory. Must resolve inside the export root.

        Raises:
            ValueError: ``project_id`` is not a valid slug, or ``dest_dir``
                resolves outside the configured export root.
        """
        if not isinstance(project_id, str) or not PROJECT_ID_RE.match(project_id):
            raise ValueError(
                "project_id must be 1-64 chars of letters, digits, '_' or '-' "
                "and must start with a letter or digit"
            )

        root = self._export_root()
        try:
            destination = Path(dest_dir).expanduser().resolve()
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"dest_dir is not resolvable: {exc}") from exc

        # Resolve before comparing, so a ".." path that lands back inside the
        # root is fine and one that leaves is not.
        if destination != root and root not in destination.parents:
            raise ValueError(
                f"'{destination}' is outside the export root '{root}'"
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
        from backend.storage import DesktopStorage, get_default_db_path

        store = DesktopStorage(get_default_db_path())
        all_sessions = store.list_sessions()

        # A session is included only when it names this project. An
        # unattributable session is excluded rather than guessed at: an export
        # is the most portable artifact this codebase produces, so a wrong
        # inclusion is permanent once the archive leaves.
        session_data_list, unattributed = _sessions_for_project(
            all_sessions, project_id)

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
        with zipfile.ZipFile(archive_path, 'w') as archive:
            archive.writestr("metadata.json", json.dumps({
                "project_id": project_id,
                "export_version": "1.0",
                "sessions_included": len(session_data_list),
                "sessions_omitted_unattributed": unattributed,
                "reports_included": 0,
                "reports_omitted_reason": (
                    "The storage authority has no reports table. Reports were "
                    "never linked to a project -- there is no project_id column "
                    "to filter on -- so exporting them would leak every other "
                    "project's reports into this archive."
                ),
            }))
            archive.writestr("sessions.json", json.dumps(session_data_list, indent=2))

        return {
            "status": "success",
            "archive_path": str(archive_path),
            "project_id": project_id
        }


#: Payload keys a session may carry its project link under. `project_id` is
#: the canonical name; the others are read so a session written by an older
#: build is still attributable rather than silently dropped from an export.
_PROJECT_KEYS = ("project_id", "projectId", "project")


def _sessions_for_project(sessions, project_id: str):
    """Split sessions into those attributable to ``project_id`` and the rest."""
    included, unattributed = [], 0
    for session in sessions:
        if not isinstance(session, dict):
            unattributed += 1
            continue
        owner = next((session[key] for key in _PROJECT_KEYS
                      if isinstance(session.get(key), str)), None)
        if owner == project_id:
            included.append(session)
        else:
            unattributed += 1
    return included, unattributed
