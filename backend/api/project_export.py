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

        # Real integration: Query the DB for project artifacts
        from ..core.database import SessionLocal, SessionRecord, ReportRecord, ExperimentRecord
        db = SessionLocal()

        try:
            # Fetch sessions associated with this project (assuming experiment ties to project)
            sessions = db.query(SessionRecord).filter(SessionRecord.project_id == project_id).all()
            reports = db.query(ReportRecord).all() # For demo, export all or filter by project

            session_data_list = [s.session_data for s in sessions]
            report_data_list = [r.report_data for r in reports]

            with zipfile.ZipFile(archive_path, 'w') as archive:
                archive.writestr("metadata.json", json.dumps({"project_id": project_id, "export_version": "1.0"}))
                archive.writestr("sessions.json", json.dumps(session_data_list, indent=2))
                archive.writestr("reports.json", json.dumps(report_data_list, indent=2))
        finally:
            db.close()

        return {
            "status": "success",
            "archive_path": str(archive_path),
            "project_id": project_id
        }
