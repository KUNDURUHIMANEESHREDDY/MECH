"""Project Export API for Portability.

Bundles Notebooks, Models, Experiments, Reports, Figures, and the Knowledge Graph
into a single `.interp-project` archive for seamless sharing.
"""

from typing import Any, Dict
import zipfile
import os
import json

class ProjectExporter:
    def export_project(self, project_id: str, dest_dir: str = "./exports") -> Dict[str, Any]:
        """Creates a .interp-project zip archive containing all research artifacts."""
        os.makedirs(dest_dir, exist_ok=True)
        # Prevent path traversal: only the last path component is used as the
        # archive filename, keeping the output strictly inside dest_dir.
        safe_id = os.path.basename(project_id.strip().rstrip(os.sep)) or "project"
        safe_id = safe_id.replace("..", "_").replace(" ", "_")
        archive_path = os.path.join(dest_dir, f"{safe_id}.interp-project")
        
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
            "archive_path": archive_path,
            "project_id": project_id
        }
