import os
import json
from typing import Dict, Any, List

class DashboardDataGenerator:
    """
    Aggregates completion reports for the UI dashboard.
    """
    def __init__(self, projects_root: str = "backend/reproductions/projects"):
        self.projects_root = projects_root

    def generate_summary(self) -> Dict[str, Any]:
        summary = {
            "landmark_papers": [],
            "discovery_projects": [],
            "overall_health": "stable"
        }
        
        if not os.path.exists(self.projects_root):
             return summary
             
        for project_id in os.listdir(self.projects_root):
            project_path = os.path.join(self.projects_root, project_id)
            if not os.path.isdir(project_path):
                continue
                
            cert_path = os.path.join(project_path, "completion_certificate.json")
            if os.path.exists(cert_path):
                with open(cert_path, 'r') as f:
                    cert = json.load(f)
                
                # Simplified status color mapping
                score = float(cert["completion"].strip('%'))
                status = "🟢" if score >= 95 else "🟡" if score >= 50 else "🔴"
                
                entry = {
                    "id": project_id,
                    "name": cert["project_name"],
                    "completion": cert["completion"],
                    "status_icon": status,
                    "is_complete": cert["is_complete"],
                    "timestamp": cert["timestamp"]
                }
                
                # Categorize (heuristic)
                if any(x in project_id.lower() for x in ["ioi", "induction", "sae", "acdc"]):
                    summary["landmark_papers"].append(entry)
                else:
                    summary["discovery_projects"].append(entry)
                    
        return summary
