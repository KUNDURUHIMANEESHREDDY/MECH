import os
import json
from typing import Dict, Any, List
from datetime import datetime

class ResearchPortal:
    """
    The 'GitHub for Experiments' backend.
    Tracks the lifecycle of all research projects: Queued, Running, Peer Review, Published, Rejected.
    """
    def __init__(self, portal_root: str = "backend/science/portal"):
        self.portal_root = portal_root
        self.projects_dir = os.path.join(portal_root, "projects")
        self.discoveries_dir = os.path.join(portal_root, "discoveries")
        os.makedirs(self.projects_dir, exist_ok=True)
        os.makedirs(self.discoveries_dir, exist_ok=True)

    def get_portal_summary(self) -> Dict[str, Any]:
        """
        Returns a high-level summary of the research portal state.
        """
        summary = {
            "projects": {
                "Completed": 0,
                "Running": 0,
                "Queued": 0,
                "Rejected": 0
            },
            "peer_review": {
                "Published": 0,
                "Pending": 0
            },
            "novel_discoveries": 0,
            "project_list": []
        }
        
        for project_id in os.listdir(self.projects_dir):
            path = os.path.join(self.projects_dir, project_id, "status.json")
            if os.path.exists(path):
                with open(path, 'r') as f:
                    status = json.load(f)
                
                state = status.get("state", "Queued")
                if state in summary["projects"]:
                    summary["projects"][state] += 1
                
                review = status.get("peer_review", "None")
                if review == "Published":
                    summary["peer_review"]["Published"] += 1
                elif review == "Pending":
                    summary["peer_review"]["Pending"] += 1
                
                summary["project_list"].append({
                    "id": project_id,
                    "name": status.get("name"),
                    "state": state,
                    "review": review,
                    "updated_at": status.get("updated_at")
                })
        
        summary["novel_discoveries"] = len(os.listdir(self.discoveries_dir))
        return summary

    def update_project_status(self, project_id: str, state: str, name: str = None, review: str = None):
        """
        Updates or creates a project status record.
        """
        project_path = os.path.join(self.projects_dir, project_id)
        os.makedirs(project_path, exist_ok=True)
        
        status_path = os.path.join(project_path, "status.json")
        status = {}
        if os.path.exists(status_path):
            with open(status_path, 'r') as f:
                status = json.load(f)
        
        status.update({
            "id": project_id,
            "name": name or status.get("name", project_id),
            "state": state,
            "updated_at": datetime.utcnow().isoformat()
        })
        
        if review:
            status["peer_review"] = review
            
        with open(status_path, 'w') as f:
            json.dump(status, f, indent=2)

    def register_discovery(self, discovery_id: str, details: Dict[str, Any]):
        """
        Registers a new novel discovery in the portal.
        """
        path = os.path.join(self.discoveries_dir, f"{discovery_id}.json")
        with open(path, 'w') as f:
            json.dump({
                "id": discovery_id,
                "timestamp": datetime.utcnow().isoformat(),
                **details
            }, f, indent=2)
