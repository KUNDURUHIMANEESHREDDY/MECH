import os
import json
import shutil
from datetime import datetime
from typing import Dict, Any, List

class CompletionHistoryManager:
    """
    Manages versioned completion reports for tracking scientific progress over time.
    """
    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.history_dir = os.path.join(project_dir, ".completion_history")
        os.makedirs(self.history_dir, exist_ok=True)

    def archive_current_version(self, version_tag: str = None):
        """
        Archives the current completion_certificate.json to history.
        """
        cert_src = os.path.join(self.project_dir, "completion_certificate.json")
        if not os.path.exists(cert_src):
            return
            
        if not version_tag:
            # Generate auto-tag based on timestamp
            version_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
        dest_path = os.path.join(self.history_dir, f"certificate_{version_tag}.json")
        shutil.copy2(cert_src, dest_path)

    def get_history(self) -> List[Dict[str, Any]]:
        history = []
        for filename in sorted(os.listdir(self.history_dir)):
            if filename.endswith(".json"):
                with open(os.path.join(self.history_dir, filename), 'r') as f:
                    history.append(json.load(f))
        return history
