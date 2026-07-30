import json
import hashlib
from datetime import datetime
from typing import Dict, Any

class CompletionCertificateGenerator:
    """
    Generates a signed, verifiable completion certificate for a research project.
    """

    @staticmethod
    def generate(project_name: str, validator_report: Dict[str, Any], project_dir: str) -> Dict[str, Any]:
        """
        Creates a JSON certificate summarizing project completion status.
        """
        timestamp = datetime.utcnow().isoformat()
        
        # Load core manifests if available for linking
        manifest_hash = "unknown"
        try:
             with open(f"{project_dir}/research_manifest.json", 'rb') as f:
                 manifest_hash = hashlib.sha256(f.read()).hexdigest()
        except: pass

        certificate = {
            "project_name": project_name,
            "completion": f"{validator_report['completion_score']}%",
            "is_complete": validator_report["is_complete"],
            "validator_version": "2.1",
            "timestamp": timestamp,
            "research_manifest_hash": manifest_hash,
            "satisfied_milestones": validator_report["satisfied"],
            "peer_review_status": "PASS" if "Peer review PASS" in validator_report["satisfied"] else "PENDING"
        }

        # Generate signature (simplified hash of certificate content)
        cert_str = json.dumps(certificate, sort_keys=True)
        certificate["signature"] = hashlib.sha256(cert_str.encode('utf-8')).hexdigest()
        
        return certificate

    @classmethod
    def save(cls, project_name: str, validator_report: Dict[str, Any], project_dir: str):
        cert = cls.generate(project_name, validator_report, project_dir)
        path = f"{project_dir}/completion_certificate.json"
        with open(path, 'w') as f:
            json.dump(cert, f, indent=2)
        return path
