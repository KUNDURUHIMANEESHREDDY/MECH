from typing import List, Dict, Any
import json
import os

class PhaseGatekeeper:
    """
    Blocks advanced research phases until prerequisite landmarks are complete.
    """
    
    DEPENDENCIES = {
        "Novel Discovery": ["ioi_reproduction", "induction_heads", "sae_reproduction"],
        "Cross-Family Study": ["sae_reproduction", "cross_model_universality"],
        "Publication": ["statistical_validation", "peer_review_pass"]
    }

    def __init__(self, projects_root: str = "backend/reproductions/projects"):
        self.projects_root = projects_root

    def check_gate(self, phase_name: str) -> Dict[str, Any]:
        """
        Checks if a phase is blocked based on prerequisite completion scores.
        """
        prereqs = self.DEPENDENCIES.get(phase_name, [])
        blocks = []
        
        for p in prereqs:
            project_path = os.path.join(self.projects_root, p)
            cert_path = os.path.join(project_path, "completion_certificate.json")
            
            score = 0
            if os.path.exists(cert_path):
                 with open(cert_path, 'r') as f:
                     cert = json.load(f)
                 score = float(cert["completion"].strip('%'))
            
            if score < 95.0:
                 blocks.append({
                     "prerequisite": p,
                     "current_score": score,
                     "required_score": 95.0,
                     "reason": f"{p} completion only {score}%"
                 })
                 
        return {
            "phase": phase_name,
            "is_blocked": len(blocks) > 0,
            "blocking_reasons": blocks
        }
