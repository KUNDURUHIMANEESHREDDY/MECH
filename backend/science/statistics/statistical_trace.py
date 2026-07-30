import json
import hashlib
from typing import Dict, Any, List
from datetime import datetime

class StatisticalTrace:
    """
    Logs every statistical decision made during an analysis.
    """
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.trace: List[Dict[str, Any]] = []
        
    def add_step(self, step_name: str, details: Dict[str, Any]):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "step": step_name,
            "details": details
        }
        self.trace.append(entry)
        
    def generate_report(self) -> Dict[str, Any]:
        trace_str = json.dumps(self.trace, sort_keys=True)
        trace_hash = hashlib.sha256(trace_str.encode('utf-8')).hexdigest()
        
        return {
            "experiment_id": self.experiment_id,
            "trace_hash": trace_hash,
            "steps": self.trace
        }
        
    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.generate_report(), f, indent=2)
