"""Scientific Session Recorder.

Creates an immutable, verifiable ledger of an experiment's execution context.
Records the exact prompts, model versions, datasets, hardware state, and 
hyperparameters so that any experiment can be deterministically replayed.
"""

from typing import Any, Dict, List, Optional
import datetime
import hashlib
import json
import uuid

class ScientificSessionRecorder:
    """Records the exact state of an experiment for reproducibility."""

    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.start_time = datetime.datetime.now(datetime.timezone.utc)
        self.events: List[Dict[str, Any]] = []
        self._environment_cache: Optional[Dict[str, Any]] = None

    def get_environment(self) -> Dict[str, Any]:
        if self._environment_cache is not None:
            return self._environment_cache
            
        import platform
        import sys
        
        env = {
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "cuda_available": False,
            "cuda_version": None,
            "torch_version": None,
            "transformers_version": None
        }
        
        try:
            import torch
            env["torch_version"] = torch.__version__
            if torch.cuda.is_available():
                env["cuda_available"] = True
                env["cuda_version"] = torch.version.cuda
        except ImportError:
            pass

        try:
            import transformers
            env["transformers_version"] = transformers.__version__
        except ImportError:
            pass
            
        self._environment_cache = env
        return env

    def record_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Records an action in the experiment timeline."""
        self.events.append({
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "type": event_type,
            "details": details
        })

    def finalize_session(
        self, 
        project_id: str, 
        model_id: str, 
        dataset_id: str, 
        seed: int,
        hyperparameters: Dict[str, Any],
        final_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Finalizes the ledger and computes a verification hash."""
        end_time = datetime.datetime.now(datetime.timezone.utc)
        
        session_data = {
            "session_id": self.session_id,
            "project_id": project_id,
            "model_id": model_id,
            "dataset_id": dataset_id,
            "random_seed": seed,
            "hyperparameters": hyperparameters,
            "environment": self.get_environment(),
            "timeline": self.events,
            "final_metrics": final_metrics,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat()
        }
        
        # Create a deterministic hash of the session
        json_str = json.dumps(session_data, sort_keys=True)
        session_hash = hashlib.sha256(json_str.encode('utf-8')).hexdigest()
        session_data["verification_hash"] = session_hash
        
        return session_data
