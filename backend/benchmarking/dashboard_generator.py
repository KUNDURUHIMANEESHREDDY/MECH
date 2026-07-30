import json
import os
from typing import Dict, Any, List

class DashboardGenerator:
    """
    Generates dashboard-ready JSON summaries from continuous benchmarking history.
    """
    def __init__(self, history_dir: str = "backend/benchmarking/history"):
        self.history_dir = history_dir

    def generate_summary(self) -> Dict[str, Any]:
        """
        Aggregates the latest status of all tracked models for a dashboard.
        """
        dashboard_data = {
            "models": [],
            "recent_regressions": [],
            "global_status": "healthy"
        }
        
        if not os.path.exists(self.history_dir):
            return dashboard_data
            
        from .continuous_benchmarking import ContinuousBenchmarking
        cb = ContinuousBenchmarking(self.history_dir)
        
        for filename in os.listdir(self.history_dir):
            if not filename.endswith("_history.json"):
                continue
                
            model_name = filename.replace("_history.json", "")
            filepath = os.path.join(self.history_dir, filename)
            
            with open(filepath, 'r') as f:
                history = json.load(f)
                
            if not history:
                continue
                
            latest = history[-1]
            regressions = cb.detect_regressions(model_name)
            
            model_summary = {
                "name": model_name,
                "latest_run": latest.get("timestamp"),
                "status": latest.get("status"),
                "metrics": latest.get("metrics", {}),
                "regression_count": len(regressions)
            }
            
            dashboard_data["models"].append(model_summary)
            
            for reg in regressions:
                dashboard_data["recent_regressions"].append({
                    "model": model_name,
                    "metric": reg["metric"],
                    "drop": f"{reg['drop_percentage']:.1f}%"
                })
                
            if latest.get("status") != "success" or regressions:
                 dashboard_data["global_status"] = "warning"
                 
        return dashboard_data
