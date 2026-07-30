import time
import json
from datetime import datetime
from typing import Dict, Any, List

class ContinuousBenchmarking:
    """
    Nightly/Continuous benchmarking system.
    Runs suites against models, detects regressions, and formats data for dashboards.
    """
    def __init__(self, history_dir: str = "backend/benchmarking/history"):
        self.history_dir = history_dir
        import os
        os.makedirs(self.history_dir, exist_ok=True)

    def run_suite(self, model_name: str, suite: callable) -> Dict[str, Any]:
        """
        Executes a benchmarking suite and logs the results.
        `suite` should be a callable that returns a Dict of benchmark scores.
        """
        print(f"[{datetime.utcnow().isoformat()}] Starting CI Benchmarks for {model_name}...")
        start_time = time.time()
        
        try:
             results = suite(model_name)
             status = "success"
        except Exception as e:
             results = {"error": str(e)}
             status = "failed"
             
        duration = time.time() - start_time
        
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": model_name,
            "status": status,
            "duration_sec": duration,
            "metrics": results
        }
        
        self._save_record(model_name, record)
        return record

    def _save_record(self, model_name: str, record: Dict[str, Any]):
        import os
        filepath = os.path.join(self.history_dir, f"{model_name}_history.json")
        
        history = []
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                 history = json.load(f)
                 
        history.append(record)
        
        with open(filepath, 'w') as f:
             json.dump(history, f, indent=2)

    def detect_regressions(self, model_name: str, threshold: float = 0.05) -> List[Dict[str, Any]]:
        """
        Compares the latest run against the historical moving average to detect performance drops.
        """
        import os
        filepath = os.path.join(self.history_dir, f"{model_name}_history.json")
        if not os.path.exists(filepath):
            return []
            
        with open(filepath, 'r') as f:
            history = json.load(f)
            
        if len(history) < 2:
            return []
            
        # Get latest successful run
        successful_runs = [r for r in history if r.get("status") == "success"]
        if len(successful_runs) < 2:
            return []
            
        latest = successful_runs[-1]
        
        # Calculate moving average of previous 5 runs
        prev_runs = successful_runs[-6:-1]
        regressions = []
        
        for metric, latest_val in latest.get("metrics", {}).items():
            if not isinstance(latest_val, (int, float)):
                 continue
                 
            # Aggregate past values for this metric
            past_vals = [r.get("metrics", {}).get(metric) for r in prev_runs]
            past_vals = [v for v in past_vals if isinstance(v, (int, float))]
            
            if not past_vals:
                continue
                
            avg = sum(past_vals) / len(past_vals)
            
            # If performance dropped by more than threshold
            # Note: Assumes higher is better. Adjust logic if lower is better for some metrics.
            if latest_val < avg * (1 - threshold):
                regressions.append({
                    "metric": metric,
                    "previous_average": avg,
                    "latest_value": latest_val,
                    "drop_percentage": ((avg - latest_val) / avg) * 100
                })
                
        return regressions
