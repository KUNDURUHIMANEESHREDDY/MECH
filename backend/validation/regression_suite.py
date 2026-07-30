import json
import os
from typing import Dict, Any, List
from .verification_engine import VerificationEngine

class RegressionSuite:
    """
    Manages Golden Results and verifies performance across iterations.
    Organizes results by benchmark and model.
    """
    def __init__(self, golden_dir: str = "backend/validation/golden_results"):
        self.golden_dir = golden_dir
        os.makedirs(self.golden_dir, exist_ok=True)

    def define_golden(self, benchmark: str, model: str, metrics: Dict[str, Any], tolerances: Dict[str, float]):
        """
        Defines the expected values and tolerances for a specific benchmark/model pair.
        """
        golden_config = {
            "benchmark": benchmark,
            "model": model,
            "expected": metrics,
            "tolerances": tolerances
        }
        path = os.path.join(self.golden_dir, f"{benchmark}_{model}.json")
        with open(path, 'w') as f:
            json.dump(golden_config, f, indent=2)

    def verify_against_golden(self, benchmark: str, model: str, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        path = os.path.join(self.golden_dir, f"{benchmark}_{model}.json")
        if not os.path.exists(path):
            return {"status": "NOT_RUN", "message": "No golden record found"}
            
        with open(path, 'r') as f:
            golden = json.load(f)
            
        expected = golden["expected"]
        tolerances = golden["tolerances"]
        
        failures = []
        results = {}
        
        for metric, exp_val in expected.items():
            curr_val = current_metrics.get(metric)
            if curr_val is None:
                failures.append(f"Metric '{metric}' missing in current run.")
                continue
                
            tol = tolerances.get(metric, 0.05)
            match = VerificationEngine.check_tolerance(exp_val, curr_val, tol)
            results[metric] = {
                "expected": exp_val,
                "current": curr_val,
                "tolerance": tol,
                "passed": match
            }
            if not match:
                failures.append(f"Metric '{metric}' out of tolerance.")
                
        return {
            "benchmark": benchmark,
            "model": model,
            "passed": len(failures) == 0,
            "failures": failures,
            "detailed_results": results
        }
