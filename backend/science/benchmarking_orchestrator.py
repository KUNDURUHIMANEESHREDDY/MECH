import os
import json
from typing import List, Dict, Any
from datetime import datetime
from .statistics.statistical_trace import StatisticalTrace

class BenchmarkingOrchestrator:
    """
    Orchestrates the execution of landmark benchmarks across multiple real models.
    Generates the benchmark database.
    """
    def __init__(self, database_path: str = "backend/science/benchmark_database.json"):
        self.database_path = database_path
        self.models = ["gpt2-small", "gpt2-medium", "gemma-2b", "llama-3-8b", "qwen-7b"]
        self.database: List[Dict[str, Any]] = self._load_database()

    def _load_database(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.database_path):
            with open(self.database_path, 'r') as f:
                return json.load(f)
        return []

    def _save_database(self):
        with open(self.database_path, 'w') as f:
            json.dump(self.database, f, indent=2)

    def run_full_campaign(self):
        """
        Executes all benchmarks across all supported models.
        """
        from ..reproductions.ioi_reproduction import IOIReproduction
        from ..reproductions.induction_heads import InductionHeadsReproduction
        from ..reproductions.sae_reproduction import SparseAutoencoderReproduction
        from ..reproductions.acdc_reproduction import ACDCReproduction

        for model_name in self.models:
            print(f"--- Running Benchmarks for {model_name} ---")
            
            # 1. IOI
            ioi = IOIReproduction(f"bench_{model_name}_ioi")
            ioi_res = ioi.run(model_name=model_name)
            self._record_result(model_name, "IOI", ioi_res)

            # 2. Induction
            ind = InductionHeadsReproduction(f"bench_{model_name}_ind")
            ind_res = ind.run(model_name=model_name)
            self._record_result(model_name, "Induction", ind_res)

            # 3. SAE
            sae = SparseAutoencoderReproduction(f"bench_{model_name}_sae")
            sae_res = sae.run(model_name=model_name)
            self._record_result(model_name, "SAE", sae_res)
            
            # 4. ACDC
            acdc = ACDCReproduction(f"bench_{model_name}_acdc")
            acdc_res = acdc.run(model_name=model_name)
            self._record_result(model_name, "ACDC", acdc_res)

        self._save_database()
        print("Campaign Completed. Database Updated.")

    def _record_result(self, model: str, benchmark: str, results: Dict[str, Any]):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": model,
            "benchmark": benchmark,
            "reproduction_successful": results.get("reproduction_successful"),
            "quality_score": results.get("completion_report", {}).get("completion_score"),
            "p_value": results.get("statistical_results", {}).get("p_value_raw"),
            "effect_size": results.get("statistical_results", {}).get("effect_sizes", {}).get("cohens_d")
        }
        self.database.append(entry)
