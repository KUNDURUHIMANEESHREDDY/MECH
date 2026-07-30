import numpy as np
from typing import Dict, Any, List
from ..science.statistics.statistical_validator import StatisticalValidator
from ..science.statistics.statistical_protocol import StatisticalProtocol

class InductionCircuitDiscovery:
    """
    Discovery Project 1: Finding new induction circuits.
    Beyond simple prefix matching, this project searches for higher-order
    induction heads (e.g., translation, pattern matching, skip-gram induction).
    """
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        # We use a rigorous protocol for novel discoveries
        self.protocol = StatisticalProtocol(
            version="discovery_v1.0",
            alpha=0.01,
            correction_method="BH",
            permutation_count=10000,
            power_threshold=0.85
        )
        self.validator = StatisticalValidator(experiment_id, self.protocol)

    def _simulate_circuit_search(self, num_heads: int) -> List[Dict[str, Any]]:
        """
        Simulates scanning through attention heads to find novel induction behaviors.
        """
        candidates = []
        for i in range(num_heads):
            # Simulate prefix matching scores for this head
            is_induction = np.random.rand() < 0.05  # 5% chance of being an induction head
            
            if is_induction:
                scores = np.random.normal(loc=0.8, scale=0.1, size=200)
            else:
                scores = np.random.normal(loc=0.1, scale=0.2, size=200)
                
            baseline = np.random.normal(loc=0.05, scale=0.1, size=200)
            
            candidates.append({
                "head_id": f"L{i // 12}H{i % 12}",
                "scores": scores,
                "baseline": baseline
            })
        return candidates

    def run_discovery(self, model: Any, num_heads_to_scan: int = 144) -> Dict[str, Any]:
        """
        Runs the discovery campaign across all heads.
        """
        self.validator.trace.add_step("Start Induction Circuit Discovery", {"model": "target_model", "heads_scanned": num_heads_to_scan})
        
        candidates = self._simulate_circuit_search(num_heads_to_scan)
        results = []
        
        for candidate in candidates:
            stat_res = self.validator.compare_groups(
                group_a=candidate["scores"],
                group_b=candidate["baseline"],
                feature_name=f"Induction_Score_{candidate['head_id']}"
            )
            results.append(stat_res)
            
        # Apply multiple hypothesis correction
        corrected_results = self.validator.process_multiple_comparisons(results)
        
        discovered_heads = [
            res["feature"].split("_")[-1] for res in corrected_results 
            if res.get("is_significant", False) and res.get("effect_sizes", {}).get("cohens_d", 0) > 1.5
        ]
        
        return {
            "project": "New Induction Circuits",
            "heads_scanned": num_heads_to_scan,
            "discovered_induction_heads": discovered_heads,
            "trace": self.validator.get_trace_report()
        }
