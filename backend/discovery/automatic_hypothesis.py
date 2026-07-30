import numpy as np
from typing import Dict, Any, List
from ..science.statistics.statistical_validator import StatisticalValidator

class AutomaticHypothesisGenerator:
    """
    Discovery Project 7: Automatic hypothesis generation.
    Suggests research directions based on statistical anomalies.
    """
    def __init__(self, experiment_id: str):
        self.validator = StatisticalValidator(experiment_id)

    def generate_hypotheses(self, discovery_results: List[Dict[str, Any]]) -> List[str]:
        """
        Analyzes previous results to suggest new hypotheses.
        """
        hypotheses = []
        
        # Example logic: if two heads have similar effect sizes and layers, suggest interaction
        # (This would be more complex in reality, utilizing circuit analysis)
        
        high_impact_heads = [r for r in discovery_results if r.get("effect_size", 0) > 2.0]
        
        if len(high_impact_heads) >= 2:
            h1, h2 = high_impact_heads[0], high_impact_heads[1]
            hypotheses.append(
                f"Head {h1['id']} and Head {h2['id']} both show extreme impact. "
                f"I recommend testing for composition (Path Patching: {h1['id']} -> {h2['id']})."
            )
            
        if any(r.get("quality", 0) < 50 for r in discovery_results):
            hypotheses.append("Certain discoveries have low quality scores. I recommend increasing N and re-running with Bootstrap BCa.")
            
        return hypotheses
