import numpy as np
from typing import Dict, Any, List
from ..science.statistics.statistical_validator import StatisticalValidator

class TrainingDynamicsDiscovery:
    """
    Discovery Project: Training dynamics of concept emergence.
    Analyzes how specific features (e.g., induction, names) develop over training steps.
    """
    def __init__(self, experiment_id: str):
        self.validator = StatisticalValidator(experiment_id)

    def analyze_emergence(self, checkpoints: List[int], activation_data: Dict[int, np.ndarray]) -> Dict[str, Any]:
        """
        Tracks the 'Scientific Quality' and 'Effect Size' of a concept across training checkpoints.
        """
        dynamics = []
        for step in checkpoints:
            acts = activation_data[step]
            # Mocking concept separation at this training step
            pos = acts[:50]
            neg = acts[50:]
            
            stats = self.validator.compare_groups(pos, neg, f"Step_{step}_Concept_Emergence")
            
            dynamics.append({
                "step": step,
                "effect_size": stats["effect_sizes"]["cohens_d"],
                "p_value": stats["p_value_raw"],
                "quality": stats["quality_score"]["score"]
            })
            
        # Detect the 'Phase Change' (Step where p < 0.001 and quality > 80)
        phase_change = next((d["step"] for d in dynamics if d["p_value"] < 0.001 and d["quality"] > 80), None)
        
        return {
            "dynamics": dynamics,
            "emergence_step": phase_change,
            "plateau_step": checkpoints[-1] if phase_change else None
        }
