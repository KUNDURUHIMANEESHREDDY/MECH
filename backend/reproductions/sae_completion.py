from .completion_criteria import CompletionCriteria, Requirement
from typing import Dict, Any

class SAECompletionValidator:
    """
    Validator for SAE reproduction completion.
    """
    def __init__(self):
        self.criteria = CompletionCriteria(requirements=[
            Requirement("Load real SAE checkpoint", 25, is_hard=True),
            Requirement("Measure reconstruction", 15, is_hard=True, dependencies=["Load real SAE checkpoint"]),
            Requirement("FVE", 10, is_hard=True, dependencies=["Measure reconstruction"]),
            Requirement("L0", 10, is_hard=True, dependencies=["Measure reconstruction"]),
            Requirement("Compare against Bricken et al.", 20, is_hard=True, dependencies=["FVE", "L0"]),
            Requirement("Dead features", 10, is_hard=False),
            Requirement("Visualizations", 10, is_hard=False)
        ])

    def validate(self, reproduction_results: Dict[str, Any]) -> Dict[str, Any]:
        res = reproduction_results
        
        if res.get("sae_checkpoint_loaded"):
            self.criteria.mark_satisfied("Load real SAE checkpoint")
            
        if "reconstruction_loss" in res:
            self.criteria.mark_satisfied("Measure reconstruction")
            
        if "fve" in res:
            self.criteria.mark_satisfied("FVE")
            
        if "l0_norm" in res:
            self.criteria.mark_satisfied("L0")
            
        if "bricken_comparison" in res:
            self.criteria.mark_satisfied("Compare against Bricken et al.")
            
        if "dead_features_count" in res:
            self.criteria.mark_satisfied("Dead features")
            
        return self.criteria.status_report()
