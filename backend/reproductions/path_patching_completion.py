from .completion_criteria import CompletionCriteria
from typing import Dict, Any

class PathPatchingCompletionValidator:
    """
    Validator for Path Patching reproduction completion.
    """
    def __init__(self):
        self.criteria = CompletionCriteria(requirements=[
            "Real activation patching",
            "Necessity",
            "Sufficiency",
            "Published comparison"
        ])

    def validate(self, reproduction_results: Dict[str, Any]) -> Dict[str, Any]:
        res = reproduction_results
        
        if res.get("technique") == "activation_patching":
            self.criteria.mark_satisfied("Real activation patching")
            
        if "necessity_results" in res:
            self.criteria.mark_satisfied("Necessity")
            
        if "sufficiency_results" in res:
            self.criteria.mark_satisfied("Sufficiency")
            
        if "published_comparison" in res:
            self.criteria.mark_satisfied("Published comparison")
            
        return self.criteria.status_report()
