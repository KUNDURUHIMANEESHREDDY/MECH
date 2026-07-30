from .completion_criteria import CompletionCriteria
from typing import Dict, Any

class ACDCCompletionValidator:
    """
    Validator for ACDC reproduction completion.
    """
    def __init__(self):
        self.criteria = CompletionCriteria(requirements=[
            "Real edge search",
            "Recover IOI circuit",
            "Precision",
            "Recall",
            "Edge F1",
            "Functional recovery"
        ])

    def validate(self, reproduction_results: Dict[str, Any]) -> Dict[str, Any]:
        res = reproduction_results
        
        if res.get("algorithm") == "ACDC_edge_search":
            self.criteria.mark_satisfied("Real edge search")
            
        if res.get("circuit_recovered") == "IOI":
            self.criteria.mark_satisfied("Recover IOI circuit")
            
        if "precision" in res:
            self.criteria.mark_satisfied("Precision")
            
        if "recall" in res:
            self.criteria.mark_satisfied("Recall")
            
        if "edge_f1" in res:
            self.criteria.mark_satisfied("Edge F1")
            
        if res.get("functional_recovery_score", 0) > 0.9:
            self.criteria.mark_satisfied("Functional recovery")
            
        return self.criteria.status_report()
