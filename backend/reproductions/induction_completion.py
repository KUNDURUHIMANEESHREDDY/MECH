from .completion_criteria import CompletionCriteria
from typing import Dict, Any

class InductionCompletionValidator:
    """
    Validator for Induction Heads reproduction completion.
    """
    def __init__(self):
        self.criteria = CompletionCriteria(requirements=[
            "Real repeated-token prompts",
            "Detect actual induction heads",
            "Compare rankings with Olsson et al.",
            "Attention visualization",
            "Statistical report"
        ])

    def validate(self, reproduction_results: Dict[str, Any]) -> Dict[str, Any]:
        res = reproduction_results
        
        if res.get("prompt_type") == "repeated_tokens":
            self.criteria.mark_satisfied("Real repeated-token prompts")
            
        if "detected_induction_heads" in res:
            self.criteria.mark_satisfied("Detect actual induction heads")
            
        if "olsson_ranking_correlation" in res:
            self.criteria.mark_satisfied("Compare rankings with Olsson et al.")
            
        if "attention_plots" in res:
            self.criteria.mark_satisfied("Attention visualization")
            
        if "statistical_results" in res:
            self.criteria.mark_satisfied("Statistical report")
            
        return self.criteria.status_report()
