from .completion_criteria import CompletionCriteria, Requirement
from typing import Dict, Any

class IOICompletionValidator:
    """
    Validator for IOI reproduction with automatic evidence collection.
    """
    def __init__(self):
        self.criteria = CompletionCriteria(requirements=[
            Requirement(
                "Real GPT-2 Small", 20, is_hard=True, 
                evidence_file="research_manifest.json", 
                evidence_check=lambda d: d.get("model") == "gpt2-small"
            ),
            Requirement(
                "Golden IOI dataset", 10, is_hard=True,
                evidence_file="research_manifest.json",
                evidence_check=lambda d: d.get("dataset_type") == "golden_ioi"
            ),
            Requirement(
                "TransformerLens parity", 10, is_hard=False,
                evidence_file="alignment_report.json",
                evidence_check=lambda d: d.get("alignment", 0) > 0.99
            ),
            Requirement(
                "Statistics", 15, is_hard=True,
                evidence_file="statistical_report.json",
                evidence_check=lambda d: d.get("is_significant", False)
            ),
            Requirement(
                "Peer review PASS", 15, is_hard=True,
                evidence_file="peer_review_report.json",
                evidence_check=lambda d: d.get("status") == "PASS"
            ),
            Requirement(
                "Artifacts", 5, is_hard=False,
                evidence_file="benchmark_certificate.json"
            )
        ])

    def validate(self, reproduction_results: Dict[str, Any]) -> Dict[str, Any]:
        for req in self.criteria.requirements:
            if req.evidence_check and req.evidence_check(reproduction_results):
                self.criteria.mark_satisfied(req.name)
        return self.criteria.status_report()

    def validate_project(self, project_dir: str) -> Dict[str, Any]:
        self.criteria.collect_evidence(project_dir)
        return self.criteria.status_report()
