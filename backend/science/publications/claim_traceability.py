import hashlib
from typing import Dict, Any, List

class ClaimTraceability:
    """
    Ensures every scientific claim in the paper is traceable back to a specific experiment and dataset.
    Generates unique Traceability IDs for claims.
    """

    @staticmethod
    def generate_trace_id(claim_text: str, source_experiment_id: str) -> str:
        """
        Creates a deterministic hash for a claim tied to an experiment.
        """
        raw = f"{claim_text}_{source_experiment_id}"
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:8].upper()

    @classmethod
    def link_claim(cls, claim: str, experiment_id: str, evidence_key: str) -> Dict[str, str]:
        trace_id = cls.generate_trace_id(claim, experiment_id)
        return {
            "trace_id": trace_id,
            "claim": claim,
            "experiment_id": experiment_id,
            "evidence_key": evidence_key,
            "verification_link": f"mech://verify/{experiment_id}/{evidence_key}"
        }

    def generate_traceability_appendix(self, claims: List[Dict[str, str]]) -> str:
        """
        Generates a 'Claim Traceability' section for the paper appendix.
        """
        lines = ["## Appendix: Claim Traceability\n", "| ID | Claim | Evidence Source |", "|---|---|---|"]
        for c in claims:
            lines.append(f"| {c['trace_id']} | {c['claim']} | {c['evidence_key']} ({c['experiment_id']}) |")
        return "\n".join(lines)
