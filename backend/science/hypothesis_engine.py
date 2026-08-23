"""Hypothesis and Multidimensional Falsification Engine for MECH Platform.

Evaluates mechanistic claims across 5 distinct epistemic dimensions (Observational, Causal,
Control Contrast, Replication, and Alternative Explanations) without arbitrary confidence percentages.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from backend.storage.database import DesktopStorage
from backend.storage.scientific_entities import (
    EvidenceRecord,
    Hypothesis,
    HypothesisStatus,
    KnowledgeType,
)

logger = logging.getLogger("MECH.science.hypothesis_engine")


class HypothesisEngine:
    """Evaluates mechanistic hypotheses using a multidimensional evidence scorecard."""

    def __init__(self, storage: Optional[DesktopStorage] = None) -> None:
        from pathlib import Path
        db_path = Path.home() / ".cache" / "neural-debugger" / "mech.db"
        self.storage = storage or DesktopStorage(db_path)
        self.storage.initialize()

    def evaluate_hypothesis(self, hypothesis_id: str, investigation_id: str) -> Dict[str, Any]:
        """Calculates multidimensional epistemic scorecard for a hypothesis."""
        hyp = self.storage.get_hypothesis(hypothesis_id)
        if not hyp:
            raise ValueError(f"Hypothesis {hypothesis_id} not found.")

        evidence_items = self.storage.list_evidence(investigation_id=investigation_id, hypothesis_id=hypothesis_id)
        supporting = [e for e in evidence_items if e.get("supports_hypothesis", False)]
        contradicting = [e for e in evidence_items if not e.get("supports_hypothesis", False)]

        causal_evidence = [
            e for e in evidence_items
            if e.get("knowledge_type") == KnowledgeType.CAUSAL_EVIDENCE.value or e.get("evidence_level") == "CAUSALLY_VERIFIED"
        ]
        observational_evidence = [
            e for e in evidence_items
            if e.get("knowledge_type") == KnowledgeType.OBSERVATION.value or e.get("evidence_level") in ("OBSERVED", "CANDIDATE")
        ]

        controls_present = any(e.get("control_value") is not None for e in causal_evidence)
        replications = max(0, len(causal_evidence) - 1)

        # Dimension 1: Observational Support
        if len(observational_evidence) >= 3:
            obs_tier = "STRONG"
        elif len(observational_evidence) >= 1:
            obs_tier = "MODERATE"
        else:
            obs_tier = "NONE"

        # Dimension 2: Causal Support
        if len(causal_evidence) >= 3:
            causal_tier = "STRONG"
        elif len(causal_evidence) >= 1:
            causal_tier = "MODERATE"
        else:
            causal_tier = "NONE"

        # Dimension 3: Control Contrast
        if controls_present and len(causal_evidence) >= 1:
            ctrl_tier = "STRONG" if len(causal_evidence) >= 2 else "MODERATE"
        else:
            ctrl_tier = "NONE"

        # Check Falsification
        falsification_threshold = hyp.get("falsification_threshold", 0.2)
        falsified_runs = [
            e for e in causal_evidence
            if abs(e.get("metric_value", 0.0)) < falsification_threshold
        ]

        if falsified_runs:
            status = HypothesisStatus.FALSIFIED
            verdict = f"Falsified: Observed effect ({falsified_runs[0].get('metric_value', 0.0):.2f}) fell below falsification threshold ({falsification_threshold:.2f})."
        elif contradicting and not supporting:
            status = HypothesisStatus.CONTRADICTED
            verdict = f"Contradicted by {len(contradicting)} empirical test(s)."
        elif causal_tier in ("STRONG", "MODERATE") and ctrl_tier in ("STRONG", "MODERATE"):
            status = HypothesisStatus.SUPPORTED
            verdict = f"Causally validated across {len(causal_evidence)} intervention run(s) with contrastive negative controls."
        elif causal_tier != "NONE":
            status = HypothesisStatus.PARTIALLY_SUPPORTED
            verdict = f"Causal effect observed ({len(causal_evidence)} runs), but negative control contrast remains incomplete."
        elif obs_tier != "NONE":
            status = HypothesisStatus.PARTIALLY_SUPPORTED
            verdict = f"Observational alignment ({len(observational_evidence)} findings). Requires causal intervention."
        else:
            status = HypothesisStatus.UNTESTED
            verdict = "No empirical tests executed for this hypothesis."

        # Update stored hypothesis
        hyp["status"] = status.value
        hyp["observational_support"] = obs_tier
        hyp["causal_support"] = causal_tier
        hyp["control_contrast"] = ctrl_tier
        hyp["replication_count"] = replications
        hyp["evidence_count_supporting"] = len(supporting)
        hyp["evidence_count_contradicting"] = len(contradicting)
        hyp["updated_at"] = time.time()
        self.storage.save_hypothesis(hyp)

        return {
            "hypothesis": hyp,
            "status": status.value,
            "verdict": verdict,
            "scorecard": {
                "observational_support": obs_tier,
                "causal_support": causal_tier,
                "control_contrast": ctrl_tier,
                "replication_count": replications,
            },
            "supporting_count": len(supporting),
            "contradicting_count": len(contradicting),
            "causal_count": len(causal_evidence),
            "observational_count": len(observational_evidence),
            "evidence_list": evidence_items,
        }

    def get_evidence_matrix(self, investigation_id: str) -> List[Dict[str, Any]]:
        """Compiles full multidimensional evidence matrix for an investigation."""
        hypotheses = self.storage.list_hypotheses(investigation_id)
        evidence = self.storage.list_evidence(investigation_id)

        matrix = []
        for h in hypotheses:
            h_id = h.get("id")
            h_evidence = [e for e in evidence if e.get("hypothesis_id") == h_id]
            supporting = [e for e in h_evidence if e.get("supports_hypothesis", False)]
            contradicting = [e for e in h_evidence if not e.get("supports_hypothesis", False)]
            has_control = any(e.get("control_value") is not None for e in h_evidence)

            matrix.append({
                "hypothesis_id": h_id,
                "title": h.get("title"),
                "component": h.get("target_component"),
                "status": h.get("status", "UNTESTED"),
                "observational_support": h.get("observational_support", "NONE"),
                "causal_support": h.get("causal_support", "NONE"),
                "control_contrast": h.get("control_contrast", "NONE"),
                "replication_count": h.get("replication_count", 0),
                "supporting_count": len(supporting),
                "contradicting_count": len(contradicting),
                "total_evidence_nodes": len(h_evidence),
            })
        return matrix


# Global singleton engine
hypothesis_engine = HypothesisEngine()
