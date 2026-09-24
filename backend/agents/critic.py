"""Society Critic agent (capability: validate).

Validates discoveries and reflects on runs. Wraps:
- backend.validation.validation_engine.ScientificValidationEngine
  .validate_discovery(discovery_id, hypothesis_statement)
  (validated == confidence >= 0.85 AND peer review == Accept)
- backend.research_platform.meta.self_reflection_engine.SelfReflectionEngine
  .generate_reflection_report(campaign_id, successful_hypotheses,
  failed_hypotheses, compute_used_gb_hours, planner_decisions)

CONFIDENCE_THRESHOLD mirrors the 0.85 gate inside ScientificValidationEngine
so the Supervisor can fail-fast without re-running validation.
"""

from __future__ import annotations

from typing import Any, Dict, List

CONFIDENCE_THRESHOLD = 0.85


class Critic:
    """Validates discoveries, scores confidence, reflects on campaigns."""

    capability = "validate"

    def validate(self, discovery_id: str, hypothesis: str) -> Dict[str, Any]:
        try:
            from backend.validation.validation_engine import (
                ScientificValidationEngine,
            )
            res = ScientificValidationEngine().validate_discovery(
                discovery_id=discovery_id, hypothesis_statement=hypothesis)
            return {"status": "completed", "result": res}
        except Exception as exc:
            return {"status": "error", "error": str(exc)[:500]}

    def is_confident(self, validation_result: Dict[str, Any]) -> bool:
        try:
            conf = validation_result.get("confidence", {})
            score = float(conf.get("confidence_score", 0.0))
            review = validation_result.get("peer_review", {})
            decision = review.get("decision", "")
            if score:
                return score >= CONFIDENCE_THRESHOLD and decision in ("", "Accept")
            return bool(validation_result.get("validated", False))
        except Exception:
            return False

    def reflect(self, campaign_id: str,
                successful: List[str], failed: List[str],
                planner_decisions: List[str]) -> Dict[str, Any]:
        try:
            from backend.research_platform.meta.self_reflection_engine import (
                SelfReflectionEngine,
            )
            res = SelfReflectionEngine().generate_reflection_report(
                campaign_id=campaign_id,
                successful_hypotheses=successful,
                failed_hypotheses=failed,
                compute_used_gb_hours=0.0,
                planner_decisions=planner_decisions,
            )
            return {"status": "completed", "result": res}
        except Exception as exc:
            return {"status": "error", "error": str(exc)[:500]}
