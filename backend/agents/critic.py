"""Society Critic agent (capability: validate).

Validates discoveries and reflects on runs. Wraps:
- backend.validation.validation_engine.ScientificValidationEngine
  .validate_discovery(discovery_id, hypothesis_statement)
  (validated == confidence >= 0.85 AND peer review == Accept)
- backend.research_platform.meta.self_reflection_engine.SelfReflectionEngine
  .generate_reflection_report(campaign_id, successful_hypotheses,
  failed_hypotheses, compute_used_gb_hours, planner_decisions)

CONFIDENCE_THRESHOLD mirrors the 0.85 gate inside ScientificValidationEngine
so the Supervisor can fail-fast without re-running validation. All validation
inputs must also carry explicit live provenance and downstream eligibility.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .evidence_policy import (
    blocked_reason,
    discovery_is_live,
    field_map,
    provenance_of,
    reproduction_is_live,
    validation_is_live,
)

CONFIDENCE_THRESHOLD = 0.85

# Paper metric <- pipeline observed metric (documented, semantic match).
# circuit_completeness is proxied by functional_recovery (injection-recovered
# logit-diff ratio); circuit_minimality is the measured fraction of circuit
# heads individually necessary on usable prompts.
METRIC_MAP = {"circuit_faithfulness": "circuit_faithfulness",
              "circuit_completeness": "functional_recovery",
              "circuit_minimality": "circuit_minimality"}


class Critic:
    """Validates discoveries, scores confidence, reflects on campaigns."""

    capability = "validate"

    def validate(
        self,
        discovery_id: str,
        hypothesis: str,
        discovery_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not discovery_is_live(discovery_result):
            return {
                "status": "unavailable",
                "provenance": provenance_of(discovery_result),
                "field_provenance": field_map(
                    ("status", "validated", "reason"), provenance_of(discovery_result)
                ),
                "validated": False,
                "validation_eligible": False,
                "publication_eligible": False,
                "reason": blocked_reason(discovery_result, "Validation input"),
            }
        try:
            from backend.validation.validation_engine import (
                ScientificValidationEngine,
            )
            res = ScientificValidationEngine().validate_discovery(
                discovery_id=discovery_id, hypothesis_statement=hypothesis,
                discovery_result=discovery_result)
            if not validation_is_live(res):
                return {
                    "status": "unavailable",
                    "provenance": provenance_of(res),
                    "field_provenance": field_map(
                        ("status", "validated", "reason"), provenance_of(res)
                    ),
                    "validated": False,
                    "validation_eligible": False,
                    "publication_eligible": False,
                    "reason": blocked_reason(res, "Validation"),
                }
            return {
                "status": "completed",
                "provenance": "live",
                "field_provenance": field_map(
                    ("status", "validated", "result", "confidence", "peer_review"),
                    "live",
                ),
                "result": res,
            }
        except Exception as exc:
            return {
                "status": "error",
                "provenance": "unavailable",
                "field_provenance": field_map(
                    ("status", "validated", "error"), "unavailable"
                ),
                "validated": False,
                "validation_eligible": False,
                "publication_eligible": False,
                "error": str(exc)[:500],
            }

    def is_confident(self, validation_result: Dict[str, Any]) -> bool:
        if not validation_is_live(validation_result):
            return False
        try:
            conf = validation_result.get("confidence", {})
            score = float(conf.get("confidence_score", 0.0))
            review = validation_result.get("peer_review", {})
            decision = review.get("decision", "")
            return score >= CONFIDENCE_THRESHOLD and decision in ("", "Accept")
        except Exception:
            return False

    def reproduce(self, paper_id: str = "ioi",
                    n_prompts: int = 16) -> Dict[str, Any]:
        """Closed validation loop: live pipeline -> observed metrics ->
        ReproducibilityReport vs published (TransformerLens-paper) baselines
        -> 0.85 gate verdict.

        Live weights only: returns status unavailable without torch. Small
        default n keeps routes/tests tractable; n_samples is reported.
        """
        if paper_id != "ioi":
            return {
                "status": "unavailable",
                "provenance": "unavailable",
                "field_provenance": field_map(
                    ("status", "observed_metrics", "report", "gate", "reason"),
                    "unavailable",
                ),
                "publication_eligible": False,
                "reason": f"live reproduction for '{paper_id}' "
                          "not implemented (ioi only)",
            }
        try:
            from backend.services import gpt2_engine
            if not gpt2_engine.is_available():
                return {
                    "status": "unavailable",
                    "provenance": "unavailable",
                    "field_provenance": field_map(
                        ("status", "observed_metrics", "report", "gate", "reason"),
                        "unavailable",
                    ),
                    "publication_eligible": False,
                    "reason": "torch/transformers not installed",
                }
            from backend.science.reproducibility.ioi_pipeline import (
                IOIReproductionPipeline,
            )
            from backend.science.reproducibility.reproducibility_report import (
                ReproducibilityReportEngine,
            )
            run = IOIReproductionPipeline(mock_mode=False).run(
                n_prompts=n_prompts)
            if run.get("mock_mode") is not False or not reproduction_is_live(run):
                return {
                    "status": "unavailable",
                    "provenance": provenance_of(run),
                    "field_provenance": field_map(
                        ("status", "observed_metrics", "report", "gate", "reason"),
                        provenance_of(run),
                    ),
                    "publication_eligible": False,
                    "reason": "The reproduction pipeline did not return explicit live evidence.",
                }
            observed = dict(run.get("observed_metrics", {}))
            try:
                mapped = {}
                for required, observed_key in METRIC_MAP.items():
                    value = float(observed[observed_key])
                    if not 0.0 <= value <= 1.0:
                        raise ValueError(f"{observed_key} outside [0, 1]")
                    mapped[required] = value
            except (KeyError, TypeError, ValueError) as exc:
                return {
                    "status": "unavailable",
                    "provenance": "live",
                    "field_provenance": field_map(
                        ("status", "observed_metrics", "report", "gate", "reason"),
                        "unavailable",
                    ),
                    "publication_eligible": False,
                    "reason": f"Live reproduction metrics are incomplete: {exc}",
                }
            report_engine = ReproducibilityReportEngine()
            report = report_engine.generate_report(
                paper_id=paper_id,
                pipeline_name="IOIReproductionPipeline-HighFidelity",
                model_id=run.get("model_id", "gpt2-small"),
                dataset_manifest_id=run.get("manifest_id", ""),
                observed_metrics=mapped,
                explanation_of_diffs=[
                    "circuit_completeness proxied by functional_recovery "
                    "(injection-recovered logit-diff ratio).",
                    "circuit_minimality measured as the fraction of circuit "
                    "heads individually necessary on usable prompts.",
                    f"live run over {observed.get('n_samples', n_prompts)} "
                    "prompts; mock_mode=False.",
                ],
            )
            if isinstance(report, dict):
                report = dict(report)
                report.setdefault("provenance", "live")
            fidelity = 0.0
            try:
                metrics = report.get("metrics", report.get("metric_results",
                                                           []))
                scored = [m.get("fidelity_pct", 0.0) for m in metrics
                          if isinstance(m, dict)]
                if scored:
                    fidelity = round(sum(scored) / len(scored), 2)
            except Exception:
                pass
            gate = {
                "status": "completed",
                "provenance": "live",
                "threshold": CONFIDENCE_THRESHOLD,
                "metric": "overall_fidelity_pct",
                "value": fidelity,
                "passed": fidelity >= CONFIDENCE_THRESHOLD * 100,
            }
            return {
                "status": "completed",
                "provenance": "live",
                "field_provenance": field_map(
                    ("status", "paper_id", "n_prompts", "observed_metrics", "report", "gate"),
                    "live",
                ),
                "paper_id": paper_id,
                "n_prompts": n_prompts,
                "mock_mode": False,
                "observed_metrics": mapped,
                "report": report if isinstance(report, dict) else
                {"report": report},
                "gate": gate,
            }
        except Exception as exc:
            return {
                "status": "error",
                "provenance": "unavailable",
                "field_provenance": field_map(
                    ("status", "observed_metrics", "report", "gate", "error"),
                    "unavailable",
                ),
                "publication_eligible": False,
                "error": str(exc)[:500],
            }

    def reflect(self, campaign_id: str,
                successful: List[str], failed: List[str],
                planner_decisions: List[str]) -> Dict[str, Any]:
        return {
            "status": "unavailable",
            "provenance": "unavailable",
            "field_provenance": field_map(
                ("status", "campaign_id", "successful", "failed", "result"),
                "unavailable",
            ),
            "campaign_id": campaign_id,
            "successful": successful,
            "failed": failed,
            "planner_decisions": planner_decisions,
            "reason": "No live reflection executor is connected.",
        }
