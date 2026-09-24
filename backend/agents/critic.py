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

# Paper metric <- pipeline observed metric (documented, semantic match).
# circuit_minimality has no pipeline measure: reported honestly as
# unmeasured (0.0 + explanation) rather than fabricated.
METRIC_MAP = {"circuit_faithfulness": "circuit_faithfulness",
              "circuit_completeness": "functional_recovery"}


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

    def reproduce(self, paper_id: str = "ioi",
                    n_prompts: int = 16) -> Dict[str, Any]:
        """Closed validation loop: live pipeline -> observed metrics ->
        ReproducibilityReport vs published (TransformerLens-paper) baselines
        -> 0.85 gate verdict.

        Live weights only: returns status unavailable without torch. Small
        default n keeps routes/tests tractable; n_samples is reported.
        """
        if paper_id != "ioi":
            return {"status": "unavailable",
                    "reason": f"live reproduction for '{paper_id}' "
                              "not implemented (ioi only)"}
        try:
            from backend.services import gpt2_engine
            if not gpt2_engine.is_available():
                return {"status": "unavailable",
                        "reason": "torch/transformers not installed"}
            from backend.science.reproducibility.ioi_pipeline import (
                IOIReproductionPipeline,
            )
            from backend.science.reproducibility.reproducibility_report import (
                ReproducibilityReportEngine,
            )
            run = IOIReproductionPipeline(mock_mode=False).run(
                n_prompts=n_prompts)
            observed = dict(run.get("observed_metrics", {}))
            mapped = {req: float(observed.get(obs, 0.0) or 0.0)
                      for req, obs in METRIC_MAP.items()}
            report_engine = ReproducibilityReportEngine()
            report = report_engine.generate_report(
                paper_id=paper_id,
                pipeline_name="IOIReproductionPipeline-HighFidelity",
                model_id=run.get("model_id", "gpt2-small"),
                dataset_manifest_id=run.get("manifest_id", ""),
                observed_metrics=mapped,
                explanation_of_diffs=[
                    "circuit_completeness proxied by functional_recovery "
                    "(isolated-circuit top-logit ratio).",
                    "circuit_minimality not measured by this pipeline; "
                    "reported as 0 pending a redundancy ablation.",
                    f"live run over {observed.get('n_samples', n_prompts)} "
                    "prompts; mock_mode=False.",
                ],
            )
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
            gate = {"threshold": CONFIDENCE_THRESHOLD,
                    "metric": "overall_fidelity_pct",
                    "value": fidelity,
                    "passed": fidelity >= CONFIDENCE_THRESHOLD * 100}
            return {"status": "completed",
                    "paper_id": paper_id,
                    "n_prompts": n_prompts,
                    "mock_mode": run.get("mock_mode", False),
                    "observed_metrics": mapped,
                    "report": report if isinstance(report, dict) else
                    {"report": report},
                    "gate": gate}
        except Exception as exc:
            return {"status": "error", "error": str(exc)[:500]}

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
