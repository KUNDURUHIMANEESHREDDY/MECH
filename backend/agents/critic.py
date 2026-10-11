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

from backend.core.provenance import pass_through

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
                # Propagated from the validation engine's re-measurement, never
                # asserted: the Critic judges, it does not run forward passes.
                "attested": res.get("attested", False),
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
        """Whether a validation result is confident *and* peer-accepted.

        The peer-review decision must be exactly "Accept". This used to be

            return score >= CONFIDENCE_THRESHOLD and decision in ("", "Accept")

        and `decision` defaults to "" at two levels -- an absent `peer_review`
        dict, and an absent `decision` key inside it. So a validation carrying a
        0.90 confidence score and *no peer review at all* passed this gate. The
        surrounding documentation says peer review must be Accept; the code
        permitted its absence, which is the one state that can never mean
        Accept.

        Only an explicit Accept passes. Missing, empty, pending, revise and
        reject all fail, and failing closed is the point: a review that has not
        happened is not a review that succeeded.
        """
        if not validation_is_live(validation_result):
            return False
        try:
            conf = validation_result.get("confidence", {})
            score = float(conf.get("confidence_score", 0.0))
            review = validation_result.get("peer_review", {})
            decision = str(review.get("decision", "") or "").strip()
            return score >= CONFIDENCE_THRESHOLD and decision == "Accept"
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
            # The pipeline reports whether each metric was actually measured
            # (e.g. circuit_minimality needs >= 10 usable prompts for its
            # majority vote). That flag has to survive into the report: mapping
            # an unmeasured 0.0 into observed_metrics made "not attempted" look
            # like "attempted and scored zero", and earned it a fidelity_pct.
            unmeasured = {
                required: observed.get(f"{observed_key}_measured", True) is False
                for required, observed_key in METRIC_MAP.items()
            }
            try:
                mapped = {}
                for required, observed_key in METRIC_MAP.items():
                    if unmeasured[required]:
                        mapped[required] = None
                        continue
                    value = float(observed[observed_key])
                    if not 0.0 <= value <= 1.0:
                        raise ValueError(f"{observed_key} outside [0, 1]")
                    mapped[required] = value
            except (KeyError, TypeError, ValueError) as exc:
                return {
                    "status": "unavailable",
                    "provenance": "unavailable",
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
                # Never infer `live` here.
                #
                # Was `report.setdefault("provenance", "live")` on a record built
                # by `ReproducibilityReportEngine.generate_report`. The Critic is a
                # *gate*, not the measurement layer: it did not run a forward
                # pass, so it cannot know whether the run it is judging measured
                # anything. Defaulting the label meant a report whose metrics were
                # all `NOT_RUN` still arrived labelled as a live measurement, and
                # the evidence policy -- which reads exactly this field -- would
                # accept it.
                report = pass_through(dict(report))
            # Gate value is the mean fidelity_pct over *measured* metrics, on
            # the same 0-100 scale the threshold is expressed in
            # (CONFIDENCE_THRESHOLD * 100).
            #
            # This used to do `m.get("fidelity_pct", 0.0)` inside a bare
            # `except Exception: pass`. Once unmeasured metrics correctly
            # carry fidelity_pct=None, the sum raised TypeError, the bare
            # except swallowed it, and the gate reported value=0.0 -- a
            # permanent fail that silently contradicted the report's own
            # overall_fidelity_pct and hid the error that caused it.
            metrics = report.get("metric_results", []) if isinstance(report, dict) else []
            scored = [
                m["fidelity_pct"] for m in metrics
                if isinstance(m, dict)
                and isinstance(m.get("fidelity_pct"), (int, float))
            ]
            unmeasured_names = [
                m.get("name") for m in metrics
                if isinstance(m, dict) and m.get("measured") is False
            ]
            if scored:
                fidelity: Optional[float] = round(sum(scored) / len(scored), 2)
            else:
                fidelity = None

            # Like-for-like calibration, when the pipeline supplied one.
            #
            # `fidelity` above compares each metric against the registry's
            # published constant. For IOI that constant is not the same
            # measurement this harness makes -- running the *published* circuit
            # through this harness gives a materially lower number than the
            # published 0.86. So the pipeline also measures the published
            # circuit through identical code and reports the ratio. That ratio
            # is the answerable question: is the discovered circuit as good as
            # the published one, measured the same way?
            same_harness_ref = observed.get(
                "reference_circuit_faithfulness_same_harness")
            discovered_faith = mapped.get("circuit_faithfulness")
            calibrated: Optional[Dict[str, Any]] = None
            if (isinstance(same_harness_ref, (int, float))
                    and same_harness_ref > 0
                    and isinstance(discovered_faith, (int, float))):
                ratio = discovered_faith / same_harness_ref
                calibrated = {
                    "metric": "circuit_faithfulness",
                    "discovered": round(discovered_faith, 4),
                    "published_circuit_same_harness": round(same_harness_ref, 4),
                    "ratio": round(ratio, 4),
                    # Matching or beating the published circuit through the
                    # same code is the like-for-like success condition.
                    "at_least_published": ratio >= 1.0,
                    "basis": "both circuits measured by this pipeline, same prompts",
                }

            gate = {
                "status": "completed" if fidelity is not None else "unavailable",
                "provenance": "live" if fidelity is not None else "unavailable",
                "threshold": CONFIDENCE_THRESHOLD,
                "metric": "overall_fidelity_pct",
                "value": fidelity,
                "metrics_scored": len(scored),
                "metrics_unmeasured": unmeasured_names,
                # An unmeasurable gate cannot pass.
                "passed": fidelity is not None and fidelity >= CONFIDENCE_THRESHOLD * 100,
                "reason": (
                    None if fidelity is not None else
                    "No metric produced a fidelity score; the gate cannot be "
                    "evaluated and does not pass."
                ),
                # Both fidelity_pct values above are measured against the
                # registry's *published* reference numbers. For IOI those are
                # not the same measurement as what this pipeline computes:
                # running the published IOI circuit through this same harness
                # yields a lower figure than the published 0.86. So
                # `passed` here is a comparison against an external constant,
                # not a like-for-like one. The calibrated comparison is below
                # and is the one that answers "is this circuit as good as the
                # published one".
                "reference_basis": "published_external_constants",
                "reference_basis_is_like_for_like": False,
                "calibrated": calibrated,
            }
            return {
                "status": "completed",
                "provenance": "live",
                # Propagated from the pipeline run above (measured or not);
                # the Critic maps metrics, it does not re-measure them.
                "attested": run.get("attested", False) if isinstance(run, dict) else False,
                "field_provenance": field_map(
                    ("status", "paper_id", "n_prompts", "observed_metrics", "report", "gate"),
                    "live",
                ),
                "paper_id": paper_id,
                # The like-for-like comparison, hoisted to the top level
                # because it is the answerable question. `report` and `gate`
                # compare against external published constants, which for IOI
                # are not the same measurement as this harness makes.
                "calibrated_vs_published_circuit": calibrated,
                "n_prompts": n_prompts,
                "mock_mode": False,
                "observed_metrics": mapped,
                # Which of the above carry a real measurement. `mapped` only
                # holds the validated metric set, so without this a caller
                # cannot tell a measured value from a None.
                "unmeasured_metrics": sorted(
                    name for name, is_unmeasured in unmeasured.items()
                    if is_unmeasured
                ),
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
