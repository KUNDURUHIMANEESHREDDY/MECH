"""Society Discoverer agent (capability: discover).

Runs circuit/feature discovery over observed activations. Wraps:
- backend.interpretability.discovery.discovery_engine.DiscoveryEngine
  .discover_and_orchestrate(hypothesis_statement)  [currently unavailable
  until a live discovery executor is connected]
- backend.interpretability.discovery.cross_model_circuits
  .CrossModelCircuitsEngine.compare_circuits(source_model, target_model,
  circuit_type)
- backend.interpretability.discovery.feature_genealogy
  .FeatureGenealogyEngine.get_genealogy(feature_id)
- backend.science.explorer.circuit_explorer.CircuitExplorer
  .list_circuits() / .get_circuit(circuit_id)

Legacy stubs replaced (all returned hardcoded literals):
_circuits_discover {c_ioi, 0.945}, _causal_trace {0.85},
_attribution_patch {Gradient, 0.91}, _features_label/cluster,
_polysemanticity_detect, _circuits_evolution/name_auto.
"""

from __future__ import annotations

from typing import Any, Dict, List

from backend.core.provenance import admit_result

#: Sample sizes the Society's discovery asks for. These are
#: `live_discovery.MIN_PROMPTS_FOR_VALIDATION` and
#: `..._MIN_PROMPTS_FOR_INTERACTION` -- the thresholds the executor itself uses
#: to derive eligibility. Asserted equal below so a change to the executor's
#: thresholds fails a test rather than silently making the Society's discovery
#: ineligible again.
SOCIETY_DISCOVERY_N_PROMPTS = 10
SOCIETY_DISCOVERY_N_INTERACTION_PROMPTS = 5


def _wrap_engine_result(result: Any, engine: str) -> Dict[str, Any]:
    """Wrap a sub-engine's result without overriding its own verdict.

    These wrappers used to hardcode ``{"status": "completed"}``. That mattered
    because the engines underneath had already been quarantined: they return
    ``status: "unavailable"`` with a reason explaining that nothing was
    measured. The wrapper discarded that and reported success, so the
    quarantine was invisible to every caller reaching it through the agent --
    and an unreachable engine looked like a working one.

    An empty list is reported as "completed" because a list genuinely is a
    result: there may simply be no circuits. It is still not evidence.
    """
    try:
        from .evidence_policy import field_map
    except Exception:  # pragma: no cover - evidence_policy is a hard dep
        field_map = None  # type: ignore[assignment]

    if isinstance(result, dict):
        status = str(result.get("status") or "completed")
        provenance = str(result.get("provenance") or "unavailable")
        body = dict(result)
    else:
        # A bare value (e.g. a list of circuits). It is a result, but it
        # carries no provenance of its own.
        status = "completed"
        provenance = "reference"
        body = {"result": result}

    measured = bool(body.get("measured", body.get("validation_eligible", False)))
    fields: List[str] = ("status", "result", "reason")
    out: Dict[str, Any] = {
        "status": status,
        "provenance": provenance,
        "engine": engine,
        "result": body,
        "measured": measured,
        "validation_eligible": measured,
        "publication_eligible": measured,
    }
    if field_map is not None:
        out["field_provenance"] = field_map(tuple(fields), provenance)
    if body.get("reason"):
        out["reason"] = body["reason"]
    return out


def _engine_error(engine: str, exc: Exception) -> Dict[str, Any]:
    try:
        from .evidence_policy import field_map
    except Exception:  # pragma: no cover
        field_map = None  # type: ignore[assignment]
    out: Dict[str, Any] = {
        "status": "error",
        "provenance": "unavailable",
        "engine": engine,
        "error": str(exc)[:500],
        "measured": False,
        "validation_eligible": False,
        "publication_eligible": False,
    }
    if field_map is not None:
        out["field_provenance"] = field_map(
            ("status", "error", "result"), "unavailable")
    return out


class Discoverer:
    """Discovers circuits, cross-model alignments, and feature genealogies."""

    capability = "discover"

    def discover(self, hypothesis: str) -> Dict[str, Any]:
        try:
            from .evidence_policy import (
                blocked_reason,
                discovery_is_live,
            )
            from backend.interpretability.discovery.discovery_engine import (
                DiscoveryEngine,
            )
            engine = DiscoveryEngine()
            res = engine.discover_and_orchestrate(
                hypothesis_statement=hypothesis,
                # The Society's purpose is a publishable finding, so its
                # discovery must ask for a sample adequate to support one. These
                # are `live_discovery.MIN_PROMPTS_FOR_VALIDATION` and
                # `..._FOR_INTERACTION`; passing them here is the caller's cost
                # decision, which is where the executor's docstring says it
                # belongs. Omitting them left every Society discovery at the
                # four-prompt default and permanently ineligible.
                n_prompts=SOCIETY_DISCOVERY_N_PROMPTS,
                n_interaction_prompts=SOCIETY_DISCOVERY_N_INTERACTION_PROMPTS,
            )
            if not discovery_is_live(res):
                # The same gate, on the refusal path. The label is still the
                # engine's own, the eligibility is still derived rather than
                # asserted, and the reason still says why. The synthetic
                # result is deliberately not placed under ``result``: that
                # key is consumed by validation, evidence graphs, and the
                # scribe as scientific evidence.
                return admit_result(
                    res, stage="discovery",
                    extra={"discovery_id": res.get("discovery_id", "")},
                    reason=blocked_reason(res, "Discovery"),
                )
            # `**res` first, the guarded verdict last.
            #
            # The spread used to come last, so it overwrote the keys this
            # function had just decided. `res` carries `status`, `provenance` and
            # `field_provenance` of its own, so the guard's answer was discarded
            # and replaced by whatever the engine happened to say -- the same
            # shape as the P0 defect, where a wrapper's provenance verdict was not
            # authoritative.
            #
            # It was not an active misreport: `discovery_is_live` is
            # `provenance_of(payload) == "live"` among other conditions, so the
            # value being spread in agreed with the value being guarded. But the
            # guard decided nothing, and it would start mattering the moment that
            # predicate diverged from a bare provenance comparison.
            # Built from the response's own keys, not a fixed tuple.
            #
            # `**res` copies the engine's payload verbatim. The map used to
            # name three fields, so a live response returned every field the
            # engine chose to emit with no statement of their standing --
            # measured at ten unmarked keys on a payload carrying `raw_traces`,
            # `per_prompt` and `metrics`, all measurement-shaped values
            # reaching the caller at `provenance: "live"`. Those are the
            # fields a consumer quotes, and the guarded-verdict ordering above
            # protects only the three keys the wrapper decided for itself.
            # The verdict itself is no longer assembled here either: the
            # shared admission gate derives it, so the label, the attestation
            # flag and the eligibility flags are decided in exactly one
            # place. `**res` still comes first, so the payload cannot
            # overwrite the verdict.
            admitted = admit_result(res, stage="discovery", extra=res)
            out = {
                **res,
                "status": admitted["status"],
                "provenance": admitted["provenance"],
                "attested": admitted["attested"],
                "validation_eligible": admitted["validation_eligible"],
                "publication_eligible": admitted["publication_eligible"],
            }
            out["field_provenance"] = admitted["field_provenance"]
            return out
        except Exception as exc:
            # The exception arm builds a fresh envelope rather than echoing
            # whatever it had in hand: a payload reached under `status:
            # "error"` would still be read by the evidence chain. The gate
            # derives the label and the eligibility here too.
            return admit_result(
                {}, stage="discovery", status="error",
                extra={"error": str(exc)[:500]},
                reason=str(exc)[:500],
            )

    def cross_model(self, source_model: str = "gpt2",
                    target_model: str = "gpt2-medium",
                    circuit_type: str = "ioi") -> Dict[str, Any]:
        try:
            from backend.interpretability.discovery.cross_model_circuits import (
                CrossModelCircuitsEngine,
            )
            res = CrossModelCircuitsEngine().compare_circuits(
                source_model=source_model, target_model=target_model,
                circuit_type=circuit_type)
            return _wrap_engine_result(res, "cross_model_circuits")
        except Exception as exc:
            return _engine_error("cross_model_circuits", exc)

    def genealogy(self, feature_id: str = "f_1402") -> Dict[str, Any]:
        try:
            from backend.interpretability.discovery.feature_genealogy import (
                FeatureGenealogyEngine,
            )
            res = FeatureGenealogyEngine().get_genealogy(feature_id=feature_id)
            return _wrap_engine_result(res, "feature_genealogy")
        except Exception as exc:
            return _engine_error("feature_genealogy", exc)

    def circuits(self) -> Dict[str, Any]:
        try:
            from backend.science.explorer.circuit_explorer import CircuitExplorer
            res = CircuitExplorer().list_circuits()
            return _wrap_engine_result(res, "circuit_explorer")
        except Exception as exc:
            return _engine_error("circuit_explorer", exc)

    def circuit(self, circuit_id: str = "c_ioi") -> Dict[str, Any]:
        try:
            from backend.science.explorer.circuit_explorer import CircuitExplorer
            res = CircuitExplorer().get_circuit(circuit_id=circuit_id)
            return _wrap_engine_result(res, "circuit_explorer")
        except Exception as exc:
            return _engine_error("circuit_explorer", exc)
