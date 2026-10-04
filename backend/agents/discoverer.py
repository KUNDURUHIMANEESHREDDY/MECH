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
                field_map,
                provenance_of,
            )
            from backend.interpretability.discovery.discovery_engine import (
                DiscoveryEngine,
            )
            engine = DiscoveryEngine()
            res = engine.discover_and_orchestrate(hypothesis_statement=hypothesis)
            if not discovery_is_live(res):
                # Do not place the synthetic result under ``result``: that
                # key is consumed by validation, evidence graphs, and the
                # scribe as scientific evidence.
                return {
                    "status": "unavailable",
                    "provenance": provenance_of(res),
                    "field_provenance": field_map(
                        ("status", "discovery_id", "reason", "result"),
                        provenance_of(res),
                    ),
                    "validation_eligible": False,
                    "publication_eligible": False,
                    "discovery_id": res.get("discovery_id", ""),
                    "reason": blocked_reason(res, "Discovery"),
                }
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
            return {
                **res,
                "status": "completed",
                "provenance": "live",
                "field_provenance": field_map(
                    ("status", "result", "discovery_id"), "live"
                ),
            }
        except Exception as exc:
            return {
                "status": "error",
                "provenance": "unavailable",
                "field_provenance": field_map(
                    ("status", "error", "result"), "unavailable"
                ),
                "validation_eligible": False,
                "publication_eligible": False,
                "error": str(exc)[:500],
            }

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
