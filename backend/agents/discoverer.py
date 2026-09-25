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

from typing import Any, Dict


class Discoverer:
    """Discovers circuits, cross-model alignments, and feature genealogies."""

    capability = "discover"

    def discover(self, hypothesis: str) -> Dict[str, Any]:
        try:
            from .evidence_policy import (
                blocked_reason,
                discovery_is_live,
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
                    "validation_eligible": False,
                    "publication_eligible": False,
                    "discovery_id": res.get("discovery_id", ""),
                    "reason": blocked_reason(res, "Discovery"),
                }
            return {
                "status": "completed",
                "provenance": "live",
                "result": res,
            }
        except Exception as exc:
            return {
                "status": "error",
                "provenance": "unavailable",
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
            return {"status": "completed", "result": res}
        except Exception as exc:
            return {"status": "error", "error": str(exc)[:500]}

    def genealogy(self, feature_id: str = "f_1402") -> Dict[str, Any]:
        try:
            from backend.interpretability.discovery.feature_genealogy import (
                FeatureGenealogyEngine,
            )
            res = FeatureGenealogyEngine().get_genealogy(feature_id=feature_id)
            return {"status": "completed", "result": res}
        except Exception as exc:
            return {"status": "error", "error": str(exc)[:500]}

    def circuits(self) -> Dict[str, Any]:
        try:
            from backend.science.explorer.circuit_explorer import CircuitExplorer
            res = CircuitExplorer().list_circuits()
            return {"status": "completed", "result": res}
        except Exception as exc:
            return {"status": "error", "error": str(exc)[:500]}

    def circuit(self, circuit_id: str = "c_ioi") -> Dict[str, Any]:
        try:
            from backend.science.explorer.circuit_explorer import CircuitExplorer
            res = CircuitExplorer().get_circuit(circuit_id=circuit_id)
            return {"status": "completed", "result": res}
        except Exception as exc:
            return {"status": "error", "error": str(exc)[:500]}
