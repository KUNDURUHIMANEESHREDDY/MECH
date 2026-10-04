"""Society Inspector agent (capability: interpret).

Read-only introspection of live model internals. Primary source is
backend.services.gpt2_engine (simple module-function API); the six
backend/interpretability/inspectors/* classes are used opportunistically.

Real signatures (verified against the local tree):
- NeuronInspector().inspect(layer, neuron_index, activations)
- AttentionInspector().inspect(layer, head, tokens)
- ResidualInspector / LayerInspector / TokenInspector / PredictionInspector

Every method returns a dict and never raises: {"status": "error", ...}
on failure so the Supervisor can fail-fast cleanly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.core.provenance import pass_through, withhold


class Inspector:
    """Reads activations, heads, neurons, and predictions."""

    capability = "interpret"

    # -- gpt2_engine reads (preferred: live weights, simple API) --------
    def _engine(self):  # type: ignore[no-untyped-def]
        from backend.services import gpt2_engine
        return gpt2_engine

    def _engine_call(self, name: str, *args: Any) -> Dict[str, Any]:
        try:
            engine = self._engine()
        except Exception as exc:
            return {
                "status": "unavailable",
                "provenance": "unavailable",
                "field_provenance": {"status": "unavailable", "reason": "unavailable"},
                "reason": str(exc)[:300],
            }
        if not engine.is_available():
            return {
                "status": "unavailable",
                "provenance": "unavailable",
                "field_provenance": {"status": "unavailable", "reason": "unavailable"},
                "reason": "torch/transformers not installed",
            }
        try:
            res = getattr(engine, name)(*args)
            if isinstance(res, dict):
                # Never infer `live` here. This was a line-for-line copy of the
                # same defect in `agents/executor.py`: `setdefault("provenance",
                # "live")` plus a `field_provenance` defaulting every key to
                # live, so a returned `{"status": "unavailable"}` became a live
                # measurement and its `reason` was marked live too.
                return pass_through(res)
            # A bare return value carries no provenance, so it cannot be presented
            # as a measurement. This previously wrapped it as `status: ok,
            # provenance: live`, so a function returning None produced a live
            # record asserting a successful result.
            return withhold(
                {"status": "unmeasured", "op": name, "result": res},
                reason=(f"{name}() returned {type(res).__name__}, not a record. "
                        f"A bare return value carries no provenance."),
            )
        except Exception as exc:
            return withhold(
                {"status": "error", "op": name, "error": str(exc)[:500]},
                reason=f"{name} raised {type(exc).__name__}: {exc}"[:300],
            )

    def architecture(self) -> Dict[str, Any]:
        return self._engine_call("architecture")

    def layer(self, layer: int) -> Dict[str, Any]:
        return self._engine_call("layer_detail", layer)

    def neurons(self, layer: int, component: str = "mlp") -> Dict[str, Any]:
        return self._engine_call("list_neurons", layer, component)

    def neuron(self, layer: int, neuron_index: int) -> Dict[str, Any]:
        out = self._engine_call("neuron_detail", layer, neuron_index)
        if out.get("status") in ("unavailable", "error"):
            # Fall back to the interpretability inspector (needs activations).
            try:
                from backend.interpretability.inspectors.neuron_inspector import (
                    NeuronInspector,
                )
                insp = NeuronInspector().inspect(
                    layer=layer, neuron_index=neuron_index, activations=None)
                return {
                    "status": "ok",
                    "provenance": "unavailable",
                    "field_provenance": {"status": "unavailable", "result": "unavailable"},
                    "source": "NeuronInspector",
                    "result": insp,
                }
            except Exception as exc:
                out.setdefault("fallback_error", str(exc)[:300])
        return out

    def attention(self, layer: int, head: int,
                  tokens: Optional[List[str]] = None) -> Dict[str, Any]:
        out = self._engine_call("attention_head", layer, head)
        if out.get("status") == "error" and "prompt first" in str(
                out.get("error", "")):
            # The engine needs a cached forward pass before a head can be
            # read.  Prime it with the canonical IOI prompt so the returned
            # matrix is a live measurement, then retry once.
            primed = self._engine_call(
                "run_prompt",
                "When Alice and Bob went to the store, Alice gave a bottle to")
            if primed.get("status") == "ok":
                out = self._engine_call("attention_head", layer, head)
        if out.get("status") in ("unavailable", "error"):
            try:
                from backend.interpretability.inspectors.attention_inspector import (
                    AttentionInspector,
                )
                insp = AttentionInspector().inspect(
                    layer=layer, head=head, tokens=tokens or [])
                return {
                    "status": "ok",
                    "provenance": "unavailable",
                    "field_provenance": {"status": "unavailable", "result": "unavailable"},
                    "source": "AttentionInspector",
                    "result": insp,
                }
            except Exception as exc:
                out.setdefault("fallback_error", str(exc)[:300])
        return out

    def head(self, layer: int, head: int) -> Dict[str, Any]:
        return self._engine_call("head_detail", layer, head)

    def prediction(self, prompt: str) -> Dict[str, Any]:
        res = self._engine_call("run_prompt", prompt)
        if res.get("status") == "ok" and "result" in res:
            return res
        return res
