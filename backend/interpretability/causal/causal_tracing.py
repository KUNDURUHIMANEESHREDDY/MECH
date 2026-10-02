"""Causal Tracing Engine.

NOT IMPLEMENTED. ``trace_causal_effect`` synthesises a layer-effect curve from
a closed-form expression (``0.12 + 0.78`` for layers 8-9, else a small
modulo term) and reports ``max_causal_layer: 8`` unconditionally. No model is
loaded, no clean/corrupted run is performed, and no residual stream is
ablation-swept.

The clean/corrupted prompts were echoed back into the response, which made the
result look like the supplied prompts had been run. The fixture is retained for
inspection but is now labelled and ineligible.

A real implementation: for each layer, zero the clean residual stream, restore
the corrupted one, and measure the change in the (Paris - Rome) logit
difference. The live dispatcher route (`interpretability/causal/trace`) already
does exactly this when GPT-2 is loaded.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List


class CausalTracingEngine:
    """Reference fixture only. No model is loaded and nothing is ablated."""

    def trace_causal_effect(
        self,
        clean_prompt: str = "The capital of France is",
        corrupted_prompt: str = "The capital of Rome is",
    ) -> Dict[str, Any]:
        layer_effects: List[Dict[str, Any]] = []
        for l in range(12):
            causal_effect = round(0.12 + (0.78 if l in [8, 9] else 0.05 * (l % 3)), 3)
            layer_effects.append({
                "layer": l,
                "indirect_effect": causal_effect,
                "top_causal_component": "MLP" if l in [8, 9] else "Attention",
            })

        return {
            "status": "unavailable",
            "provenance": "reference",
            "method": "SyntheticCurve",
            "trace_measured": False,
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": (
                "Causal tracing is not implemented here. The layer effects are "
                "a closed-form expression, not an ablation sweep, and the "
                "supplied prompts were not run. Use the live dispatcher route "
                "for a real leave-one-layer-out trace."
            ),
            "clean_prompt": clean_prompt,
            "corrupted_prompt": corrupted_prompt,
            "max_causal_layer": 8,
            "max_indirect_effect": 0.90,
            "layer_effects": layer_effects,
            "layer_effects_field_provenance": {
                str(e["layer"]): "reference" for e in layer_effects
            },
            "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
        }
