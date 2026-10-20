"""Causal Tracing Engine.

NOT IMPLEMENTED. ``trace_causal_effect`` synthesises a layer-effect curve
from a prompt-dependent expression and reports ``max_causal_layer``.
No model is loaded, no clean/corrupted run is performed, and no residual
stream is ablation-swept.

The clean/corrupted prompts were echoed back into the response, which made the
result look like the supplied prompts had been run. The fixture is retained
for inspection but is now labelled and ineligible.

A real implementation: for each layer, zero the clean residual stream, restore
the corrupted one, and measure the change in the (Paris - Rome) logit
difference. The live dispatcher route (`interpretability/causal/trace`) already
does exactly this when GPT-2 is loaded.

The closed form
---------------

``layer_effects`` is drawn from four constants. Nothing is measured:

    pair    = json.dumps({"clean": clean, "corrupt": corrupt}, sort_keys=True)
    digest  = int(sha256(pair).hexdigest(), 16)
    byte    = (digest >> (8 * layer)) & 0xFF       # one byte per layer
    indirect_effect = 0.90                        for layers 8 and 9
                     = 0.12 + 0.05 * (byte % 3)   for every other layer

so every value is one of 0.12, 0.17, 0.22 or 0.90. The curve is a function of
the prompt pair and of nothing else -- no model is loaded, no tensor is read.

This is documented rather than left implicit because the shape of the curve
is a *pin*: `tests/pytest/test_causal_tracing_refusal.py` asserts that the
returned values equal the expression above and lie on the four-value grid. A
real ablation sweep measures twelve layers and returns twelve arbitrary
numbers, so it cannot satisfy either assertion -- which is the point. The
test fails the day the module becomes real, and the person who implements it
has to replace the test deliberately rather than discover that a synthetic
curve had quietly become a measurement.
"""

from __future__ import annotations

import hashlib
import json as _json
import datetime as _dt
from typing import Any, Dict, List


class CausalTracingEngine:
    """Reference fixture only. No model is loaded and nothing is ablated."""

    def trace_causal_effect(
        self,
        clean_prompt: str = "The capital of France is",
        corrupted_prompt: str = "The capital of Rome is",
    ) -> Dict[str, Any]:
        # Derive prompt-dependent values so that different prompt pairs
        # produce different layer effects (after _strip_volatile removes
        # echoed inputs and volatile fields).
        prompt_pair_json = _json.dumps(
            {"clean": clean_prompt, "corrupt": corrupted_prompt},
            sort_keys=True,
        )
        prompt_hash = int(
            hashlib.sha256(prompt_pair_json.encode()).hexdigest(),
            16,
        )

        layer_effects: List[Dict[str, Any]] = []
        for l in range(12):
            # Use a different byte of the hash for each layer, so that
            # all 12 layers are affected uniquely by the prompt pair.
            byte_idx = (prompt_hash >> (l * 8)) & 0xFF
            # For layers 8 and 9, use a special component; for others,
            # vary the indirect_effect based on the prompt.
            if l in [8, 9]:
                top_causal = "MLP"
                # Vary the indirect_effect using the layer-specific byte
                indirect_effect = round(0.12 + 0.78, 3)  # fixed 0.90 for MLP layers
            else:
                top_causal = "Attention"
                # Use the byte to get a value in {0.12, 0.17, 0.22}
                val_idx = byte_idx % 3
                indirect_effect = round(0.12 + 0.05 * val_idx, 3)

            layer_effects.append({
                "layer": l,
                "indirect_effect": indirect_effect,
                "top_causal_component": top_causal,
                # Per-entry, not per-list-index.
                #
                # The response marks the *list* by index, which says "row 3
                # came from the reference curve" and nothing about the fields
                # inside row 3. A caller who does
                # `trace["layer_effects"][3]["indirect_effect"]` -- the
                # natural way to read it -- gets a bare float with no marker
                # attached. 14 numbers in this response were unmarked that
                # way.
                #
                # Carried inside the entry, the marking survives extraction,
                # which is the only property that matters for a field someone
                # will quote.
                "field_provenance": {
                    "indirect_effect": "reference",
                    "top_causal_component": "reference",
                },
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
            # The two headline numbers are synthesised -- `max_causal_layer` is
            # the literal 8, not the argmax of anything measured -- and were
            # returned bare while `layer_effects` carried a provenance map.
            # A consumer reading either number had nothing to tell it apart
            # from a measurement, which is the specific failure an honest
            # `status: unavailable` does not cover.
            #
            # A dict keyed by field name, matching `evidence_policy.field_map`
            # and the other producers in this codebase. A bare
            # `<field>_field_provenance: "reference"` string was tried first
            # and is wrong: it records one value with nothing to associate it
            # with.
            "headline_field_provenance": {
                "max_causal_layer": "reference",
                "max_indirect_effect": "reference",
            },
            "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
        }
