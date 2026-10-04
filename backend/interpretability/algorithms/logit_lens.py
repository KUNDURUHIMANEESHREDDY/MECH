"""Logit Lens Algorithm.

Applies the direct unembedding projection to an intermediate residual stream.

What changed
------------
This previously delegated to `backend.runtime.logits.IntermediateLogitsEngine`,
which never loaded a model. It returned, for every layer:

    "residual_norm":   10.0 + layer * 1.5
    "top_prediction":  " Paris" if layer >= 6 else words[0]
    "top_logit":       4.5 + layer * 0.8
    "entropy":         2.5 - layer * 0.15

Four linear functions of the layer index, on a dict that carried no provenance
field at all. So the logit lens in this repository measured nothing, and a
consumer had no label telling it so.

On top of that, `project()` appended a second-ranked token that was hardcoded
rather than computed:

    {"token": " France", "logit": top_logit - 1.2, "probability": 0.12}

A fixed token, a fixed offset and a fixed probability, presented as the model's
second guess. That is the "stub that returns 0.94" pattern, in the one place
the README listed as an implemented algorithm.

Now it projects through `gpt2_engine.logit_lens`, which reads the hidden states
of an actual forward pass over real GPT-2 weights, applies `ln_f` and the tied
unembedding, and returns the real top-k with real probabilities. If the weights
cannot be loaded it says so and withholds, rather than falling back to
arithmetic.

`backend/runtime/logits.py` is left in place but is no longer reachable from
here; see the module docstring there.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from backend.services import gpt2_engine

LIVE = "live"
UNAVAILABLE = "unavailable"


class ProjectionModel:
    """Interface for layer projection models (Logit Lens, Tuned Lens)."""

    def project(self, prompt: str, layer: int) -> Dict[str, Any]:
        raise NotImplementedError


class LogitLens(ProjectionModel):
    """Logit Lens unembedding projection over live GPT-2 weights."""

    def __init__(self, engine: Optional[Any] = None) -> None:
        # Injectable so a test can drive the real code path with a stub engine
        # and no GPU. The default is the real engine.
        self.engine = engine or gpt2_engine

    def project(self, prompt: str, layer: int, top_k: int = 5) -> Dict[str, Any]:
        raw = self.engine.logit_lens(layer=layer, prompt=prompt, top_k=top_k)

        # `logit_lens` returns its error dict rather than raising, so an
        # unloadable model arrives here looking like a result. Anything without
        # a real top-k is withheld -- falling back to arithmetic here is exactly
        # the defect this class had.
        if not isinstance(raw, dict) or raw.get("status") != "ok":
            reason = (raw or {}).get("reason") if isinstance(raw, dict) else None
            return {
                "method": "LogitLens",
                "prompt": prompt,
                "layer": layer,
                "status": "unavailable",
                "provenance": UNAVAILABLE,
                "field_provenance": {"status": UNAVAILABLE,
                                     "top_token": UNAVAILABLE,
                                     "top_k_tokens": UNAVAILABLE},
                "validation_eligible": False,
                "publication_eligible": False,
                "top_token": None,
                "top_logit": None,
                "entropy": None,
                "top_k_tokens": None,
                "reason": (
                    reason
                    or "The logit lens needs loaded GPT-2 weights. No "
                       "projection was performed and no values are reported."
                ),
            }

        # `top_k_tokens` is the real ranked list from the forward pass. The
        # fabricated second entry is gone, and nothing is appended to it.
        top_k = raw.get("top_k_tokens") or []
        top = top_k[0] if top_k else {}

        return {
            "method": "LogitLens",
            "prompt": prompt,
            "layer": raw.get("layer", layer),
            "status": "ok",
            "provenance": LIVE,
            "field_provenance": {"status": LIVE, "top_token": LIVE,
                                 "top_k_tokens": LIVE},
            "model_id": getattr(self.engine, "MODEL_ID", None),
            "top_token": top.get("token"),
            # `prob` is what the engine measured. `top_logit` is kept as a
            # separate name because the old payload used it, but it is only
            # populated when the engine supplies it -- never reconstructed.
            "top_logit": raw.get("top_logit"),
            "top_probability": top.get("prob"),
            "entropy": raw.get("entropy"),
            "top_k_tokens": top_k,
            "n_layers": getattr(self.engine, "n_layers", lambda: None)(),
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": (
                "Projection from live GPT-2 weights via a real forward pass. "
                "Not a scientific finding on its own: a single prompt's layer "
                "projection is not evidence of a mechanism."
            ),
        }