"""Intermediate Logits Engine.

Extracts layer-by-layer logit distributions and residual-stream projections
after every transformer block.

What this used to be
--------------------
It never loaded a model. For every layer it returned four linear functions of the
layer index:

    "residual_norm":  10.0 + layer * 1.5
    "top_prediction": " Paris" if layer >= 6 else words[0]
    "top_logit":      4.5 + layer * 0.8
    "entropy":        2.5 - layer * 0.15

No forward pass, no weights, no randomness -- and, decisively, **no provenance
field**. A consumer received a dict that looked like a measurement with nothing
to tell it otherwise, and `ExecutionEngine.get_intermediate_logits` published it
directly.

So "the residual norm grows by 1.5 per layer" and "the top prediction flips to
Paris at layer 6" were statements about arithmetic, presented as statements about
GPT-2.

What it is now
--------------
Every field is computed from the hidden states of a real forward pass over loaded
GPT-2 weights: `ln_f` then the tied unembedding, exactly as the untuned logit lens
does. `residual_norm` is the actual L2 norm of the residual stream at that layer.
`top_logit` and `entropy` come from the real logit vector.

If the weights cannot be loaded, every field is `None`, `provenance` is
`unavailable`, and the reason is stated. There is no arithmetic fallback -- that
fallback *was* the defect, and keeping it "for robustness" would preserve it.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

LIVE = "live"
UNAVAILABLE = "unavailable"


class IntermediateLogitsEngine:
    """Layer-by-layer logit projections over live GPT-2 weights."""

    def __init__(self, engine: Optional[Any] = None) -> None:
        # Injectable so tests can drive the real code path without a GPU.
        from backend.services import gpt2_engine

        self.engine = engine or gpt2_engine

    # ── public API ───────────────────────────────────────────────────────

    def extract_logits(self, prompt: str, num_layers: int = 12,
                       top_k: int = 5) -> Dict[str, Any]:
        """Extract intermediate logit projections per layer.

        `num_layers` bounds how many layers are reported; the model may have
        more. Every value comes from one cached forward pass.
        """
        layers = self._layer_projections(prompt, num_layers, top_k)
        if layers is None:
            return self._unavailable(prompt, num_layers)

        return {
            "prompt": prompt,
            "total_layers": len(layers),
            "requested_layers": num_layers,
            "layer_projections": layers,
            "status": "ok",
            "provenance": LIVE,
            "field_provenance": {
                field: LIVE for field in
                ("layer", "residual_norm", "top_prediction", "top_logit",
                 "entropy", "top_k_tokens")
            },
            "model_id": getattr(self.engine, "MODEL_ID", None),
            "method": "IntermediateLogits",
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": (
                "Projected from live GPT-2 weights via a real forward pass. "
                "A single prompt's per-layer projection is not evidence of a "
                "mechanism, and no layer is identified as causal by this."
            ),
        }

    # ── internals ────────────────────────────────────────────────────────

    def _unavailable(self, prompt: str, num_layers: int) -> Dict[str, Any]:
        reason = self._load_failure()
        return {
            "prompt": prompt,
            "total_layers": 0,
            "requested_layers": num_layers,
            "layer_projections": [],
            "status": UNAVAILABLE,
            "provenance": UNAVAILABLE,
            "field_provenance": {
                field: UNAVAILABLE for field in
                ("layer", "residual_norm", "top_prediction", "top_logit",
                 "entropy", "top_k_tokens")
            },
            "method": "IntermediateLogits",
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": reason,
        }

    def _load_failure(self) -> str:
        loader = getattr(self.engine, "_ensure_loaded", None)
        if loader is None:
            return ("The intermediate-logits engine needs a loaded model to "
                    "project anything from. No values are reported.")
        error = loader()
        if error:
            return (f"{error.get('reason', 'The model could not be loaded.')} "
                    f"No layer projection was performed and no values are "
                    f"reported.")
        return ("The model reported as loaded but exposed no hidden states, so "
                "no layer projection could be computed.")

    def _layer_projections(self, prompt: str, num_layers: int,
                           top_k: int) -> Optional[List[Dict[str, Any]]]:
        torch = self._torch()
        if torch is None:
            return None

        loader = getattr(self.engine, "_ensure_prompt", None)
        if loader is not None:
            error = loader(prompt)
            if error:
                return None

        model = getattr(self.engine, "_model", None)
        hidden = getattr(self.engine, "_cache", {}).get("hidden")
        if model is None or not hidden:
            return None

        device = next(model.parameters()).device
        unembed = model.transformer.wte.weight
        n_model_layers = len(hidden) - 1
        want = max(0, min(int(num_layers), n_model_layers))
        k = max(1, min(int(top_k), int(unembed.shape[0])))

        out: List[Dict[str, Any]] = []
        for index in range(want):
            state = torch.from_numpy(hidden[index + 1][-1]).float().to(device)
            with torch.no_grad():
                normed = model.transformer.ln_f(state)
                logits = normed @ unembed.T

            probs = torch.softmax(logits, dim=-1)
            top = torch.topk(probs, k)
            tokens = [
                {"token": self._decode(int(i)), "probability": round(float(probs[i]), 6)}
                for i in top.indices
            ]

            # Real entropy of the projected distribution, in nats.
            log_probs = torch.log_softmax(logits, dim=-1)
            entropy = float(-(probs * log_probs).sum())

            out.append({
                "layer": index,
                # The actual L2 norm of this layer's residual stream. Previously
                # `10.0 + layer * 1.5`.
                "residual_norm": round(float(state.norm()), 4),
                "top_prediction": tokens[0]["token"] if tokens else None,
                "top_logit": round(float(logits.max()), 4),
                "entropy": round(entropy, 4),
                "top_k_tokens": tokens,
                "top_probability": tokens[0]["probability"] if tokens else None,
            })
        return out

    def _decode(self, token_id: int) -> str:
        decoder = getattr(self.engine, "_decode", None)
        if callable(decoder):
            return decoder(token_id)
        tokenizer = getattr(self.engine, "_tokenizer", None)
        return tokenizer.decode([token_id]) if tokenizer is not None else str(token_id)

    @staticmethod
    def _torch():
        try:
            import torch  # noqa: PLC0415
        except Exception:
            return None
        return torch