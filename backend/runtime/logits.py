"""Intermediate Logits Engine.

Extracts intermediate logit distributions and residual stream projections after every transformer layer
by projecting layer residual vectors through the unembedding matrix W_U (Logit Lens).
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional


class IntermediateLogitsEngine:
    """Extracts layer-by-layer logit projections and unembedding distributions."""

    def extract_logits(
        self,
        prompt: str,
        num_layers: int = 12,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        if not prompt or not str(prompt).strip():
            raise ValueError("Prompt cannot be empty or whitespace.")

        num_layers = max(0, int(num_layers))
        top_k = max(1, min(int(top_k), 50257))

        # Require real model execution
        try:
            import backend.services.gpt2_engine as gpt2_engine
            gpt2_engine.load()
            if not gpt2_engine.is_available() or gpt2_engine._model is None or gpt2_engine._tokenizer is None:
                raise RuntimeError("Live model is uninitialized. Logits extraction requires live PyTorch model execution.")

            _model = gpt2_engine._model
            _tokenizer = gpt2_engine._tokenizer
            import torch
            inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
            with torch.no_grad():
                outputs = _model(**inputs, output_hidden_states=True)

            hidden_states = outputs.hidden_states  # Tuple of [1, seq_len, d_model]
            n_layers = len(hidden_states) - 1
            layers_data: List[Dict[str, Any]] = []

            # Final layer norm and lm_head weights
            ln_f = getattr(_model.transformer, "ln_f", None)
            lm_head = getattr(_model, "lm_head", None)

            for layer in range(min(num_layers, n_layers + 1)):
                h = hidden_states[layer][0, -1, :]  # [d_model]
                norm_val = float(torch.norm(h).item())

                # Apply unembedding projection
                if ln_f is not None:
                    h_norm = ln_f(h.unsqueeze(0))
                else:
                    h_norm = h.unsqueeze(0)

                if lm_head is not None:
                    layer_logits = lm_head(h_norm)[0]  # [vocab_size]
                else:
                    # Tie weights with wte if lm_head not separate
                    wte = _model.transformer.wte.weight
                    layer_logits = torch.matmul(h_norm, wte.T)[0]

                probs = torch.softmax(layer_logits, dim=-1)
                top_k_res = torch.topk(probs, k=top_k)
                
                # Compute Shannon entropy
                log_probs = torch.log(probs + 1e-12)
                entropy_val = float((-torch.sum(probs * log_probs)).item())

                top_tokens = []
                for idx, prob in zip(top_k_res.indices.tolist(), top_k_res.values.tolist()):
                    tok_str = _tokenizer.decode([idx])
                    logit_val = float(layer_logits[idx].item())
                    top_tokens.append({
                        "token": tok_str,
                        "logit": round(logit_val, 3),
                        "probability": round(float(prob), 4),
                    })

                layers_data.append({
                    "layer": layer,
                    "residual_norm": round(norm_val, 2),
                    "top_prediction": top_tokens[0]["token"],
                    "top_logit": top_tokens[0]["logit"],
                    "entropy": round(entropy_val, 2),
                    "top_k_tokens": top_tokens,
                })

            return {
                "prompt": prompt,
                "total_layers": len(layers_data),
                "layer_projections": layers_data,
            }
        except Exception as exc:
            raise RuntimeError(f"Intermediate logits projection failed: {exc}. Synthetic fallback generation is prohibited.") from exc

