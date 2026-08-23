"""Reusable Logit Lens Temporal Engine for MECH.

Computes real layer-by-layer vocabulary prediction trajectories (LayerNorm + Unembedding)
using loaded PyTorch transformer weights (GPT-2) without hardcoded values.
Exposes multi-phase belief dynamics (emergence, maximum gain, stabilization, suppression).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
from backend.science.scientific_data_model import (
    FeatureEvidence,
    LogitLensTransition,
    TransitionType,
)


class LogitLensEngine:
    """Computes and tracks layer-by-layer prediction emergence across Transformer depth."""

    def __init__(self, model_id: str = "gpt2") -> None:
        self.model_id = model_id
        self.num_layers = 12

    def compute_trajectory(
        self,
        prompt: str,
        target_token: Optional[str] = None,
        distractor_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute real layer-by-layer prediction trajectory and identify predictive transitions."""
        transitions: List[LogitLensTransition] = []
        max_gain_layer = 8

        target_probs: List[float] = []
        target_ranks: List[int] = []
        target_logits: List[float] = []

        try:
            import torch
            import backend.services.gpt2_engine as gpt2_engine

            gpt2_engine.load()
            model = gpt2_engine._model
            tokenizer = gpt2_engine._tokenizer

            if model is not None and tokenizer is not None:
                self.num_layers = len(model.transformer.h)
                inputs = tokenizer(prompt or "The capital of France is", return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)

                hidden_states = outputs.hidden_states
                ln_f = model.transformer.ln_f
                lm_head = model.lm_head

                # Resolve target token ID dynamically from prompt / tokenizer
                target_id = None
                if target_token:
                    t_ids = tokenizer.encode(target_token)
                    if len(t_ids) > 0:
                        target_id = t_ids[0]

                prev_target_p = 0.0
                max_delta = -1.0
                detected_gain_layer = 0

                for l, h_layer in enumerate(hidden_states):
                    h_last = h_layer[0, -1, :]
                    h_norm = ln_f(h_last.unsqueeze(0))
                    logits = lm_head(h_norm)[0]
                    probs = torch.softmax(logits, dim=-1)

                    topk = torch.topk(probs, k=4)
                    preds = []
                    for idx, p in zip(topk.indices, topk.values):
                        tok_str = tokenizer.decode([idx])
                        preds.append({
                            "token": tok_str,
                            "probability": round(float(p.detach().item()), 4),
                            "logit": round(float(logits[idx].detach().item()), 2),
                        })

                    top_tok = preds[0]["token"]
                    top_p = preds[0]["probability"]

                    if target_id is not None:
                        curr_target_p = float(probs[target_id].detach().item())
                        curr_target_z = float(logits[target_id].detach().item())
                        curr_target_r = int((torch.sum(logits > logits[target_id]) + 1).item())
                    else:
                        curr_target_p = top_p
                        curr_target_z = float(logits[topk.indices[0]].detach().item())
                        curr_target_r = 1

                    delta_p = curr_target_p - prev_target_p
                    target_probs.append(round(curr_target_p, 4))
                    target_ranks.append(curr_target_r)
                    target_logits.append(round(curr_target_z, 2))

                    if delta_p > max_delta and l > 0:
                        max_delta = delta_p
                        detected_gain_layer = l

                    prev_target_p = curr_target_p

                    # Dynamic Multi-Phase Transition Classification
                    if l == 0:
                        t_type = TransitionType.STABLE
                    elif curr_target_p > 0.15 and (l > 0 and target_probs[l - 1] < 0.10):
                        t_type = TransitionType.EMERGENCE
                    elif delta_p > 0.20:
                        t_type = TransitionType.DIVERGENCE
                    elif curr_target_p > 0.70:
                        t_type = TransitionType.STABILIZATION
                    elif delta_p < -0.15:
                        t_type = TransitionType.SUPPRESSION
                    else:
                        t_type = TransitionType.STABLE

                    # Compute real linear directional projection for top active neuron at this layer
                    active_projs = []
                    if l < len(model.transformer.h):
                        c_proj = model.transformer.h[l].mlp.c_proj.weight  # [d_mlp, d_model]
                        c_fc = model.transformer.h[l].mlp.c_fc.weight  # [d_model, d_mlp]
                        h_in = h_last
                        mlp_acts = torch.nn.functional.gelu(torch.matmul(h_in, c_fc))
                        top_neuron = int(torch.argmax(mlp_acts).item())
                        act_val = float(mlp_acts[top_neuron].detach().item())

                        d_i = c_proj[top_neuron, :]
                        delta_logits = lm_head(ln_f(d_i.unsqueeze(0)))[0]
                        top_boost = torch.topk(delta_logits, k=2)
                        for b_idx in top_boost.indices:
                            active_projs.append({
                                "feature_id": f"L{l}_N{top_neuron}",
                                "activation": round(act_val, 2),
                                "target_token": tokenizer.decode([b_idx]),
                                "projected_delta_logit": round(float(delta_logits[b_idx].detach().item()), 2),
                            })

                    transitions.append(LogitLensTransition(
                        layer=l,
                        top_token=top_tok,
                        probability=round(top_p, 4),
                        delta_probability=round(delta_p, 4),
                        rank=curr_target_r,
                        is_predictive_transition=False,
                        transition_type=t_type,
                        predictions=preds,
                        active_feature_projections=active_projs,
                    ))

                max_gain_layer = detected_gain_layer if detected_gain_layer > 0 else max(1, len(hidden_states) // 2)
                if max_gain_layer < len(transitions):
                    transitions[max_gain_layer].is_predictive_transition = True
                    transitions[max_gain_layer].transition_type = TransitionType.DIVERGENCE

                return {
                    "model_id": self.model_id,
                    "prompt": prompt,
                    "target_token": target_token or transitions[-1].top_token,
                    "maximum_predictive_gain_layer": max_gain_layer,
                    "predictive_divergence_layer": max_gain_layer,
                    "max_delta_probability": round(max_delta, 4),
                    "target_probability_trajectory": target_probs,
                    "target_rank_trajectory": target_ranks,
                    "target_logit_trajectory": target_logits,
                    "trajectory": [t.model_dump() for t in transitions],
                }

        except Exception as err:
            import logging
            logging.getLogger("MECH").error("Real Logit Lens computation failed: %s", err)
            raise RuntimeError(f"Live Logit Lens computation failed: {err}") from err

        raise RuntimeError("Model or tokenizer is uninitialized for Logit Lens trajectory computation.")

