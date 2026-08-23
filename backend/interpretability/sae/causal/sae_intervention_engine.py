r"""SAE Causal Intervention & Feature Steering Engine for MECH.

Enables causal manipulation of language model hidden states using Sparse Autoencoder
feature directions, measuring observed logit shifts against linear DLA predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch

from ..sae_interface import SAEInterface
from ..attribution.sae_dla import SAEDirectLogitAttributor


@dataclass
class SAECausalInterventionResult:
    """Quantitative results of a causal feature steering intervention on GPT-2."""
    feature_idx: int
    layer: int
    prompt: str
    target_token: str
    steering_coefficient: float  # alpha
    baseline_target_logit: float
    steered_target_logit: float
    observed_logit_shift: float
    predicted_logit_shift: float
    baseline_target_probability: float
    steered_target_probability: float
    causal_faithfulness_ratio: float
    is_causally_effective: bool
    summary: str
    evidence_type: str = "INTERVENTIONAL_CAUSAL_STEERING"
    provenance: str = "COMPUTED_RUNTIME"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_idx": self.feature_idx,
            "layer": self.layer,
            "prompt": self.prompt,
            "target_token": self.target_token,
            "steering_coefficient": round(self.steering_coefficient, 4),
            "baseline_target_logit": round(self.baseline_target_logit, 4),
            "steered_target_logit": round(self.steered_target_logit, 4),
            "observed_logit_shift": round(self.observed_logit_shift, 4),
            "predicted_logit_shift": round(self.predicted_logit_shift, 4),
            "baseline_target_probability": round(self.baseline_target_probability, 6),
            "steered_target_probability": round(self.steered_target_probability, 6),
            "causal_faithfulness_ratio": round(self.causal_faithfulness_ratio, 4),
            "is_causally_effective": self.is_causally_effective,
            "evidence_type": self.evidence_type,
            "provenance": self.provenance,
            "summary": self.summary,
        }


class SAECausalInterventionEngine:
    """Executes live causal interventions on transformer hidden states via SAE directions."""

    def __init__(self, runtime=None) -> None:
        self.runtime = runtime

    def steer_feature(
        self,
        sae: SAEInterface,
        feature_idx: int,
        prompt: str,
        target_token: str,
        steering_coefficient: float = 2.0,
        layer: Optional[int] = None,
    ) -> SAECausalInterventionResult:
        """Injects alpha * d_i into GPT-2 residual stream at the specified layer."""
        if self.runtime is None:
            raise ValueError("SAECausalInterventionEngine requires a live runtime (InMemoryRuntime).")

        target_layer = layer if layer is not None else sae.metadata.layer
        d_i = sae.get_feature_direction(feature_idx).to(self.runtime.device).float()  # [d_in]

        # 1. Measure baseline forward pass
        fwd_base = self.runtime.forward(prompt, target_token=target_token)
        base_logit = fwd_base.target_logit or 0.0
        base_prob = fwd_base.target_probability or 0.0

        # 2. Predicted shift from DLA
        lm_head = self.runtime.adapter.get_lm_head(self.runtime.model)
        w_u = lm_head.weight.data.float()  # [vocab_size, d_in]
        attributor = SAEDirectLogitAttributor(unembedding_matrix=w_u, tokenizer=self.runtime.tokenizer)
        dla_res = attributor.attribute_feature(sae, feature_idx, target_token=target_token)
        pred_unit_shift = dla_res.target_token_logit_boost or 0.0
        pred_total_shift = steering_coefficient * pred_unit_shift

        # 3. Intervene with forward hook
        tok_out = self.runtime.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.runtime.device) if hasattr(v, "to") else v for k, v in tok_out.items()}

        def steering_hook(module, input_tensor, output_tensor):
            # output_tensor is tuple (hidden_states, ...) or hidden_states tensor
            if isinstance(output_tensor, tuple):
                h = output_tensor[0]
                delta = (steering_coefficient * d_i).to(dtype=h.dtype, device=h.device)
                # Inject perturbation at final token position across batch
                h[:, -1, :] = h[:, -1, :] + delta
                return (h,) + output_tensor[1:]
            else:
                delta = (steering_coefficient * d_i).to(dtype=output_tensor.dtype, device=output_tensor.device)
                output_tensor[:, -1, :] = output_tensor[:, -1, :] + delta
                return output_tensor

        # Register hook on target layer residual block
        hook_handle = self.runtime.model.transformer.h[target_layer].register_forward_hook(steering_hook)
        try:
            with torch.no_grad():
                out = self.runtime.model(**inputs)
            last_logits = out.logits[0, -1]
            t_ids = self.runtime.tokenizer.encode(target_token)
            t_id = t_ids[-1] if t_ids else 0
            steered_logit = float(last_logits[t_id].item())
            probs = torch.softmax(last_logits, dim=-1)
            steered_prob = float(probs[t_id].item())
        finally:
            hook_handle.remove()

        obs_shift = steered_logit - base_logit
        # Causal faithfulness ratio: agreement between observed and predicted
        if abs(pred_total_shift) > 1e-4:
            cfr = obs_shift / pred_total_shift
        else:
            cfr = 1.0 if abs(obs_shift) < 0.1 else 0.5

        is_effective = (steering_coefficient > 0 and obs_shift > 0) or (steering_coefficient < 0 and obs_shift < 0)

        summary = (
            f"Steering SAE Feature #{feature_idx} by alpha={steering_coefficient:+.2f} shifted target '{target_token}' "
            f"logit from {base_logit:.2f} -> {steered_logit:.2f} (delta={obs_shift:+.2f}, predicted={pred_total_shift:+.2f}, "
            f"CFR={cfr:.3f}, prob: {base_prob:.3f} -> {steered_prob:.3f})."
        )

        return SAECausalInterventionResult(
            feature_idx=feature_idx,
            layer=target_layer,
            prompt=prompt,
            target_token=target_token,
            steering_coefficient=steering_coefficient,
            baseline_target_logit=base_logit,
            steered_target_logit=steered_logit,
            observed_logit_shift=obs_shift,
            predicted_logit_shift=pred_total_shift,
            baseline_target_probability=base_prob,
            steered_target_probability=steered_prob,
            causal_faithfulness_ratio=cfr,
            is_causally_effective=is_effective,
            summary=summary,
            evidence_type="INTERVENTIONAL_CAUSAL_STEERING",
            provenance="COMPUTED_RUNTIME",
        )
