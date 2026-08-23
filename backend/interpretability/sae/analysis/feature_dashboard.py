r"""Feature Dashboard & Scientific Inspection Engine for MECH SAEs.

Generates unified scientific reports combining reconstruction fidelity, active latents,
Direct Logit Attribution, and mechanistic hypotheses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch

from ..sae_interface import SAEInterface, FeatureActivationSummary
from ..attribution.sae_dla import SAEDirectLogitAttributor


class FeatureDashboardEngine:
    """Produces structured dashboard summaries for UI and API consumers."""

    def __init__(
        self,
        sae: SAEInterface,
        unembedding_matrix: Optional[torch.Tensor] = None,
        tokenizer: Any = None,
    ) -> None:
        self.sae = sae
        self.attributor = (
            SAEDirectLogitAttributor(unembedding_matrix=unembedding_matrix, tokenizer=tokenizer)
            if (unembedding_matrix is not None and tokenizer is not None)
            else None
        )

    def inspect_prompt(
        self,
        hidden_state: torch.Tensor,
        prompt: str,
        top_k_features: int = 16,
        target_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Runs full SAE decomposition, reconstruction metrics, and DLA attribution for a prompt."""
        z, x_hat = self.sae.reconstruct(hidden_state)
        rec_metrics = self.sae.get_reconstruction_error(hidden_state, x_hat)
        sparsity_metrics = self.sae.get_sparsity(z)

        # Identify active features
        z_flat = z.view(-1, self.sae.metadata.d_sae).squeeze(0).float()
        top_vals, top_idx = torch.topk(z_flat, k=min(top_k_features, self.sae.metadata.d_sae))

        features_summary = []
        for val, idx in zip(top_vals, top_idx):
            f_idx = idx.item()
            act_val = float(val.item())
            if act_val <= 1e-4:
                continue

            summary = FeatureActivationSummary(
                feature_idx=f_idx,
                activation=act_val,
                l0_contribution=1.0,
            )

            # Attribute vocabulary projection if attributor is present
            if self.attributor is not None:
                dla = self.attributor.attribute_feature(
                    sae=self.sae,
                    feature_idx=f_idx,
                    top_k=5,
                    target_token=target_token,
                )
                summary.top_positive_tokens = dla.top_positive_tokens
                summary.top_negative_tokens = dla.top_negative_tokens
                summary.direct_logit_boost = dla.target_token_logit_boost
                pos_label = dla.top_positive_tokens[0][0].strip() if dla.top_positive_tokens else f"F{f_idx}"
                summary.label = f"SAE_L{self.sae.metadata.layer}_F{f_idx} ({pos_label})"
            else:
                summary.label = f"SAE_L{self.sae.metadata.layer}_F{f_idx}"

            features_summary.append(summary.to_dict())

        return {
            "metadata": self.sae.metadata.to_dict(),
            "prompt": prompt,
            "reconstruction": rec_metrics,
            "sparsity": sparsity_metrics,
            "active_features": features_summary,
            "active_feature_count": len(features_summary),
        }
