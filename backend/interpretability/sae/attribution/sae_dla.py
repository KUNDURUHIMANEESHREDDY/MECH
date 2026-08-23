r"""Direct Logit Attribution (DLA) for Sparse Autoencoder Features.

Projects SAE decoder feature directions through the model's unembedding matrix W_U
to determine which vocabulary tokens the feature promotes or suppresses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch

from ..sae_interface import SAEInterface


@dataclass
class SAEDLAResult:
    """Attribution results for a single SAE feature."""
    feature_idx: int
    top_positive_tokens: List[Tuple[str, float]]
    top_negative_tokens: List[Tuple[str, float]]
    target_token_logit_boost: Optional[float] = None
    distractor_token_logit_suppress: Optional[float] = None
    direct_effect_magnitude: float = 0.0
    is_causal: bool = False
    evidence_type: str = "VIRTUAL_UNEMBEDDED_PROJECTION"
    provenance: str = "COMPUTED_RUNTIME"
    method_limitation: str = (
        "Direct linear projection through unembedding matrix W_U. "
        "Does NOT measure mediated downstream attention/MLP layers or LayerNorm re-scaling."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_idx": self.feature_idx,
            "top_positive_tokens": [(t, round(float(s), 4)) for t, s in self.top_positive_tokens],
            "top_negative_tokens": [(t, round(float(s), 4)) for t, s in self.top_negative_tokens],
            "target_token_logit_boost": round(float(self.target_token_logit_boost), 4) if self.target_token_logit_boost is not None else None,
            "distractor_token_logit_suppress": round(float(self.distractor_token_logit_suppress), 4) if self.distractor_token_logit_suppress is not None else None,
            "direct_effect_magnitude": round(float(self.direct_effect_magnitude), 4),
            "is_causal": self.is_causal,
            "evidence_type": self.evidence_type,
            "provenance": self.provenance,
            "method_limitation": self.method_limitation,
        }


class SAEDirectLogitAttributor:
    """Computes Direct Logit Attribution across SAE feature dictionaries."""

    def __init__(self, unembedding_matrix: torch.Tensor, tokenizer: Any) -> None:
        self.w_u = unembedding_matrix.float()  # [vocab_size, d_in]
        self.tokenizer = tokenizer

    def attribute_feature(
        self,
        sae: SAEInterface,
        feature_idx: int,
        top_k: int = 10,
        target_token: Optional[str] = None,
        distractor_token: Optional[str] = None,
    ) -> SAEDLAResult:
        """Projects feature direction vector through W_U."""
        d_i = sae.get_feature_direction(feature_idx).to(self.w_u.device).float()  # [d_in]
        if torch.all(d_i == 0) or torch.isnan(d_i).any() or float(torch.norm(d_i).item()) == 0.0:
            raise ValueError(
                f"Feature #{feature_idx} direction vector has zero norm or NaN. Cannot compute valid DLA attribution."
            )

        if d_i.ndim == 1:
            d_i = d_i.unsqueeze(0)

        with torch.no_grad():
            logits = torch.matmul(d_i, self.w_u.T).squeeze(0)  # [vocab_size]

        # Top positive tokens
        top_pos_vals, top_pos_idx = torch.topk(logits, k=top_k)
        pos_tokens = [
            (self.tokenizer.decode([idx.item()]), float(val.item()))
            for val, idx in zip(top_pos_vals, top_pos_idx)
        ]

        # Top negative tokens
        top_neg_vals, top_neg_idx = torch.topk(-logits, k=top_k)
        neg_tokens = [
            (self.tokenizer.decode([idx.item()]), float(-val.item()))
            for val, idx in zip(top_neg_vals, top_neg_idx)
        ]

        # Target token boost
        target_boost = None
        if target_token is not None:
            t_ids = self.tokenizer.encode(target_token)
            t_id = t_ids[-1] if t_ids else 0
            if t_id < logits.shape[0]:
                target_boost = float(logits[t_id].item())

        distractor_suppress = None
        if distractor_token is not None:
            d_ids = self.tokenizer.encode(distractor_token)
            d_id = d_ids[-1] if d_ids else 0
            if d_id < logits.shape[0]:
                distractor_suppress = float(logits[d_id].item())

        magnitude = float(torch.norm(logits).item())

        return SAEDLAResult(
            feature_idx=feature_idx,
            top_positive_tokens=pos_tokens,
            top_negative_tokens=neg_tokens,
            target_token_logit_boost=target_boost,
            distractor_token_logit_suppress=distractor_suppress,
            direct_effect_magnitude=magnitude,
            is_causal=False,
            evidence_type="VIRTUAL_UNEMBEDDED_PROJECTION",
            provenance="COMPUTED_RUNTIME",
        )
