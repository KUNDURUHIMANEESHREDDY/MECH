r"""Feature Activation Analyzer for MECH SAEs.

Analyzes activation distributions, firing frequencies, and max-activating contexts
across prompt datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import torch

from ..sae_interface import SAEInterface, FeatureActivationSummary


@dataclass
class FeatureActivationStats:
    """Statistics for a single feature over an evaluation set."""
    feature_idx: int
    firing_count: int
    firing_frequency: float
    max_activation: float
    mean_active_activation: float
    top_contexts: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_idx": self.feature_idx,
            "firing_count": self.firing_count,
            "firing_frequency": round(self.firing_frequency, 6),
            "max_activation": round(self.max_activation, 4),
            "mean_active_activation": round(self.mean_active_activation, 4),
            "top_contexts": self.top_contexts,
        }


class FeatureActivationAnalyzer:
    """Analyzes latent activations across forward passes."""

    def __init__(self, sae: SAEInterface) -> None:
        self.sae = sae
        self.d_sae = sae.metadata.d_sae
        self._firing_counts = torch.zeros(self.d_sae, dtype=torch.long)
        self._max_activations = torch.zeros(self.d_sae, dtype=torch.float32)
        self._sum_activations = torch.zeros(self.d_sae, dtype=torch.float32)
        self._total_tokens = 0

    def record_activations(self, z: torch.Tensor, prompt: Optional[str] = None) -> None:
        """Updates activation statistics from a latent tensor z."""
        z_flat = z.view(-1, self.d_sae).float()
        n_tokens = z_flat.shape[0]
        self._total_tokens += n_tokens

        active_mask = z_flat > 1e-4
        self._firing_counts += active_mask.sum(dim=0).cpu()
        max_vals, _ = z_flat.max(dim=0)
        self._max_activations = torch.maximum(self._max_activations, max_vals.cpu())
        self._sum_activations += z_flat.sum(dim=0).cpu()

    def get_top_features(self, top_k: int = 20) -> List[FeatureActivationStats]:
        """Returns the most frequently / strongly firing features."""
        freqs = self._firing_counts.float() / max(1, self._total_tokens)
        top_vals, top_idx = torch.topk(self._max_activations, k=min(top_k, self.d_sae))

        stats = []
        for val, idx in zip(top_vals, top_idx):
            f_idx = idx.item()
            count = self._firing_counts[f_idx].item()
            freq = freqs[f_idx].item()
            mean_act = (self._sum_activations[f_idx].item() / count) if count > 0 else 0.0

            stats.append(FeatureActivationStats(
                feature_idx=f_idx,
                firing_count=count,
                firing_frequency=freq,
                max_activation=float(val.item()),
                mean_active_activation=mean_act,
            ))
        return stats
