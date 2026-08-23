r"""Feature Sparsity & Dead-Neuron Tracker for MECH SAEs.

Computes L0 norm, L1 norm, and tracks dead (never-firing) features across evaluation steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Set
import torch

from ..sae_interface import SAEInterface


@dataclass
class SparsityProfile:
    """Snapshot of sparsity metrics across a batch or dataset."""
    mean_l0: float
    mean_l1: float
    active_feature_ratio: float
    total_features: int
    dead_features_count: int
    dead_features_fraction: float
    dead_feature_indices: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_l0": round(self.mean_l0, 2),
            "mean_l1": round(self.mean_l1, 4),
            "active_feature_ratio": round(self.active_feature_ratio, 6),
            "total_features": self.total_features,
            "dead_features_count": self.dead_features_count,
            "dead_features_fraction": round(self.dead_features_fraction, 4),
            "dead_feature_indices": self.dead_feature_indices[:50],  # Truncate for display
        }


class FeatureSparsityTracker:
    """Tracks L0, L1 norms and dead features over lifetime execution."""

    def __init__(self, sae: SAEInterface) -> None:
        self.sae = sae
        self.d_sae = sae.metadata.d_sae
        self._lifetime_firing_mask = torch.zeros(self.d_sae, dtype=torch.bool)
        self._history_l0: List[float] = []
        self._history_l1: List[float] = []

    def update(self, z: torch.Tensor) -> Dict[str, float]:
        """Updates sparsity statistics from latent activations z."""
        z_flat = z.view(-1, self.d_sae).float()
        active = z_flat > 1e-4

        # Track lifetime active features
        batch_active = active.any(dim=0).cpu()
        self._lifetime_firing_mask = self._lifetime_firing_mask | batch_active

        l0 = float(active.sum(dim=-1).float().mean().item())
        l1 = float(z_flat.abs().sum(dim=-1).mean().item())
        self._history_l0.append(l0)
        self._history_l1.append(l1)

        return {"l0": l0, "l1": l1}

    def get_sparsity_profile(self) -> SparsityProfile:
        """Returns the full sparsity and dead feature profile."""
        dead_mask = ~self._lifetime_firing_mask
        dead_indices = torch.nonzero(dead_mask).flatten().tolist()
        dead_count = len(dead_indices)

        mean_l0 = sum(self._history_l0) / max(1, len(self._history_l0)) if self._history_l0 else 0.0
        mean_l1 = sum(self._history_l1) / max(1, len(self._history_l1)) if self._history_l1 else 0.0
        active_ratio = mean_l0 / max(1, self.d_sae)
        dead_fraction = dead_count / max(1, self.d_sae)

        return SparsityProfile(
            mean_l0=mean_l0,
            mean_l1=mean_l1,
            active_feature_ratio=active_ratio,
            total_features=self.d_sae,
            dead_features_count=dead_count,
            dead_features_fraction=dead_fraction,
            dead_feature_indices=dead_indices,
        )
