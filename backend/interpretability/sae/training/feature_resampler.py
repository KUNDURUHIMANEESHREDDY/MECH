r"""Dead Feature Resampler for Sparse Autoencoder Training.

Implements Anthropic-style neuron resampling for dead features during SAE training.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import torch

from ..sae_adapter import NativeMECHSAE


class DeadFeatureResampler:
    """Detects inactive latents and resamples weights using high-reconstruction-error inputs."""

    def __init__(
        self,
        dead_steps_threshold: int = 1000,
        resample_scale: float = 0.2,
    ) -> None:
        self.dead_steps_threshold = dead_steps_threshold
        self.resample_scale = resample_scale
        self._steps_since_active: Optional[torch.Tensor] = None

    def update_firing_history(self, z: torch.Tensor) -> torch.Tensor:
        """Updates counter for steps since each latent feature fired."""
        z_flat = z.view(-1, z.shape[-1]).float()
        d_sae = z_flat.shape[-1]

        if self._steps_since_active is None or self._steps_since_active.shape[0] != d_sae:
            self._steps_since_active = torch.zeros(d_sae, dtype=torch.long, device=z.device)

        batch_fired = (z_flat > 1e-4).any(dim=0)
        self._steps_since_active[batch_fired] = 0
        self._steps_since_active[~batch_fired] += 1

        return self._steps_since_active

    def get_dead_indices(self) -> List[int]:
        """Returns list of feature indices that haven't fired for dead_steps_threshold steps."""
        if self._steps_since_active is None:
            return []
        dead_mask = self._steps_since_active >= self.dead_steps_threshold
        return torch.nonzero(dead_mask).flatten().tolist()

    def resample_dead_features(
        self,
        sae: NativeMECHSAE,
        high_loss_residuals: torch.Tensor,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> int:
        """Resamples dead features in the direction of high-loss residual vectors."""
        dead_indices = self.get_dead_indices()
        if not dead_indices or high_loss_residuals.shape[0] == 0:
            return 0

        n_resample = min(len(dead_indices), high_loss_residuals.shape[0])
        resample_indices = dead_indices[:n_resample]

        with torch.no_grad():
            res_flat = high_loss_residuals.view(-1, sae.metadata.d_in).float()
            # Sample directions from high loss inputs
            sample_dirs = res_flat[:n_resample]
            norms = torch.norm(sample_dirs, dim=1, keepdim=True) + 1e-8
            unit_dirs = sample_dirs / norms

            # Update encoder weights W_enc: [d_in, d_sae]
            for i, feat_idx in enumerate(resample_indices):
                sae.w_enc[:, feat_idx] = unit_dirs[i] * self.resample_scale
                # Update decoder weights W_dec: [d_sae, d_in]
                sae.w_dec[feat_idx, :] = unit_dirs[i]
                sae.b_enc[feat_idx] = 0.0

            # Reset step counters for resampled features
            if self._steps_since_active is not None:
                for feat_idx in resample_indices:
                    self._steps_since_active[feat_idx] = 0

        return n_resample
