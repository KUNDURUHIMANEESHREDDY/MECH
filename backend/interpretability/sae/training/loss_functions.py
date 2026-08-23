r"""Loss Functions for Sparse Autoencoder Training.

Supports Standard ReLU with L1 penalty, Top-K SAE, and Gated SAE loss calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch

from ..sae_interface import SAEArchitectureType


@dataclass
class SAELossOutput:
    """Breakdown of SAE loss components."""
    total_loss: torch.Tensor
    reconstruction_loss: float
    sparsity_loss: float
    l0_norm: float
    l1_norm: float
    explained_variance: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_loss": round(float(self.total_loss.item()), 6),
            "reconstruction_loss": round(self.reconstruction_loss, 6),
            "sparsity_loss": round(self.sparsity_loss, 6),
            "l0_norm": round(self.l0_norm, 2),
            "l1_norm": round(self.l1_norm, 4),
            "explained_variance": round(self.explained_variance, 4),
        }


class SAELossCalculator:
    """Computes training loss and metric diagnostics across SAE architectures."""

    def __init__(
        self,
        architecture: SAEArchitectureType = SAEArchitectureType.STANDARD_RELU,
        l1_coefficient: float = 1e-3,
    ) -> None:
        self.architecture = architecture
        self.l1_coefficient = l1_coefficient

    def compute_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        z: torch.Tensor,
    ) -> SAELossOutput:
        """Calculates total loss = MSE(x, x_hat) + l1_coeff * ||z||_1."""
        x_flat = x.view(-1, x.shape[-1]).float()
        x_hat_flat = x_hat.view(-1, x_hat.shape[-1]).float()
        z_flat = z.view(-1, z.shape[-1]).float()

        # Reconstruction MSE
        rec_loss = torch.mean((x_flat - x_hat_flat) ** 2)

        # Sparsity penalty
        if self.architecture == SAEArchitectureType.TOP_K:
            sparsity_loss = torch.tensor(0.0, device=x.device)
        else:
            sparsity_loss = self.l1_coefficient * torch.mean(torch.sum(torch.abs(z_flat), dim=-1))

        total_loss = rec_loss + sparsity_loss

        # Diagnostic metrics
        with torch.no_grad():
            l0 = float((z_flat > 1e-4).sum(dim=-1).float().mean().item())
            l1 = float(torch.abs(z_flat).sum(dim=-1).mean().item())
            var_tot = float(torch.var(x_flat).item()) + 1e-8
            exp_var = max(0.0, 1.0 - (rec_loss.item() / var_tot))

        return SAELossOutput(
            total_loss=total_loss,
            reconstruction_loss=float(rec_loss.item()),
            sparsity_loss=float(sparsity_loss.item()),
            l0_norm=l0,
            l1_norm=l1,
            explained_variance=exp_var,
        )
