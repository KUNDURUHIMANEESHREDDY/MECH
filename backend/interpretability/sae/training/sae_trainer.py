r"""SAE Trainer for MECH.

Trains Native Sparse Autoencoders on streaming activation batches with
Adam optimization, Top-K/L1 regularization, and dead-feature resampling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch
import torch.optim as optim

from ..sae_interface import SAEArchitectureType, SAEMetadata, SAEBackendSource
from ..sae_adapter import NativeMECHSAE
from .loss_functions import SAELossCalculator, SAELossOutput
from .feature_resampler import DeadFeatureResampler


@dataclass
class SAETrainingConfig:
    """Hyperparameters for SAE training."""
    d_in: int = 768
    d_sae: int = 3072
    architecture: SAEArchitectureType = SAEArchitectureType.STANDARD_RELU
    learning_rate: float = 3e-4
    l1_coefficient: float = 1e-3
    top_k: int = 32
    dead_steps_threshold: int = 500
    resample_freq: int = 1000
    resample_scale: float = 0.2
    normalize_decoder: bool = True


@dataclass
class SAETrainingStepResult:
    """Telemetry from a single SAE training step."""
    step: int
    total_loss: float
    reconstruction_loss: float
    sparsity_loss: float
    l0_norm: float
    l1_norm: float
    explained_variance: float
    dead_features_count: int
    resampled_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "total_loss": round(self.total_loss, 6),
            "reconstruction_loss": round(self.reconstruction_loss, 6),
            "sparsity_loss": round(self.sparsity_loss, 6),
            "l0_norm": round(self.l0_norm, 2),
            "l1_norm": round(self.l1_norm, 4),
            "explained_variance": round(self.explained_variance, 4),
            "dead_features_count": self.dead_features_count,
            "resampled_count": self.resampled_count,
        }


class SAETrainer:
    """Trainer module for training Native MECH Sparse Autoencoders."""

    def __init__(
        self,
        config: SAETrainingConfig,
        sae: Optional[NativeMECHSAE] = None,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.device = device
        self.sae = sae or NativeMECHSAE(
            d_in=config.d_in,
            d_sae=config.d_sae,
            architecture=config.architecture,
        )

        # Make weights trainable parameters
        self.w_enc = torch.nn.Parameter(self.sae.w_enc.clone())
        self.b_enc = torch.nn.Parameter(self.sae.b_enc.clone())
        self.w_dec = torch.nn.Parameter(self.sae.w_dec.clone())
        self.b_dec = torch.nn.Parameter(self.sae.b_dec.clone())

        self.optimizer = optim.Adam(
            [self.w_enc, self.b_enc, self.w_dec, self.b_dec],
            lr=config.learning_rate,
            betas=(0.9, 0.999),
        )

        self.loss_calculator = SAELossCalculator(
            architecture=config.architecture,
            l1_coefficient=config.l1_coefficient,
        )

        self.resampler = DeadFeatureResampler(
            dead_steps_threshold=config.dead_steps_threshold,
            resample_scale=config.resample_scale,
        )

        self.step_count = 0

    def train_step(self, activations_batch: torch.Tensor) -> SAETrainingStepResult:
        """Executes a single forward/backward optimization step on a batch of activations."""
        self.step_count += 1
        x = activations_batch.to(self.device).float()
        orig_shape = x.shape
        x_flat = x.view(-1, self.config.d_in)

        self.optimizer.zero_grad()

        # Forward pass
        x_centered = x_flat - self.b_dec
        pre_act = torch.matmul(x_centered, self.w_enc) + self.b_enc
        acts = torch.relu(pre_act)

        # Top-K sparsification if enabled
        if self.config.architecture == SAEArchitectureType.TOP_K:
            top_vals, top_idx = torch.topk(acts, k=min(self.config.top_k, self.config.d_sae), dim=-1)
            sparse_acts = torch.zeros_like(acts)
            sparse_acts.scatter_(-1, top_idx, top_vals)
            acts = sparse_acts

        x_hat = torch.matmul(acts, self.w_dec) + self.b_dec

        # Compute loss
        loss_output = self.loss_calculator.compute_loss(x_flat, x_hat, acts)
        loss_output.total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([self.w_enc, self.b_enc, self.w_dec, self.b_dec], max_norm=1.0)
        self.optimizer.step()

        # Normalize decoder columns to unit sphere
        if self.config.normalize_decoder:
            with torch.no_grad():
                norms = torch.norm(self.w_dec, dim=1, keepdim=True) + 1e-8
                self.w_dec.data = self.w_dec.data / norms

        # Update resampler tracking
        with torch.no_grad():
            self.resampler.update_firing_history(acts)
            dead_count = len(self.resampler.get_dead_indices())

        # Check for dead feature resampling
        resampled_count = 0
        if self.step_count % self.config.resample_freq == 0 and dead_count > 0:
            residuals = torch.abs(x_flat - x_hat)
            resampled_count = self.resampler.resample_dead_features(
                sae=self.sae,
                high_loss_residuals=residuals,
                optimizer=self.optimizer,
            )
            # Sync trainer parameters back
            self.w_enc.data = self.sae.w_enc.clone()
            self.w_dec.data = self.sae.w_dec.clone()
            self.b_enc.data = self.sae.b_enc.clone()

        # Sync back to underlying SAE instance
        with torch.no_grad():
            self.sae.w_enc = self.w_enc.data.clone()
            self.sae.w_dec = self.w_dec.data.clone()
            self.sae.b_enc = self.b_enc.data.clone()
            self.sae.b_dec = self.b_dec.data.clone()

        return SAETrainingStepResult(
            step=self.step_count,
            total_loss=loss_output.total_loss.item(),
            reconstruction_loss=loss_output.reconstruction_loss,
            sparsity_loss=loss_output.sparsity_loss,
            l0_norm=loss_output.l0_norm,
            l1_norm=loss_output.l1_norm,
            explained_variance=loss_output.explained_variance,
            dead_features_count=dead_count,
            resampled_count=resampled_count,
        )

    def save_checkpoint(self, path: str) -> None:
        """Saves SAE weights and config to a checkpoint file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "config": self.config,
            "step": self.step_count,
            "W_enc": self.w_enc.data.cpu(),
            "b_enc": self.b_enc.data.cpu(),
            "W_dec": self.w_dec.data.cpu(),
            "b_dec": self.b_dec.data.cpu(),
        }, path)
