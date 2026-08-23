r"""Unified SAE Interface for MECH.

Defines the abstract base contract (SAEInterface) and common data structures
that all Sparse Autoencoder backends (Native, SAELens, SparseAutoencoder, HF)
must implement to ensure seamless interoperability across the MECH platform.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import torch


class SAEArchitectureType(str, Enum):
    STANDARD_RELU = "standard_relu"
    TOP_K = "top_k"
    GATED = "gated"
    JUMP_RELU = "jump_relu"


class SAEBackendSource(str, Enum):
    NATIVE = "native"
    SAE_LENS = "sae_lens"
    SPARSE_AUTOENCODER = "sparse_autoencoder"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


class SAEOriginState(str, Enum):
    REAL_PRETRAINED = "REAL_PRETRAINED"
    NATIVE_TRAINED = "NATIVE_TRAINED"
    SYNTHETIC_INITIALIZED = "SYNTHETIC_INITIALIZED"


@dataclass
class SAEProvenance:
    """Provenance tracking for SAE checkpoints to guarantee origin integrity."""
    source: str
    checkpoint_identifier: str
    origin_state: SAEOriginState = SAEOriginState.SYNTHETIC_INITIALIZED
    weights_sha256: str = ""
    verified_at: Optional[str] = None
    training_steps: Optional[int] = None
    author: Optional[str] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "checkpoint_identifier": self.checkpoint_identifier,
            "origin_state": self.origin_state.value if hasattr(self.origin_state, "value") else str(self.origin_state),
            "weights_sha256": self.weights_sha256,
            "verified_at": self.verified_at,
            "training_steps": self.training_steps,
            "author": self.author,
            "description": self.description,
        }


@dataclass
class SAEMetadata:
    """Metadata describing an SAE checkpoint and its target model hook point."""
    sae_id: str
    model_id: str
    layer: int
    hook_point: str  # e.g. "blocks.8.hook_resid_post", "transformer.h.8.mlp"
    d_in: int
    d_sae: int
    architecture: SAEArchitectureType = SAEArchitectureType.STANDARD_RELU
    backend_source: SAEBackendSource = SAEBackendSource.NATIVE
    expansion_factor: float = 4.0
    checkpoint_path: Optional[str] = None
    provenance: Optional[SAEProvenance] = None
    extra_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sae_id": self.sae_id,
            "model_id": self.model_id,
            "layer": self.layer,
            "hook_point": self.hook_point,
            "d_in": self.d_in,
            "d_sae": self.d_sae,
            "architecture": self.architecture.value if hasattr(self.architecture, "value") else str(self.architecture),
            "backend_source": self.backend_source.value if hasattr(self.backend_source, "value") else str(self.backend_source),
            "expansion_factor": self.expansion_factor,
            "checkpoint_path": self.checkpoint_path,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "extra_config": self.extra_config,
        }


@dataclass
class FeatureActivationSummary:
    """Summary of a single active latent feature."""
    feature_idx: int
    activation: float
    l0_contribution: float = 1.0
    direct_logit_boost: Optional[float] = None
    top_positive_tokens: List[Tuple[str, float]] = field(default_factory=list)
    top_negative_tokens: List[Tuple[str, float]] = field(default_factory=list)
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_idx": self.feature_idx,
            "activation": round(float(self.activation), 4),
            "l0_contribution": round(float(self.l0_contribution), 4),
            "direct_logit_boost": round(float(self.direct_logit_boost), 4) if self.direct_logit_boost is not None else None,
            "top_positive_tokens": [(t, round(float(s), 4)) for t, s in self.top_positive_tokens],
            "top_negative_tokens": [(t, round(float(s), 4)) for t, s in self.top_negative_tokens],
            "label": self.label,
        }


@dataclass
class SAEReconstructionResult:
    """Full results of an SAE forward reconstruction."""
    latents: torch.Tensor  # [batch, seq_len, d_sae] or [seq_len, d_sae]
    reconstructed_hidden: torch.Tensor  # [batch, seq_len, d_in]
    l0_norm: int
    l1_norm: float
    reconstruction_mse: float
    explained_variance: float
    active_features: List[FeatureActivationSummary] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "l0_norm": self.l0_norm,
            "l1_norm": round(float(self.l1_norm), 4),
            "reconstruction_mse": round(float(self.reconstruction_mse), 6),
            "explained_variance": round(float(self.explained_variance), 4),
            "active_features": [f.to_dict() for f in self.active_features],
        }


class SAEInterface(ABC):
    """Abstract Base Class for all Sparse Autoencoder backends in MECH."""

    @property
    @abstractmethod
    def metadata(self) -> SAEMetadata:
        """Returns metadata describing the SAE instance."""
        pass

    @abstractmethod
    def encode(self, x: torch.Tensor, top_k: Optional[int] = None) -> torch.Tensor:
        """Encodes hidden state tensor x -> sparse latents z."""
        pass

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes sparse latents z -> reconstructed hidden state x_hat."""
        pass

    def reconstruct(self, x: torch.Tensor, top_k: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes and decodes hidden state tensor x, returning (z, x_hat)."""
        z = self.encode(x, top_k=top_k)
        x_hat = self.decode(z)
        return z, x_hat

    @abstractmethod
    def get_feature_direction(self, feature_idx: int) -> torch.Tensor:
        """Returns the decoder unit-norm direction vector for feature_idx [d_in]."""
        pass

    def get_sparsity(self, z: torch.Tensor) -> Dict[str, float]:
        """Computes sparsity statistics (L0, L1, active ratio) on latent tensor z."""
        z_flat = z.view(-1, self.metadata.d_sae).float()
        active_mask = z_flat > 1e-4
        l0 = float(active_mask.sum(dim=-1).float().mean().item())
        l1 = float(z_flat.abs().sum(dim=-1).mean().item())
        active_ratio = l0 / max(1, self.metadata.d_sae)
        return {
            "l0": round(l0, 2),
            "l1": round(l1, 4),
            "active_ratio": round(active_ratio, 6),
            "active_count": int(l0),
        }

    def get_reconstruction_error(self, x: torch.Tensor, x_hat: torch.Tensor) -> Dict[str, float]:
        """Computes reconstruction fidelity metrics (MSE, explained variance)."""
        x_flat = x.view(-1, self.metadata.d_in).float()
        x_hat_flat = x_hat.view(-1, self.metadata.d_in).float()
        mse = float(torch.mean((x_flat - x_hat_flat) ** 2).item())
        var_tot = float(torch.var(x_flat).item()) + 1e-8
        explained_var = max(0.0, 1.0 - (mse / var_tot))
        return {
            "mse": round(mse, 6),
            "explained_variance": round(explained_var, 4),
            "rmse": round(float(mse ** 0.5), 6),
        }

    @property
    def device(self) -> torch.device:
        """Returns the device on which the SAE weights/buffers reside."""
        return torch.device("cpu")

    def to(self, device: torch.device | str) -> SAEInterface:
        """Transfers SAE internal weights/buffers to the specified target device."""
        return self
