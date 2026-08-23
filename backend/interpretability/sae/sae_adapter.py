r"""SAE Adapters for MECH.

Implements adapters mapping Native MECH SAEs, SAELens checkpoints,
SparseAutoencoder models, and generic PyTorch state dictionaries to the SAEInterface.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import torch
import numpy as np

from .sae_interface import (
    SAEInterface,
    SAEMetadata,
    SAEArchitectureType,
    SAEBackendSource,
    FeatureActivationSummary,
    SAEReconstructionResult,
)


class NativeMECHSAE(SAEInterface):
    """Primary MECH research backend: pure PyTorch Sparse Autoencoder."""

    def __init__(
        self,
        d_in: int = 768,
        d_sae: int = 3072,
        model_id: str = "gpt2",
        layer: int = 8,
        hook_point: str = "hook_resid_post",
        architecture: SAEArchitectureType = SAEArchitectureType.STANDARD_RELU,
        seed: int = 42,
    ) -> None:
        self._metadata = SAEMetadata(
            sae_id=f"native_{model_id}_L{layer}_{d_sae}",
            model_id=model_id,
            layer=layer,
            hook_point=hook_point,
            d_in=d_in,
            d_sae=d_sae,
            architecture=architecture,
            backend_source=SAEBackendSource.NATIVE,
            expansion_factor=d_sae / max(1, d_in),
        )
        self._device = torch.device("cpu")

        # Initialize normalized dictionary basis
        rng = np.random.default_rng(seed + layer * 1000)
        raw_w_enc = rng.standard_normal((d_in, d_sae), dtype=np.float32) / np.sqrt(d_in)
        raw_w_dec = rng.standard_normal((d_sae, d_in), dtype=np.float32)
        norms = np.linalg.norm(raw_w_dec, axis=1, keepdims=True) + 1e-8
        raw_w_dec = raw_w_dec / norms

        self.w_enc = torch.tensor(raw_w_enc, dtype=torch.float32)
        self.b_enc = torch.zeros(d_sae, dtype=torch.float32)
        self.w_dec = torch.tensor(raw_w_dec, dtype=torch.float32)
        self.b_dec = torch.zeros(d_in, dtype=torch.float32)

    @property
    def metadata(self) -> SAEMetadata:
        return self._metadata

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> NativeMECHSAE:
        dev = torch.device(device)
        self._device = dev
        self.w_enc = self.w_enc.to(dev)
        self.b_enc = self.b_enc.to(dev)
        self.w_dec = self.w_dec.to(dev)
        self.b_dec = self.b_dec.to(dev)
        return self

    def encode(self, x: torch.Tensor, top_k: Optional[int] = None) -> torch.Tensor:
        orig_shape = x.shape
        x_dev = x.to(self.w_enc.device).float()
        x_flat = x_dev.view(-1, self._metadata.d_in)
        x_centered = x_flat - self.b_dec
        pre_act = torch.matmul(x_centered, self.w_enc) + self.b_enc
        acts = torch.relu(pre_act)

        if top_k is not None and top_k < self._metadata.d_sae:
            top_vals, top_idx = torch.topk(acts, k=top_k, dim=-1)
            sparse_acts = torch.zeros_like(acts)
            sparse_acts.scatter_(-1, top_idx, top_vals)
            acts = sparse_acts

        out_shape = list(orig_shape[:-1]) + [self._metadata.d_sae]
        return acts.view(*out_shape)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        orig_shape = z.shape
        z_dev = z.to(self.w_dec.device).float()
        z_flat = z_dev.view(-1, self._metadata.d_sae)
        x_rec = torch.matmul(z_flat, self.w_dec) + self.b_dec
        out_shape = list(orig_shape[:-1]) + [self._metadata.d_in]
        return x_rec.view(*out_shape)

    def get_feature_direction(self, feature_idx: int) -> torch.Tensor:
        if feature_idx < 0 or feature_idx >= self._metadata.d_sae:
            raise IndexError(f"Feature index {feature_idx} out of range [0, {self._metadata.d_sae})")
        return self.w_dec[feature_idx].clone()


class GenericPyTorchSAEAdapter(SAEInterface):
    """Adapts arbitrary PyTorch SAE weight dictionaries (W_enc, W_dec, b_enc, b_dec)."""

    def __init__(
        self,
        metadata: SAEMetadata,
        w_enc: torch.Tensor,
        w_dec: torch.Tensor,
        b_enc: Optional[torch.Tensor] = None,
        b_dec: Optional[torch.Tensor] = None,
    ) -> None:
        self._metadata = metadata
        self.w_enc = w_enc.float()
        self.w_dec = w_dec.float()
        self.b_enc = b_enc.float() if b_enc is not None else torch.zeros(metadata.d_sae, device=self.w_enc.device)
        self.b_dec = b_dec.float() if b_dec is not None else torch.zeros(metadata.d_in, device=self.w_dec.device)
        self._device = self.w_enc.device

    @property
    def metadata(self) -> SAEMetadata:
        return self._metadata

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> GenericPyTorchSAEAdapter:
        dev = torch.device(device)
        self._device = dev
        self.w_enc = self.w_enc.to(dev)
        self.w_dec = self.w_dec.to(dev)
        self.b_enc = self.b_enc.to(dev)
        self.b_dec = self.b_dec.to(dev)
        return self

    def encode(self, x: torch.Tensor, top_k: Optional[int] = None) -> torch.Tensor:
        orig_shape = x.shape
        x_dev = x.to(self.w_enc.device).float()
        x_flat = x_dev.view(-1, self._metadata.d_in)
        x_centered = x_flat - self.b_dec
        pre_act = torch.matmul(x_centered, self.w_enc) + self.b_enc
        acts = torch.relu(pre_act)

        if top_k is not None and top_k < self._metadata.d_sae:
            top_vals, top_idx = torch.topk(acts, k=top_k, dim=-1)
            sparse_acts = torch.zeros_like(acts)
            sparse_acts.scatter_(-1, top_idx, top_vals)
            acts = sparse_acts

        out_shape = list(orig_shape[:-1]) + [self._metadata.d_sae]
        return acts.view(*out_shape)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        orig_shape = z.shape
        z_dev = z.to(self.w_dec.device).float()
        z_flat = z_dev.view(-1, self._metadata.d_sae)
        x_rec = torch.matmul(z_flat, self.w_dec) + self.b_dec
        out_shape = list(orig_shape[:-1]) + [self._metadata.d_in]
        return x_rec.view(*out_shape)

    def get_feature_direction(self, feature_idx: int) -> torch.Tensor:
        if feature_idx < 0 or feature_idx >= self._metadata.d_sae:
            raise IndexError(f"Feature index {feature_idx} out of range [0, {self._metadata.d_sae})")
        return self.w_dec[feature_idx].clone()


class SAELensAdapter(SAEInterface):
    """Adapts an external SAELens checkpoint (e.g. from sae-lens library) to SAEInterface."""

    def __init__(self, sae_lens_obj: Any, metadata: Optional[SAEMetadata] = None) -> None:
        self.sae_lens_obj = sae_lens_obj

        # Inspect properties from SAELens object
        d_in = getattr(sae_lens_obj, "d_in", getattr(sae_lens_obj.cfg, "d_in", 768))
        d_sae = getattr(sae_lens_obj, "d_sae", getattr(sae_lens_obj.cfg, "d_sae", 3072))
        layer = getattr(sae_lens_obj.cfg, "hook_layer", 8) if hasattr(sae_lens_obj, "cfg") else 8
        hook_point = getattr(sae_lens_obj.cfg, "hook_name", "blocks.8.hook_resid_post") if hasattr(sae_lens_obj, "cfg") else "hook_resid_post"

        if metadata is None:
            self._metadata = SAEMetadata(
                sae_id=getattr(sae_lens_obj.cfg, "sae_id", f"sae_lens_L{layer}_{d_sae}") if hasattr(sae_lens_obj, "cfg") else f"sae_lens_L{layer}",
                model_id=getattr(sae_lens_obj.cfg, "model_name", "gpt2") if hasattr(sae_lens_obj, "cfg") else "gpt2",
                layer=layer,
                hook_point=hook_point,
                d_in=d_in,
                d_sae=d_sae,
                architecture=SAEArchitectureType.STANDARD_RELU,
                backend_source=SAEBackendSource.SAE_LENS,
            )
        else:
            self._metadata = metadata
        self._device = torch.device("cpu")

    @property
    def metadata(self) -> SAEMetadata:
        return self._metadata

    @property
    def device(self) -> torch.device:
        if hasattr(self.sae_lens_obj, "device"):
            return torch.device(self.sae_lens_obj.device)
        return self._device

    def to(self, device: torch.device | str) -> SAELensAdapter:
        dev = torch.device(device)
        self._device = dev
        if hasattr(self.sae_lens_obj, "to") and callable(self.sae_lens_obj.to):
            self.sae_lens_obj.to(dev)
        return self

    def encode(self, x: torch.Tensor, top_k: Optional[int] = None) -> torch.Tensor:
        if hasattr(self.sae_lens_obj, "encode"):
            acts = self.sae_lens_obj.encode(x)
        else:
            w_enc = getattr(self.sae_lens_obj, "W_enc", None)
            b_enc = getattr(self.sae_lens_obj, "b_enc", 0)
            b_dec = getattr(self.sae_lens_obj, "b_dec", 0)
            x_dev = x.to(w_enc.device if hasattr(w_enc, "device") else self._device)
            x_centered = x_dev - b_dec
            acts = torch.relu(torch.matmul(x_centered, w_enc) + b_enc)

        if top_k is not None and top_k < self._metadata.d_sae:
            top_vals, top_idx = torch.topk(acts, k=top_k, dim=-1)
            sparse_acts = torch.zeros_like(acts)
            sparse_acts.scatter_(-1, top_idx, top_vals)
            acts = sparse_acts

        return acts

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if hasattr(self.sae_lens_obj, "decode"):
            return self.sae_lens_obj.decode(z)
        w_dec = getattr(self.sae_lens_obj, "W_dec", None)
        b_dec = getattr(self.sae_lens_obj, "b_dec", 0)
        z_dev = z.to(w_dec.device if hasattr(w_dec, "device") else self._device)
        return torch.matmul(z_dev, w_dec) + b_dec

    def get_feature_direction(self, feature_idx: int) -> torch.Tensor:
        if feature_idx < 0 or feature_idx >= self._metadata.d_sae:
            raise IndexError(
                f"Feature index {feature_idx} out of bounds for SAELens with d_sae={self._metadata.d_sae}"
            )
        w_dec = getattr(self.sae_lens_obj, "W_dec", None)
        if w_dec is None:
            w_dec = getattr(self.sae_lens_obj, "w_dec", None)
        if w_dec is None:
            raise RuntimeError("SAELens object has no valid decoder weight matrix (W_dec).")
        
        # Handle orientation [d_sae, d_in] vs [d_in, d_sae]
        if w_dec.shape[0] == self._metadata.d_sae:
            vec = w_dec[feature_idx].clone()
        elif w_dec.shape[1] == self._metadata.d_sae:
            vec = w_dec[:, feature_idx].clone()
        else:
            raise ValueError(f"SAELens W_dec shape {w_dec.shape} does not match d_sae={self._metadata.d_sae}")
        
        if torch.all(vec == 0) or torch.isnan(vec).any():
            raise ValueError(f"Extracted decoder vector for feature {feature_idx} is zero or NaN.")
        return vec.float()


class SparseAutoencoderAdapter(SAEInterface):
    """Adapts external sparse_autoencoder library models and dictionaries to SAEInterface."""

    def __init__(self, sparse_autoencoder_obj: Any, metadata: Optional[SAEMetadata] = None) -> None:
        self.sparse_autoencoder_obj = sparse_autoencoder_obj
        self._device = torch.device("cpu")

        if metadata is None:
            # Infer dimensions if available
            d_in = 768
            d_sae = 3072
            if hasattr(sparse_autoencoder_obj, "cfg"):
                d_in = getattr(sparse_autoencoder_obj.cfg, "d_in", d_in)
                d_sae = getattr(sparse_autoencoder_obj.cfg, "d_sae", d_sae)
            elif hasattr(sparse_autoencoder_obj, "d_in") and hasattr(sparse_autoencoder_obj, "d_sae"):
                d_in = sparse_autoencoder_obj.d_in
                d_sae = sparse_autoencoder_obj.d_sae
            elif isinstance(sparse_autoencoder_obj, dict):
                for k in ["W_dec", "w_dec", "decoder.weight", "_decoder.weight"]:
                    if k in sparse_autoencoder_obj and hasattr(sparse_autoencoder_obj[k], "shape"):
                        shp = sparse_autoencoder_obj[k].shape
                        if len(shp) == 2:
                            d_sae, d_in = (shp[0], shp[1]) if shp[0] > shp[1] else (shp[1], shp[0])
                        break

            self._metadata = SAEMetadata(
                sae_id="sparse_autoencoder_model",
                model_id="gpt2",
                layer=8,
                hook_point="hook_resid_post",
                d_in=d_in,
                d_sae=d_sae,
                architecture=SAEArchitectureType.STANDARD_RELU,
                backend_source=SAEBackendSource.SPARSE_AUTOENCODER,
            )
        else:
            self._metadata = metadata

    @property
    def metadata(self) -> SAEMetadata:
        return self._metadata

    @property
    def device(self) -> torch.device:
        if hasattr(self.sparse_autoencoder_obj, "device"):
            return torch.device(self.sparse_autoencoder_obj.device)
        return self._device

    def to(self, device: torch.device | str) -> SparseAutoencoderAdapter:
        dev = torch.device(device)
        self._device = dev
        if hasattr(self.sparse_autoencoder_obj, "to") and callable(self.sparse_autoencoder_obj.to):
            self.sparse_autoencoder_obj.to(dev)
        elif isinstance(self.sparse_autoencoder_obj, dict):
            for k, v in self.sparse_autoencoder_obj.items():
                if torch.is_tensor(v):
                    self.sparse_autoencoder_obj[k] = v.to(dev)
        return self

    def _extract_weight_or_attr(self, names: List[str]) -> Optional[torch.Tensor]:
        """Helper to locate a tensor across attribute and dict keys."""
        obj = self.sparse_autoencoder_obj
        if isinstance(obj, dict):
            for name in names:
                if name in obj and torch.is_tensor(obj[name]):
                    return obj[name]
        for name in names:
            if hasattr(obj, name):
                val = getattr(obj, name)
                if torch.is_tensor(val):
                    return val
                if hasattr(val, "weight") and torch.is_tensor(val.weight):
                    return val.weight
        return None

    def encode(self, x: torch.Tensor, top_k: Optional[int] = None) -> torch.Tensor:
        x_in = x.float()
        if hasattr(self.sparse_autoencoder_obj, "forward_encode"):
            acts = self.sparse_autoencoder_obj.forward_encode(x_in)
        elif hasattr(self.sparse_autoencoder_obj, "encode"):
            acts = self.sparse_autoencoder_obj.encode(x_in)
        elif hasattr(self.sparse_autoencoder_obj, "encoder") and callable(self.sparse_autoencoder_obj.encoder):
            acts = self.sparse_autoencoder_obj.encoder(x_in)
        else:
            w_enc = self._extract_weight_or_attr(["W_enc", "w_enc", "encoder.weight", "_encoder.weight"])
            if w_enc is not None:
                b_enc = self._extract_weight_or_attr(["b_enc", "encoder.bias", "_encoder.bias"])
                b_dec = self._extract_weight_or_attr(["b_dec", "decoder.bias", "_decoder.bias"])
                if b_enc is None:
                    b_enc = torch.zeros(self._metadata.d_sae, device=w_enc.device)
                if b_dec is None:
                    b_dec = torch.zeros(self._metadata.d_in, device=w_enc.device)
                
                # Check orientation of w_enc: [d_in, d_sae] or [d_sae, d_in]
                x_dev = x_in.to(w_enc.device)
                if w_enc.shape[0] == self._metadata.d_in:
                    pre_act = torch.matmul(x_dev - b_dec, w_enc) + b_enc
                else:
                    pre_act = torch.matmul(x_dev - b_dec, w_enc.t()) + b_enc
                acts = torch.relu(pre_act)
            else:
                raise RuntimeError("SparseAutoencoder object lacks valid encoder methods or weights.")

        if top_k is not None and top_k < self._metadata.d_sae:
            top_vals, top_idx = torch.topk(acts, k=top_k, dim=-1)
            sparse_acts = torch.zeros_like(acts)
            sparse_acts.scatter_(-1, top_idx, top_vals)
            acts = sparse_acts
        return acts

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z_in = z.float()
        if hasattr(self.sparse_autoencoder_obj, "forward_decode"):
            return self.sparse_autoencoder_obj.forward_decode(z_in)
        elif hasattr(self.sparse_autoencoder_obj, "decode"):
            return self.sparse_autoencoder_obj.decode(z_in)
        elif hasattr(self.sparse_autoencoder_obj, "decoder") and callable(self.sparse_autoencoder_obj.decoder):
            return self.sparse_autoencoder_obj.decoder(z_in)
        else:
            w_dec = self._extract_weight_or_attr(["W_dec", "w_dec", "decoder.weight", "_decoder.weight", "decoder_weights"])
            if w_dec is not None:
                b_dec = self._extract_weight_or_attr(["b_dec", "decoder.bias", "_decoder.bias"])
                if b_dec is None:
                    b_dec = torch.zeros(self._metadata.d_in, device=w_dec.device)
                z_dev = z_in.to(w_dec.device)
                if w_dec.shape[0] == self._metadata.d_sae:
                    return torch.matmul(z_dev, w_dec) + b_dec
                else:
                    return torch.matmul(z_dev, w_dec.t()) + b_dec
            raise RuntimeError("SparseAutoencoder object lacks valid decoder methods or weights.")

    def get_feature_direction(self, feature_idx: int) -> torch.Tensor:
        if feature_idx < 0 or feature_idx >= self._metadata.d_sae:
            raise IndexError(
                f"Feature index {feature_idx} out of bounds for SparseAutoencoder with d_sae={self._metadata.d_sae}"
            )
        
        # Discover decoder weights from standard module locations or dict
        w_dec = self._extract_weight_or_attr([
            "W_dec", "w_dec", "decoder.weight", "_decoder.weight", "decoder_weights"
        ])
        if w_dec is None and hasattr(self.sparse_autoencoder_obj, "decoder"):
            dec = self.sparse_autoencoder_obj.decoder
            if hasattr(dec, "weight"):
                w_dec = dec.weight
        if w_dec is None and hasattr(self.sparse_autoencoder_obj, "_decoder"):
            dec = self.sparse_autoencoder_obj._decoder
            if hasattr(dec, "weight"):
                w_dec = dec.weight

        if w_dec is None:
            raise RuntimeError(
                f"Could not locate decoder weights in {type(self.sparse_autoencoder_obj).__name__}. "
                "SparseAutoencoder object must contain 'decoder.weight', 'W_dec', or state dictionary."
            )

        # Handle weight orientations [d_sae, d_in] vs [d_in, d_sae]
        if w_dec.shape[0] == self._metadata.d_sae:
            vec = w_dec[feature_idx].clone()
        elif w_dec.shape[1] == self._metadata.d_sae:
            vec = w_dec[:, feature_idx].clone()
        else:
            raise ValueError(
                f"Decoder weight matrix shape {w_dec.shape} does not match configured d_sae={self._metadata.d_sae}"
            )

        if torch.all(vec == 0) or torch.isnan(vec).any():
            raise ValueError(f"Extracted decoder vector for feature {feature_idx} is all zeros or NaN.")
        return vec.float()


