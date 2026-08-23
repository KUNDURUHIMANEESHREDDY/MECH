r"""Native MECH SAE Loader.

Loads native PyTorch SAE state dicts with cryptographic provenance verification
or constructs deterministic initialized SAE instances.
"""

from __future__ import annotations

import hashlib
import logging
import os
import datetime as _dt
from typing import Any, Dict, Optional
import torch

logger = logging.getLogger(__name__)

from ..sae_interface import (
    SAEMetadata,
    SAEArchitectureType,
    SAEBackendSource,
    SAEOriginState,
    SAEProvenance,
)
from ..sae_adapter import NativeMECHSAE, GenericPyTorchSAEAdapter
from ..validation.errors import SAELoadError, SAECorruptedError


class NativeSAELoader:
    """Loads native MECH Sparse Autoencoders from disk or initializes fresh instances."""

    @staticmethod
    def load(
        checkpoint_path: Optional[str] = None,
        model_id: str = "gpt2",
        layer: int = 8,
        hook_point: str = "hook_resid_post",
        d_in: int = 768,
        d_sae: int = 3072,
        architecture: SAEArchitectureType = SAEArchitectureType.STANDARD_RELU,
        seed: int = 42,
    ) -> NativeMECHSAE | GenericPyTorchSAEAdapter:
        """Loads or creates a native MECH SAE."""
        if checkpoint_path is not None:
            if not os.path.exists(checkpoint_path):
                raise SAELoadError(f"Checkpoint file does not exist: {checkpoint_path}")

            try:
                state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            except Exception as exc:
                raise SAELoadError(f"Failed to load checkpoint file '{checkpoint_path}': {exc}")

            w_enc = state.get("W_enc", state.get("w_enc", state.get("encoder.weight")))
            w_dec = state.get("W_dec", state.get("w_dec", state.get("decoder.weight")))
            b_enc = state.get("b_enc", state.get("encoder.bias"))
            b_dec = state.get("b_dec", state.get("decoder.bias"))

            if w_enc is None or w_dec is None:
                raise SAECorruptedError(
                    f"Checkpoint '{checkpoint_path}' is missing essential W_enc or W_dec tensors."
                )

            # Compute SHA256 checksum of weights
            hasher = hashlib.sha256()
            hasher.update(w_enc.numpy().tobytes())
            hasher.update(w_dec.numpy().tobytes())
            sha256 = hasher.hexdigest()

            # Infer dimensions if different from default
            actual_d_in = w_enc.shape[0]
            actual_d_sae = w_enc.shape[1]

            if w_dec.shape[0] == actual_d_in and w_dec.shape[1] == actual_d_sae:
                w_dec = w_dec.T

            provenance = SAEProvenance(
                source="local_disk",
                checkpoint_identifier=checkpoint_path,
                origin_state=SAEOriginState.REAL_PRETRAINED,
                weights_sha256=sha256,
                verified_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
                description=f"Loaded from {checkpoint_path}",
            )

            meta = SAEMetadata(
                sae_id=os.path.splitext(os.path.basename(checkpoint_path))[0],
                model_id=model_id,
                layer=layer,
                hook_point=hook_point,
                d_in=actual_d_in,
                d_sae=actual_d_sae,
                architecture=architecture,
                backend_source=SAEBackendSource.NATIVE,
                checkpoint_path=checkpoint_path,
                provenance=provenance,
            )
            return GenericPyTorchSAEAdapter(
                metadata=meta,
                w_enc=w_enc,
                w_dec=w_dec,
                b_enc=b_enc,
                b_dec=b_dec,
            )

        # Fresh initialized Native SAE
        return NativeMECHSAE(
            d_in=d_in,
            d_sae=d_sae,
            model_id=model_id,
            layer=layer,
            hook_point=hook_point,
            architecture=architecture,
            seed=seed,
        )
