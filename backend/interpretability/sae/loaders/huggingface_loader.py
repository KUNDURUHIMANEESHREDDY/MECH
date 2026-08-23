r"""Hugging Face SAE Loader for MECH.

Loads community Sparse Autoencoder weights from Hugging Face hub repositories
(e.g., Joseph Bloom's GPT-2 SAEs or Gemma Scope) with strict provenance tracking.
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
from ..sae_adapter import GenericPyTorchSAEAdapter
from ..validation.errors import SAELoadError, SAECorruptedError


class HuggingFaceSAELoader:
    """Loads SAE weights from Hugging Face Hub repos or local HF cache."""

    @staticmethod
    def load(
        repo_id: str,
        filename: str = "sae_weights.pt",
        model_id: str = "gpt2",
        layer: int = 8,
        hook_point: str = "hook_resid_post",
        d_in: int = 768,
        d_sae: int = 3072,
    ) -> GenericPyTorchSAEAdapter:
        """Loads SAE weights from HF repo or local file path."""
        file_path = None
        # Check if repo_id is already a local file path
        if os.path.exists(repo_id):
            file_path = repo_id
        elif os.path.exists(filename):
            file_path = filename
        else:
            try:
                from huggingface_hub import hf_hub_download
                file_path = hf_hub_download(repo_id=repo_id, filename=filename)
            except ImportError:
                raise SAELoadError(
                    "huggingface_hub library is not installed. Run `pip install huggingface_hub` "
                    "or provide a local checkpoint file path."
                )
            except Exception as exc:
                raise SAELoadError(
                    f"Failed to download SAE checkpoint from Hugging Face repo '{repo_id}' ({filename}): {exc}"
                )

        if not file_path or not os.path.exists(file_path):
            raise SAELoadError(f"Could not resolve checkpoint file path for '{repo_id}'.")

        try:
            state = torch.load(file_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise SAELoadError(f"Failed to load checkpoint file '{file_path}': {exc}")

        w_enc = state.get("W_enc", state.get("w_enc", state.get("encoder.weight")))
        w_dec = state.get("W_dec", state.get("w_dec", state.get("decoder.weight")))
        b_enc = state.get("b_enc", state.get("encoder.bias"))
        b_dec = state.get("b_dec", state.get("decoder.bias"))

        if w_enc is None or w_dec is None:
            raise SAECorruptedError(
                f"Hugging Face checkpoint '{file_path}' is missing essential W_enc or W_dec tensor fields."
            )

        actual_d_in = w_enc.shape[0]
        actual_d_sae = w_enc.shape[1]

        if w_dec.shape[0] == actual_d_in and w_dec.shape[1] == actual_d_sae:
            w_dec = w_dec.T

        # Compute SHA256
        hasher = hashlib.sha256()
        hasher.update(w_enc.numpy().tobytes())
        hasher.update(w_dec.numpy().tobytes())
        sha256 = hasher.hexdigest()

        provenance = SAEProvenance(
            source="huggingface",
            checkpoint_identifier=f"{repo_id}/{filename}",
            origin_state=SAEOriginState.REAL_PRETRAINED,
            weights_sha256=sha256,
            verified_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
            description=f"Hugging Face SAE checkpoint from {repo_id}",
        )

        meta = SAEMetadata(
            sae_id=f"hf_{repo_id.replace('/', '_')}_L{layer}",
            model_id=model_id,
            layer=layer,
            hook_point=hook_point,
            d_in=actual_d_in,
            d_sae=actual_d_sae,
            backend_source=SAEBackendSource.HUGGINGFACE,
            checkpoint_path=file_path,
            provenance=provenance,
        )

        return GenericPyTorchSAEAdapter(
            metadata=meta,
            w_enc=w_enc,
            w_dec=w_dec,
            b_enc=b_enc,
            b_dec=b_dec,
        )
