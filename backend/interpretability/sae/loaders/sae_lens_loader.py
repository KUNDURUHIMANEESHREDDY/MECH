r"""SAELens Loader for MECH.

Loads pretrained SAEs using the SAELens library if available,
or provides graceful fallback when sae-lens is not installed.
"""

from __future__ import annotations

from typing import Any, Optional
import torch

from ..sae_interface import SAEMetadata, SAEArchitectureType, SAEBackendSource
from ..sae_adapter import SAELensAdapter


class SAELensLoader:
    """Loads checkpoints via sae-lens (SAE.from_pretrained)."""

    @staticmethod
    def is_available() -> bool:
        """Returns True if sae-lens package is installed."""
        try:
            import sae_lens  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def load(
        release: str = "gpt2-small-res-jb",
        sae_id: str = "blocks.8.hook_resid_post",
        device: str = "cpu",
    ) -> SAELensAdapter:
        """Loads a pretrained SAELens checkpoint."""
        try:
            from sae_lens import SAE
            sae_obj, cfg, sparsity = SAE.from_pretrained(
                release=release,
                sae_id=sae_id,
                device=device,
            )
            return SAELensAdapter(sae_lens_obj=sae_obj)
        except ImportError:
            raise ImportError(
                "sae-lens is not installed. To load SAELens checkpoints, "
                "run `pip install sae-lens` in your Python environment."
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load SAELens release '{release}' [{sae_id}]: {exc}")
