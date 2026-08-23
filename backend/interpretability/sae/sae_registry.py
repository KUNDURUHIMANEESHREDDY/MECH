r"""Central Registry for Sparse Autoencoders in MECH.

Provides registration, lookup, caching, and discovery of SAE instances
across models, layers, and backend frameworks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from .sae_interface import SAEInterface, SAEMetadata, SAEBackendSource


class SAERegistry:
    """Singleton-style registry storing active SAE instances and configurations."""

    def __init__(self) -> None:
        self._instances: Dict[str, SAEInterface] = {}
        self._metadata_catalog: Dict[str, SAEMetadata] = {}

    def register(self, sae: SAEInterface) -> None:
        """Registers a live SAE instance in the registry."""
        meta = sae.metadata
        key = self._make_key(meta.model_id, meta.layer, meta.hook_point, meta.sae_id)
        self._instances[key] = sae
        self._metadata_catalog[key] = meta

    def get(
        self,
        model_id: str = "gpt2",
        layer: int = 8,
        hook_point: Optional[str] = None,
        sae_id: Optional[str] = None,
    ) -> Optional[SAEInterface]:
        """Retrieves a registered SAE matching the query parameters."""
        # Exact match if sae_id provided
        if sae_id:
            for key, sae in self._instances.items():
                if sae.metadata.sae_id == sae_id:
                    return sae

        # Fallback to model_id + layer match
        for key, sae in self._instances.items():
            meta = sae.metadata
            if meta.model_id == model_id and meta.layer == layer:
                if hook_point is None or meta.hook_point == hook_point:
                    return sae

        return None

    def list_saes(
        self,
        model_id: Optional[str] = None,
        layer: Optional[int] = None,
    ) -> List[SAEMetadata]:
        """Lists all registered SAE metadata records."""
        results = []
        for meta in self._metadata_catalog.values():
            if model_id and meta.model_id != model_id:
                continue
            if layer is not None and meta.layer != layer:
                continue
            results.append(meta)
        return results

    def clear(self) -> None:
        """Clears all registered instances."""
        self._instances.clear()
        self._metadata_catalog.clear()

    @staticmethod
    def _make_key(model_id: str, layer: int, hook_point: str, sae_id: str) -> str:
        return f"{model_id}::L{layer}::{hook_point}::{sae_id}"


# Global default registry instance
default_sae_registry = SAERegistry()
