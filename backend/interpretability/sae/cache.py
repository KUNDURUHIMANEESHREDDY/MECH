"""SAE Cache & Memory Manager with Provenance Tracking.

Caches feature activations and model hidden states to disk, locking them
to specific model, dataset, and SAE versions via the unified Execution Artifact Store.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional

import torch

from ...runtime.artifacts import (
    ArtifactMetadata,
    ArtifactStore,
    ExecutionArtifact,
    Provenance,
    compute_artifact_key,
    get_artifact_store,
)

logger = logging.getLogger("MECH.sae_cache")


class PersistentActivationCache:
    """Cache manager for SAE feature activations with full provenance tracking and CAS storage."""

    def __init__(
        self,
        cache_dir: str = "backend/storage/activation_cache",
        store: Optional[ArtifactStore] = None,
    ) -> None:
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.store = store or get_artifact_store()

    def _get_cache_key(self, sae_version: str, dataset_hash: str, prompt_hash: str) -> str:
        raw_key = f"{sae_version}_{dataset_hash}_{prompt_hash}"
        return hashlib.sha256(raw_key.encode()).hexdigest()

    def put(
        self,
        sae_version: str,
        dataset_hash: str,
        prompt: str,
        activations: Dict[str, Any],
        model_sha: str = "unknown",
        layer: Optional[int] = None,
    ) -> str:
        """Stores feature firings in the persistent CAS cache."""
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        key = self._get_cache_key(sae_version, dataset_hash, prompt_hash)

        # 1. Write legacy JSON for backwards compatibility
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        entry = {
            "activations": activations,
            "provenance": {
                "sae_version": sae_version,
                "dataset_hash": dataset_hash,
                "prompt_hash": prompt_hash,
                "model_sha": model_sha,
                "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            }
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2)

        # 2. Store in unified Execution Artifact Store
        prov = Provenance(
            model_id=model_sha,
            operation="sae_encode",
            operation_params={"sae_version": sae_version, "dataset_hash": dataset_hash},
        )
        meta = ArtifactMetadata(
            name=f"sae_latents_{sae_version}",
            component="sae_latents",
            layer=layer,
            prompt_id=prompt_hash,
            custom={"activations_count": len(activations)},
        )
        art = ExecutionArtifact(
            artifact_id=key,
            metadata=meta,
            provenance=prov,
            parent_ids=[f"residual_prompt_{prompt_hash[:12]}"],
            data=entry,
        )
        self.store.put(art, persist_to_disk=True)

        return key

    def get(self, sae_version: str, dataset_hash: str, prompt: str) -> Optional[Dict[str, Any]]:
        """Retrieves cached activations if hashes match exactly."""
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        key = self._get_cache_key(sae_version, dataset_hash, prompt_hash)

        # Check ArtifactStore first
        art = self.store.get(key, load_tensor=False)
        if art is not None and art.data is not None:
            return art.data

        # Fallback to legacy file
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Swallowed exception: %s", exc)
                return None

        return None

    def clear(self) -> None:
        """Clears all cached activations."""
        if not os.path.exists(self.cache_dir):
            return
        for f in os.listdir(self.cache_dir):
            if f.endswith(".json"):
                try:
                    os.remove(os.path.join(self.cache_dir, f))
                except OSError:
                    pass


class SAECache:
    """In-memory cache for loaded SAE models."""

    def __init__(self) -> None:
        self._cache: Dict[str, Any] = {}

    def put(self, key: str, value: Any) -> None:
        self._cache[key] = value

    def get(self, key: str) -> Any | None:
        return self._cache.get(key)

    def clear(self) -> None:
        self._cache.clear()
