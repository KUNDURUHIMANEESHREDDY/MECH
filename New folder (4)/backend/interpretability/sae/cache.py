"""SAE Cache & Memory Manager with Provenance Tracking.

Caches feature activations and model hidden states to disk, locking them
to specific model, dataset, and SAE versions.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import hashlib
from typing import Any, Dict, List, Optional


class PersistentActivationCache:
    """Cache manager for SAE feature activations with full provenance tracking."""

    def __init__(self, cache_dir: str = "backend/storage/activation_cache") -> None:
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _get_cache_key(self, sae_version: str, dataset_hash: str, prompt_hash: str) -> str:
        # Create a unique SHA256 key for the combination
        raw_key = f"{sae_version}_{dataset_hash}_{prompt_hash}"
        return hashlib.sha256(raw_key.encode()).hexdigest()

    def put(
        self,
        sae_version: str,
        dataset_hash: str,
        prompt: str,
        activations: Dict[str, Any],
        model_sha: str = "unknown"
    ) -> str:
        """Stores feature firings in the persistent cache."""
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        key = self._get_cache_key(sae_version, dataset_hash, prompt_hash)
        cache_file = os.path.join(self.cache_dir, f"{key}.json")

        entry = {
            "activations": activations,
            "provenance": {
                "sae_version": sae_version,
                "dataset_hash": dataset_hash,
                "prompt_hash": prompt_hash,
                "model_sha": model_sha,
                "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            }
        }

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2)
        return key

    def get(self, sae_version: str, dataset_hash: str, prompt: str) -> Optional[Dict[str, Any]]:
        """Retrieves cached activations if hashes match exactly."""
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        key = self._get_cache_key(sae_version, dataset_hash, prompt_hash)
        cache_file = os.path.join(self.cache_dir, f"{key}.json")

        if not os.path.exists(cache_file):
            return None

        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def clear(self) -> None:
        """Clears all cached activations."""
        if not os.path.exists(self.cache_dir): return
        for f in os.listdir(self.cache_dir):
            if f.endswith(".json"):
                os.remove(os.path.join(self.cache_dir, f))


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
