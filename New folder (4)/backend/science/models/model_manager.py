"""Model Manager.

Handles the lifecycle of real Hugging Face models (download, caching, device placement).
Ensures models are loaded lazily and resources are managed efficiently.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages lazy loading and caching of PyTorch/Transformers models."""

    _instance = None

    def __new__(cls) -> ModelManager:
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._cache = {}
            cls._instance._tokenizer_cache = {}
            cls._instance._setup_done = False
        return cls._instance

    def __init__(self) -> None:
        if not self._setup_done:
            # Optionally check device availability here
            self._device = self._detect_device()
            self._setup_done = True

    def _detect_device(self) -> str:
        """Detect the optimal device (CUDA > MPS > CPU)."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def get_model_and_tokenizer(self, hf_repo_id: str, revision: str = "main") -> Tuple[Any, Any]:
        """Lazy load and cache model and tokenizer."""
        cache_key = f"{hf_repo_id}@{revision}"

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise RuntimeError("transformers and torch must be installed to load real models.")

        if cache_key not in self._cache:
            logger.info(f"Loading model {hf_repo_id} (revision: {revision}) to {self._device}...")
            
            tokenizer = AutoTokenizer.from_pretrained(hf_repo_id, revision=revision)
            
            # Simplified loading logic for demo purposes, avoiding device_map="auto" 
            # if accelerate isn't available.
            model = AutoModelForCausalLM.from_pretrained(
                hf_repo_id,
                revision=revision,
                torch_dtype=torch.float16 if self._device != "cpu" else torch.float32,
            )
            model.to(self._device)
            model.eval()

            self._cache[cache_key] = model
            self._tokenizer_cache[cache_key] = tokenizer
        else:
            logger.debug(f"Cache hit for {cache_key}")

        return self._cache[cache_key], self._tokenizer_cache[cache_key]

    def unload_model(self, hf_repo_id: str, revision: str = "main") -> bool:
        """Free memory for a specific model."""
        cache_key = f"{hf_repo_id}@{revision}"
        if cache_key in self._cache:
            del self._cache[cache_key]
            del self._tokenizer_cache[cache_key]
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            return True
        return False

    def unload_all(self) -> None:
        """Clear all loaded models."""
        self._cache.clear()
        self._tokenizer_cache.clear()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @property
    def loaded_models(self) -> list[str]:
        return list(self._cache.keys())
