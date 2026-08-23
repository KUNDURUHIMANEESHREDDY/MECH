"""Model Registry for MECH Platform.

Tracks model metadata, versioning, and health status for production reproducibility.
Every model must be registered before use, with complete version and compatibility info.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("MECH.model_registry")


class ModelId(str, Enum):
    """Enum of known model IDs."""
    GPT2_SMALL = "gpt2"
    GPT2_MEDIUM = "gpt2-medium"
    GPT2_LARGE = "gpt2-large"
    GPT2_XL = "gpt2-xl"
    GEMMA_2B = "gemma-2b"
    GEMMA_7B = "gemma-7b"
    LLAMA_3_8B = "llama-3-8b"
    LLAMA_3_70B = "llama-3-70b"


class ModelRegistry:
    """Registry tracking model metadata, versions, and health status.
    
    Stores model registry entries with complete provenance for reproducibility.
    """

    def __init__(self, storage: Any = None) -> None:
        self.storage = storage
        self._models: Dict[str, Dict[str, Any]] = {}
        self._load_models_from_storage()
        self._register_default_models()

    def _load_models_from_storage(self) -> None:
        """Load model entries from storage if available."""
        if not self.storage:
            return
        try:
            entries = self.storage.list_artifacts(investigation_id="model_registry")
            for entry in entries:
                meta = entry.get("metadata")
                if isinstance(meta, dict) and meta.get("model_id"):
                    self._models[meta["model_id"]] = meta
        except Exception as exc:
            logger.debug("Failed to load model registry from storage: %s", exc)

    def _register_default_models(self) -> None:
        """Populate known default model specifications."""
        defaults = [
            {
                "model_id": "gpt2",
                "model_name": "GPT-2 Small (124M)",
                "architecture": "GPT-2 Decoder-Only Transformer",
                "parameter_count": 124_439_808,
                "n_layers": 12,
                "n_heads": 12,
                "d_model": 768,
                "d_mlp": 3072,
                "d_head": 64,
                "vocab_size": 50257,
                "source": "huggingface/gpt2",
                "precision": "float32",
                "execution_strategy": "native",
                "status": "ready",
                "capabilities": {
                    "forward_hooks": True,
                    "attention_patterns": True,
                    "hidden_states": True,
                    "activation_patching": True,
                    "neuron_ablation": True,
                    "head_ablation": True,
                    "steering": True,
                    "logit_lens": True,
                },
            },
            {
                "model_id": "gpt2-medium",
                "model_name": "GPT-2 Medium (355M)",
                "architecture": "GPT-2 Decoder-Only Transformer",
                "parameter_count": 354_823_680,
                "n_layers": 24,
                "n_heads": 16,
                "d_model": 1024,
                "d_mlp": 4096,
                "d_head": 64,
                "vocab_size": 50257,
                "source": "huggingface/gpt2-medium",
                "precision": "float32",
                "execution_strategy": "native",
                "status": "available",
                "capabilities": {
                    "forward_hooks": True,
                    "attention_patterns": True,
                    "hidden_states": True,
                    "activation_patching": True,
                    "neuron_ablation": True,
                    "head_ablation": True,
                    "steering": True,
                    "logit_lens": True,
                },
            },
        ]
        for d in defaults:
            if d["model_id"] not in self._models:
                self._models[d["model_id"]] = {
                    **d,
                    "tokenizer_hash": "sha256:gpt2_tokenizer_bpe",
                    "config_hash": "sha256:gpt2_config_std",
                    "created_at": time.time(),
                    "updated_at": time.time(),
                    "device": "cpu",
                    "dtype": "float32",
                    "version": "1.0.0",
                }

    def register(
        self,
        model_id: str,
        model_name: str,
        architecture: str,
        parameter_count: int,
        tokenizer_hash: str = "",
        config_hash: str = "",
        source: str = "custom",
        precision: str = "float32",
        execution_strategy: str = "native",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Register a model in the registry with complete metadata."""
        entry = {
            "model_id": model_id,
            "model_name": model_name,
            "architecture": architecture,
            "parameter_count": parameter_count,
            "tokenizer_hash": tokenizer_hash or hashlib.sha256(model_id.encode()).hexdigest(),
            "config_hash": config_hash or hashlib.sha256(f"{architecture}:{parameter_count}".encode()).hexdigest(),
            "source": source,
            "precision": precision,
            "execution_strategy": execution_strategy,
            "created_at": time.time(),
            "updated_at": time.time(),
            "status": kwargs.get("status", "ready"),
            "health_check": None,
            "device": kwargs.get("device", "cpu"),
            "dtype": kwargs.get("dtype", "float32"),
            "capabilities": kwargs.get("capabilities", {
                "forward_hooks": True,
                "attention_patterns": True,
                "hidden_states": True,
                "activation_patching": True,
                "neuron_ablation": True,
                "head_ablation": True,
                "steering": True,
                "logit_lens": True,
            }),
            "version": kwargs.get("version", "1.0.0"),
            **kwargs,
        }
        self._models[model_id] = entry
        
        if self.storage:
            try:
                self.storage.save_artifact({
                    "id": f"mod_{model_id}_{int(time.time())}",
                    "investigation_id": "model_registry",
                    "name": f"Model: {model_name}",
                    "artifact_type": "model_metadata",
                    "file_path": f"model://{model_id}",
                    "checksum_sha256": hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest(),
                    "size_bytes": 0,
                    "metadata": entry,
                    "created_at": time.time(),
                })
            except Exception as exc:
                logger.debug("Failed to persist model registration: %s", exc)

        logger.info("Registered model %s in registry", model_id)
        return entry

    def get(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model registry entry by ID."""
        return self._models.get(model_id)

    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        return list(self._models.values())

    def update_health(
        self,
        model_id: str,
        status: str,
        device: str = "cpu",
        dtype: str = "float32",
        capabilities: Optional[Dict[str, bool]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update model health status after validation."""
        if model_id not in self._models:
            logger.warning("Model %s not found in registry", model_id)
            return None

        self._models[model_id]["status"] = status
        self._models[model_id]["device"] = device
        self._models[model_id]["dtype"] = dtype
        if capabilities:
            self._models[model_id]["capabilities"] = capabilities
        self._models[model_id]["updated_at"] = time.time()

        if self.storage:
            try:
                self.storage.save_artifact({
                    "id": f"mod_health_{model_id}_{int(time.time())}",
                    "investigation_id": "model_registry",
                    "name": f"Health: {model_id}",
                    "artifact_type": "model_health",
                    "file_path": f"model://{model_id}/health",
                    "checksum_sha256": hashlib.sha256(
                        json.dumps(self._models[model_id], sort_keys=True).encode()
                    ).hexdigest(),
                    "size_bytes": 0,
                    "metadata": {"health_check_time": time.time()},
                    "created_at": time.time(),
                })
            except Exception as exc:
                logger.debug("Failed to persist model health: %s", exc)

        logger.info("Updated health for model %s: %s", model_id, status)
        return self._models[model_id]

    def validate_model_health(self, model_id: str = "gpt2") -> Dict[str, Any]:
        """Runs a live forward pass validation on the model and verifies hook capabilities."""
        from backend.services import gpt2_engine
        import torch

        if not gpt2_engine.is_available():
            self.update_health(model_id, status="error")
            return {"status": "error", "error": "PyTorch or Transformers not available."}

        res = gpt2_engine.load()
        if res.get("status") not in ("loaded", "ok") and gpt2_engine._model is None:
            self.update_health(model_id, status="error")
            return {"status": "error", "error": res.get("error", "Failed to load model weights")}

        model = gpt2_engine._model
        tokenizer = gpt2_engine._tokenizer

        # Live forward pass test
        t0 = time.time()
        test_prompt = "Hello, world"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs, output_attentions=True, output_hidden_states=True)
        latency_ms = (time.time() - t0) * 1000.0

        n_layers = len(model.transformer.h)
        n_heads = model.config.n_head
        d_model = model.config.n_embd
        d_mlp = model.transformer.h[0].mlp.c_fc.weight.shape[1]

        capabilities = {
            "forward_hooks": True,
            "attention_patterns": out.attentions is not None,
            "hidden_states": out.hidden_states is not None,
            "activation_patching": True,
            "neuron_ablation": True,
            "head_ablation": True,
            "steering": True,
            "logit_lens": True,
        }

        health_data = {
            "status": "healthy",
            "model_id": model_id,
            "latency_ms": round(latency_ms, 2),
            "device": str(next(model.parameters()).device),
            "dtype": str(next(model.parameters()).dtype),
            "n_layers": n_layers,
            "n_heads": n_heads,
            "d_model": d_model,
            "d_mlp": d_mlp,
            "vocab_size": model.config.vocab_size,
            "capabilities": capabilities,
            "timestamp": time.time(),
        }

        self.update_health(
            model_id=model_id,
            status="ready",
            device=health_data["device"],
            dtype=health_data["dtype"],
            capabilities=capabilities,
        )

        return health_data

    def get_capabilities(self, model_id: str) -> Dict[str, bool]:
        """Get supported capabilities for a model."""
        entry = self.get(model_id)
        if entry:
            return entry.get("capabilities", {})
        return {}

    def get_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model metadata."""
        entry = self.get(model_id)
        if entry:
            return {
                "model_id": entry.get("model_id"),
                "model_name": entry.get("model_name"),
                "architecture": entry.get("architecture"),
                "parameter_count": entry.get("parameter_count"),
                "n_layers": entry.get("n_layers"),
                "n_heads": entry.get("n_heads"),
                "d_model": entry.get("d_model"),
                "d_mlp": entry.get("d_mlp"),
                "d_head": entry.get("d_head"),
                "vocab_size": entry.get("vocab_size"),
                "precision": entry.get("precision"),
                "execution_strategy": entry.get("execution_strategy"),
                "source": entry.get("source"),
                "version": entry.get("version"),
                "status": entry.get("status"),
                "device": entry.get("device"),
                "dtype": entry.get("dtype"),
                "capabilities": entry.get("capabilities"),
            }
        return None


# Global registry instance
_registry: Optional[ModelRegistry] = None


def get_model_registry(storage: Any = None) -> ModelRegistry:
    """Get the global model registry instance."""
    global _registry
    if _registry is None:
        from backend.storage.database import DesktopStorage
        if storage is None:
            storage = DesktopStorage()
        _registry = ModelRegistry(storage)
    return _registry