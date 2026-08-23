"""Epic 2: Model Manager — multi-model support & layer paging.

Supports loading, unloading, switching, and listing models.
Provides progressive layer streaming and provenance weight digests.

Supported architectures:
    - GPT2 family (gpt2, gpt2-medium, distilgpt2)
    - Gemma (gemma-2b, gemma-7b)          — coming soon
    - Llama (llama-7b)                     — coming soon
    - Pythia (pythia-1b)                   — coming soon
    - Mistral (mistral-7b)                 — coming soon
    - Qwen (qwen-1.5b)                     — coming soon
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .errors import ModelLoadError, ModelNotFoundError
from .event_bus import MODEL_LOADED, MODEL_LOADING, MODEL_SWITCHED, MODEL_UNLOADED, bus
from .memory.layer_pager import LayerExecutionOutput, LayerPager

# ── Registry ─────────────────────────────────────────────────────

ModelEntry = dict[str, str | int | None]

_MODEL_REGISTRY: dict[str, ModelEntry] = {
    "gpt2": {"hf_id": "gpt2", "family": "gpt2", "n_layer": 12, "n_head": 12, "n_embd": 768},
    "gpt2-medium": {"hf_id": "gpt2-medium", "family": "gpt2", "n_layer": 24, "n_head": 16, "n_embd": 1024},
    "distilgpt2": {"hf_id": "distilgpt2", "family": "gpt2", "n_layer": 6, "n_head": 12, "n_embd": 768},
}

# Models that need additional trust or authentication
_FUTURE_MODELS: dict[str, ModelEntry] = {
    "gemma-2b": {"hf_id": "google/gemma-2b", "family": "gemma"},
    "llama-7b": {"hf_id": "meta-llama/Llama-2-7b-hf", "family": "llama"},
    "pythia-1b": {"hf_id": "EleutherAI/pythia-1b", "family": "pythia"},
    "mistral-7b": {"hf_id": "mistralai/Mistral-7B-v0.1", "family": "mistral"},
    "qwen-1.5b": {"hf_id": "Qwen/Qwen1.5-1.8B", "family": "qwen"},
}


# ── Loaded model state ───────────────────────────────────────────

_loaded_model: nn.Module | None = None
_loaded_tokenizer = None
_loaded_name: str | None = None
_loaded_weights_digest: str = "unknown"


# ── API ──────────────────────────────────────────────────────────


def list_models() -> list[str]:
    return list(_MODEL_REGISTRY.keys()) + list(_FUTURE_MODELS.keys())


def model_info(name: str) -> dict:
    entry = _MODEL_REGISTRY.get(name) or _FUTURE_MODELS.get(name)
    if entry is None:
        raise ModelNotFoundError(f"Model '{name}' not in registry")
    return {
        "model_name": name,
        "hf_id": entry["hf_id"],
        "family": entry["family"],
        "loaded": name == _loaded_name,
        "weights_digest": _loaded_weights_digest if name == _loaded_name else "unloaded",
        **{k: v for k, v in entry.items() if k not in ("hf_id", "family")},
    }


def compute_model_weights_digest(model: nn.Module) -> str:
    """Computes a fast structural checksum of model parameter shapes and dtypes."""
    summary_parts = []
    for name, param in model.named_parameters():
        summary_parts.append(f"{name}:{list(param.shape)}:{param.dtype}")
    raw = "|".join(summary_parts[:100])  # Sample first 100 tensors for speed
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def load_model(name: str) -> dict:
    global _loaded_model, _loaded_tokenizer, _loaded_name, _loaded_weights_digest

    entry = _MODEL_REGISTRY.get(name)
    if entry is None:
        future = _FUTURE_MODELS.get(name)
        if future is None:
            raise ModelNotFoundError(
                f"Unknown model '{name}'. Available: {list_models()}"
            )
        raise ModelLoadError(
            f"Model '{name}' is registered but not yet supported in this build. "
            f"Use one of: {list(_MODEL_REGISTRY.keys())}"
        )

    hf_id = str(entry["hf_id"])
    bus.emit(MODEL_LOADING, model_name=name, hf_id=hf_id)

    try:
        if _loaded_model is not None:
            _unload_current()

        _loaded_tokenizer = AutoTokenizer.from_pretrained(hf_id)
        if _loaded_tokenizer.pad_token is None:
            _loaded_tokenizer.pad_token = _loaded_tokenizer.eos_token

        _loaded_model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=torch.float32,
            output_attentions=True,
            output_hidden_states=True,
        )
        _loaded_model.eval()
        _loaded_name = name
        _loaded_weights_digest = compute_model_weights_digest(_loaded_model)

        config = _loaded_model.config
        info = {
            "model_name": name,
            "status": "loaded",
            "weights_digest": _loaded_weights_digest,
            "num_layers": getattr(config, "n_layer", getattr(config, "num_hidden_layers", 0)),
            "num_heads": getattr(config, "n_head", getattr(config, "num_attention_heads", 0)),
            "hidden_dim": getattr(config, "n_embd", getattr(config, "hidden_size", 0)),
        }
        bus.emit(MODEL_LOADED, model_name=name, info=info)
        return info

    except Exception as e:
        raise ModelLoadError(f"Failed to load model '{name}': {e}")


def _unload_current() -> None:
    global _loaded_model, _loaded_tokenizer, _loaded_name, _loaded_weights_digest
    old_name = _loaded_name
    _loaded_model = None
    _loaded_tokenizer = None
    _loaded_name = None
    _loaded_weights_digest = "unknown"
    if old_name:
        bus.emit(MODEL_UNLOADED, model_name=old_name)


def unload_model(name: str | None = None) -> dict:
    if name is not None and name != _loaded_name:
        raise ModelLoadError(f"Model '{name}' is not currently loaded")
    _unload_current()
    return {"status": "unloaded"}


def switch_model(name: str) -> dict:
    """Unload current model and load a new one."""
    info = load_model(name)
    bus.emit(MODEL_SWITCHED, model_name=name)
    return info


def get_model_and_tokenizer():
    if _loaded_model is None or _loaded_tokenizer is None:
        raise ModelLoadError("No model loaded. Call load_model() first.")
    return _loaded_model, _loaded_tokenizer


def get_current_model_name() -> Optional[str]:
    return _loaded_name


def get_current_weights_digest() -> str:
    return _loaded_weights_digest


def is_loaded() -> bool:
    return _loaded_model is not None


def get_layer_pager(device: str = "cpu", offload_to_cpu: bool = False) -> LayerPager:
    """Creates a LayerPager configured for current runtime."""
    return LayerPager(device=device, offload_to_cpu=offload_to_cpu)


def run_layer_paged_forward(
    prompt: str,
    interventions: Optional[Dict[int, Callable[[torch.Tensor], torch.Tensor]]] = None,
    capture_layers: Optional[List[int]] = None,
    session_id: str = "paged_session",
) -> LayerExecutionOutput:
    """Convenience method to execute layer-paged forward pass on current loaded model."""
    if _loaded_model is None or _loaded_tokenizer is None:
        load_model("gpt2")
    pager = get_layer_pager()
    return pager.run_sequential_forward(
        model=_loaded_model,
        tokenizer=_loaded_tokenizer,
        prompt=prompt,
        interventions=interventions,
        capture_layers=capture_layers,
        session_id=session_id,
    )
