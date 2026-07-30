"""Epic 2: Model Manager — multi-model support.

Supports loading, unloading, switching, and listing models.
Extensible registry: add new models by name.

Supported architectures:
    - GPT2 family (gpt2, gpt2-medium, distilgpt2)
    - Gemma (gemma-2b, gemma-7b)          — coming soon
    - Llama (llama-7b)                     — coming soon
    - Pythia (pythia-1b)                   — coming soon
    - Mistral (mistral-7b)                 — coming soon
    - Qwen (qwen-1.5b)                     — coming soon
"""

from __future__ import annotations

import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from .errors import ModelLoadError, ModelNotFoundError
from .event_bus import bus, MODEL_LOADED, MODEL_LOADING, MODEL_UNLOADED, MODEL_SWITCHED


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
        **{k: v for k, v in entry.items() if k not in ("hf_id", "family")},
    }


def load_model(name: str) -> dict:
    global _loaded_model, _loaded_tokenizer, _loaded_name

    entry = _MODEL_REGISTRY.get(name)
    if entry is None:
        # Check future models
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

        config = _loaded_model.config
        info = {
            "model_name": name,
            "status": "loaded",
            "num_layers": getattr(config, "n_layer", getattr(config, "num_hidden_layers", 0)),
            "num_heads": getattr(config, "n_head", getattr(config, "num_attention_heads", 0)),
            "hidden_dim": getattr(config, "n_embd", getattr(config, "hidden_size", 0)),
        }
        bus.emit(MODEL_LOADED, model_name=name, info=info)
        return info

    except Exception as e:
        raise ModelLoadError(f"Failed to load model '{name}': {e}")


def _unload_current() -> None:
    global _loaded_model, _loaded_tokenizer, _loaded_name
    old_name = _loaded_name
    _loaded_model = None
    _loaded_tokenizer = None
    _loaded_name = None
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


def is_loaded() -> bool:
    return _loaded_model is not None
