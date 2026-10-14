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

import threading

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

# Do not read these three directly.
#
# They are assigned in sequence -- in `load_model` on commit, and in
# `_unload_current` on teardown -- and Python cannot assign three names
# atomically. So a reader that does *not* hold `_LOAD_LOCK` can observe an
# inconsistent set. Measured, repeatedly, on the teardown path:
#
#     tokenizer present, model gone
#     name present, model gone
#
# Inference through the public API is safe: `get_model_and_tokenizer` and
# `is_loaded` both take the lock, and a single accessor call is atomic. That
# is the only supported way to read them.
#
# The structural fix is one immutable record holding all three, swapped in a
# single assignment, with these kept only as derived conveniences for existing
# callers. Until then, the lock is the whole of the guarantee.
_loaded_model: nn.Module | None = None
_loaded_tokenizer = None
_loaded_name: str | None = None

#: Serialises loads, unloads and reads of the three globals above.
#:
#: They are assigned separately, so without a lock a reader can observe one
#: load's `_loaded_model` alongside another's `_loaded_tokenizer` -- a tokenizer
#: that does not match the weights, producing token ids the embedding table
#: does not cover, with no error anywhere. The same window let a failed load
#: leave a stale tokenizer behind.
_LOAD_LOCK = threading.RLock()


# ── API ──────────────────────────────────────────────────────────


def list_models() -> list[dict]:
    """Every known model, each marked with whether it can actually be loaded.

    Returns per-entry dicts rather than bare names. It used to return a flat
    `list[str]` of eight names with nothing distinguishing the three that load
    from the five that raise `ModelLoadError` -- so a UI rendering this list
    offered the user five unusable models and the failure only surfaced when
    they tried. `model_name` is kept so existing callers that only wanted a
    name still have one field to read.
    """
    out: list[dict] = []
    for name, entry in _MODEL_REGISTRY.items():
        out.append({
            "model_name": name,
            "hf_id": entry["hf_id"],
            "family": entry["family"],
            "supported": True,
        })
    for name, entry in _FUTURE_MODELS.items():
        out.append({
            "model_name": name,
            "hf_id": entry["hf_id"],
            "family": entry["family"],
            "supported": False,
        })
    return out


def model_info(name: str) -> dict:
    entry = _MODEL_REGISTRY.get(name)
    supported = entry is not None
    if entry is None:
        entry = _FUTURE_MODELS.get(name)
    if entry is None:
        raise ModelNotFoundError(f"Model '{name}' not in registry")
    with _LOAD_LOCK:
        loaded = name == _loaded_name
    return {
        "model_name": name,
        "hf_id": entry["hf_id"],
        "family": entry["family"],
        "loaded": loaded,
        "supported": supported,
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
                f"Unknown model '{name}'. Available: "
                + ", ".join(e["model_name"] for e in list_models())
            )
        raise ModelLoadError(
            f"Model '{name}' is registered but not yet supported in this build. "
            f"Use one of: {list(_MODEL_REGISTRY.keys())}"
        )

    hf_id = str(entry["hf_id"])

    bus.emit(MODEL_LOADING, model_name=name, hf_id=hf_id)

    # Build the tokenizer and weights into locals, and only then commit them to
    # the globals -- replacing what was loaded as the last step, not the first.
    #
    # The previous order called `_unload_current()` *before* fetching, so any
    # failure left the platform with no model at all: a dropped connection or a
    # full disk destroyed a working 500 MB checkpoint that was serving fine,
    # and the raised error did not mention it. Worse, the tokenizer was
    # assigned to the global before the weights were, so a failure in between
    # left a tokenizer for a model that never loaded -- held, with
    # `_loaded_name` None so nothing reported it, ready for any caller that
    # reached past `get_model_and_tokenizer`.
    with _LOAD_LOCK:
        try:
            new_tokenizer = AutoTokenizer.from_pretrained(hf_id)
            if new_tokenizer.pad_token is None:
                new_tokenizer.pad_token = new_tokenizer.eos_token

            new_model = AutoModelForCausalLM.from_pretrained(
                hf_id,
                torch_dtype=torch.float32,
                output_attentions=True,
                output_hidden_states=True,
            )
            new_model.eval()
        except Exception as e:
            # Nothing has been touched, so the previously loaded model is
            # still the one in use and still reported as loaded.
            raise ModelLoadError(
                f"Failed to load model '{name}': {e}. Any previously loaded "
                f"model is untouched and still in use."
            )

        previous_name = _loaded_name
        _loaded_model = new_model
        _loaded_tokenizer = new_tokenizer
        _loaded_name = name

        if previous_name:
            bus.emit(MODEL_UNLOADED, model_name=previous_name)

        config = new_model.config
        parameter = next(new_model.parameters())
        info = {
            "model_name": name,
            "status": "loaded",
            "num_layers": getattr(config, "n_layer", getattr(config, "num_hidden_layers", 0)),
            "num_heads": getattr(config, "n_head", getattr(config, "num_attention_heads", 0)),
            "hidden_dim": getattr(config, "n_embd", getattr(config, "hidden_size", 0)),
            # Read back from the loaded weights rather than from what the
            # loader was asked for, so a caller can verify the precision and
            # placement it is measuring in.
            "dtype": str(parameter.dtype),
            "device": str(parameter.device),
            "pad_token_set": new_tokenizer.pad_token is not None,
        }
        bus.emit(MODEL_LOADED, model_name=name, info=info)
        return info


def _unload_current() -> None:
    global _loaded_model, _loaded_tokenizer, _loaded_name
    old_name = _loaded_name
    _loaded_model = None
    _loaded_tokenizer = None
    _loaded_name = None
    if old_name:
        bus.emit(MODEL_UNLOADED, model_name=old_name)


def unload_model(name: str | None = None) -> dict:
    with _LOAD_LOCK:
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
    # Read both globals under the lock. Without it a concurrent load can
    # commit one of the two between the two reads, handing back a tokenizer
    # that does not match the weights.
    with _LOAD_LOCK:
        if _loaded_model is None or _loaded_tokenizer is None:
            raise ModelLoadError("No model loaded. Call load_model() first.")
        return _loaded_model, _loaded_tokenizer


def is_loaded() -> bool:
    with _LOAD_LOCK:
        return _loaded_model is not None
