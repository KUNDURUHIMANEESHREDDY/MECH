"""Gemma, Llama, Qwen, Mistral, and DeepSeek Adapters.

NOT IMPLEMENTED -- these adapters are simulated for every model family.

Every method below calls a shared mock helper unconditionally. It never reads
``self._model``, so passing ``mock_mode=False`` (which would attempt a real
multi-gigabyte weight load) changed nothing about the returned data: real
architecture metadata was paired with fabricated activations, attention
patterns, logits, and residuals. Cross-model comparisons built on these numbers
-- "causal similarity 0.91 between GPT-2 and Gemma" -- were therefore fiction
with a real model name attached.

Each constructor now forces ``mock_mode`` on, so every downstream provenance
check sees simulated data, and the docstrings say so. GPT-2 has a real
implementation in ``gpt2_adapter.py``; these families do not.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional

from .adapter_base import (
    ActivationResult, AttentionPattern, ModelAdapter, ModelSpec, PatchResult
)
from .gpt2_adapter import _mock_activation


# ─────────────────────────────────────────────────────────────────────────────
# Shared mock helpers
# ─────────────────────────────────────────────────────────────────────────────

def _force_simulated(adapter: ModelAdapter, family: str) -> None:
    """Mark an adapter as simulated regardless of the requested mode.

    These adapters never execute a model, so they must never look like they
    did. Downstream code keys provenance off ``spec.mock_mode``, so forcing it
    here is what stops simulated numbers reaching the evidence boundary.
    """
    adapter.spec = ModelSpec(**{**adapter.spec.__dict__, "mock_mode": True})
    adapter.simulated = True
    adapter.simulation_reason = (
        f"The {family} adapter is not implemented: every method returns fixed "
        "or formula-generated values and no model is ever executed. Treat all "
        "output from this adapter as unavailable, not as a measurement."
    )


def _mock_attention_patterns(num_heads: int, prompt: str, layer: int) -> List[AttentionPattern]:
    seq_len = max(4, len(prompt.split()))
    return [
        AttentionPattern(
            layer=layer, head=h,
            pattern_matrix=[[round(random.uniform(0.05, 0.3), 3) for _ in range(seq_len)] for _ in range(seq_len)],
            tokens=prompt.split()[:seq_len],
            attn_entropy=round(1.1 + h * 0.07, 4),
        )
        for h in range(num_heads)
    ]


def _mock_logits(top_token: str = " Paris") -> Dict[str, Any]:
    return {
        "status": "unavailable",
        "provenance": "seeded",
        "measured": False,
        "validation_eligible": False,
        "publication_eligible": False,
        "reason": ("Simulated logits from an adapter that runs no model; "
                   "these are not the output of this model family."),
        "top_token": top_token,
        "top_tokens": [{"token": top_token, "prob": 0.80}, {"token": " France", "prob": 0.12}],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Gemma Adapter
# ─────────────────────────────────────────────────────────────────────────────

class GemmaAdapter(ModelAdapter):
    """Adapter for Google Gemma model family (Gemma-2B, Gemma-7B, Gemma-2)."""

    def __init__(self, variant: str = "gemma-2b", mock_mode: bool = False) -> None:
        configs = {
            "gemma-2b": ModelSpec("gemma-2b", "gemma", 18, 8,  2048, 16384, 256000, 8192, "google/gemma-2b",  mock_mode=mock_mode),
            "gemma-7b": ModelSpec("gemma-7b", "gemma", 28, 16, 3072, 24576, 256000, 8192, "google/gemma-7b",  mock_mode=mock_mode),
        }
        super().__init__(configs.get(variant, configs["gemma-2b"]))
        _force_simulated(self, "Gemma")

    def get_activations(self, prompt: str, layer: int, neuron_index: Optional[int] = None) -> List[ActivationResult]:
        n_indices = [neuron_index] if neuron_index is not None else list(range(8))
        return [ActivationResult(layer=layer, token_index=0, neuron_index=n,
                                 activation_value=_mock_activation(layer, n, prompt),
                                 context_prompt=prompt) for n in n_indices]

    def get_attention_patterns(self, prompt: str, layer: int) -> List[AttentionPattern]:
        return _mock_attention_patterns(self.spec.num_heads, prompt, layer)

    def get_logits(self, prompt: str) -> Dict[str, Any]:
        return {**_mock_logits(), "prompt": prompt}

    def patch_activation(self, prompt: str, layer: int, neuron_index: int, patch_value: float) -> PatchResult:
        orig = _mock_activation(layer, neuron_index, prompt)
        return PatchResult(orig, patch_value, round(patch_value - orig, 4), " Paris", " London", layer, neuron_index, patch_value)

    def get_residual_stream(self, prompt: str) -> List[Dict[str, Any]]:
        return [{"layer": i, "norm": round(2.1 + i * 0.25, 4)} for i in range(self.spec.num_layers + 1)]


# ─────────────────────────────────────────────────────────────────────────────
# Llama Adapter
# ─────────────────────────────────────────────────────────────────────────────

class LlamaAdapter(ModelAdapter):
    """Adapter for Meta Llama model family (Llama-3.1, TinyLlama)."""

    def __init__(self, variant: str = "tinyllama", mock_mode: bool = False) -> None:
        configs = {
            "tinyllama":  ModelSpec("tinyllama-1.1b", "llama", 22, 32, 2048,  5632,  32000, 2048, "TinyLlama/TinyLlama-1.1B-Chat-v1.0", mock_mode=mock_mode),
            "llama-3-8b": ModelSpec("llama-3-8b",     "llama", 32, 32, 4096,  14336, 128256, 8192, "meta-llama/Meta-Llama-3-8B",          mock_mode=mock_mode),
            "llama-3-70b":ModelSpec("llama-3-70b",    "llama", 80, 64, 8192,  28672, 128256, 8192, "meta-llama/Meta-Llama-3-70B",         mock_mode=mock_mode),
        }
        super().__init__(configs.get(variant, configs["tinyllama"]))
        _force_simulated(self, "Llama")

    def get_activations(self, prompt: str, layer: int, neuron_index: Optional[int] = None) -> List[ActivationResult]:
        n_indices = [neuron_index] if neuron_index is not None else list(range(8))
        return [ActivationResult(layer=layer, token_index=0, neuron_index=n,
                                 activation_value=_mock_activation(layer, n, prompt),
                                 context_prompt=prompt) for n in n_indices]

    def get_attention_patterns(self, prompt: str, layer: int) -> List[AttentionPattern]:
        return _mock_attention_patterns(self.spec.num_heads, prompt, layer)

    def get_logits(self, prompt: str) -> Dict[str, Any]:
        return {**_mock_logits(), "prompt": prompt}

    def patch_activation(self, prompt: str, layer: int, neuron_index: int, patch_value: float) -> PatchResult:
        orig = _mock_activation(layer, neuron_index, prompt)
        return PatchResult(orig, patch_value, round(patch_value - orig, 4), " Paris", " Rome", layer, neuron_index, patch_value)

    def get_residual_stream(self, prompt: str) -> List[Dict[str, Any]]:
        return [{"layer": i, "norm": round(1.9 + i * 0.28, 4)} for i in range(self.spec.num_layers + 1)]


# ─────────────────────────────────────────────────────────────────────────────
# Qwen Adapter
# ─────────────────────────────────────────────────────────────────────────────

class QwenAdapter(ModelAdapter):
    """Adapter for Alibaba Qwen model family (Qwen-2, Qwen-2.5)."""

    def __init__(self, variant: str = "qwen-2-1.5b", mock_mode: bool = False) -> None:
        configs = {
            "qwen-2-1.5b": ModelSpec("qwen-2-1.5b", "qwen", 28, 16, 1536,  8960,  151936, 32768, "Qwen/Qwen2-1.5B",    mock_mode=mock_mode),
            "qwen-2-7b":   ModelSpec("qwen-2-7b",   "qwen", 28, 28, 3584,  18944, 151936, 32768, "Qwen/Qwen2-7B",      mock_mode=mock_mode),
            "qwen-2.5-7b": ModelSpec("qwen-2.5-7b", "qwen", 28, 28, 3584,  18944, 152064, 131072, "Qwen/Qwen2.5-7B",  mock_mode=mock_mode),
        }
        super().__init__(configs.get(variant, configs["qwen-2-1.5b"]))
        _force_simulated(self, "Qwen")

    def get_activations(self, prompt: str, layer: int, neuron_index: Optional[int] = None) -> List[ActivationResult]:
        n_indices = [neuron_index] if neuron_index is not None else list(range(8))
        return [ActivationResult(layer=layer, token_index=0, neuron_index=n,
                                 activation_value=_mock_activation(layer, n, prompt),
                                 context_prompt=prompt) for n in n_indices]

    def get_attention_patterns(self, prompt: str, layer: int) -> List[AttentionPattern]:
        return _mock_attention_patterns(self.spec.num_heads, prompt, layer)

    def get_logits(self, prompt: str) -> Dict[str, Any]:
        return {**_mock_logits(), "prompt": prompt}

    def patch_activation(self, prompt: str, layer: int, neuron_index: int, patch_value: float) -> PatchResult:
        orig = _mock_activation(layer, neuron_index, prompt)
        return PatchResult(orig, patch_value, round(patch_value - orig, 4), " Paris", " Beijing", layer, neuron_index, patch_value)

    def get_residual_stream(self, prompt: str) -> List[Dict[str, Any]]:
        return [{"layer": i, "norm": round(2.0 + i * 0.22, 4)} for i in range(self.spec.num_layers + 1)]


# ─────────────────────────────────────────────────────────────────────────────
# Mistral Adapter
# ─────────────────────────────────────────────────────────────────────────────

class MistralAdapter(ModelAdapter):
    """Adapter for Mistral AI model family (Mistral-7B, Mixtral-8x7B)."""

    def __init__(self, variant: str = "mistral-7b", mock_mode: bool = False) -> None:
        configs = {
            "mistral-7b":   ModelSpec("mistral-7b",   "mistral", 32, 32, 4096, 14336, 32000, 32768, "mistralai/Mistral-7B-v0.3", mock_mode=mock_mode),
            "mixtral-8x7b": ModelSpec("mixtral-8x7b", "mistral", 32, 32, 4096, 14336, 32000, 32768, "mistralai/Mixtral-8x7B-v0.1", mock_mode=mock_mode),
        }
        super().__init__(configs.get(variant, configs["mistral-7b"]))
        _force_simulated(self, "Mistral")

    def get_activations(self, prompt: str, layer: int, neuron_index: Optional[int] = None) -> List[ActivationResult]:
        n_indices = [neuron_index] if neuron_index is not None else list(range(8))
        return [ActivationResult(layer=layer, token_index=0, neuron_index=n,
                                 activation_value=_mock_activation(layer, n, prompt),
                                 context_prompt=prompt) for n in n_indices]

    def get_attention_patterns(self, prompt: str, layer: int) -> List[AttentionPattern]:
        return _mock_attention_patterns(self.spec.num_heads, prompt, layer)

    def get_logits(self, prompt: str) -> Dict[str, Any]:
        return {**_mock_logits(), "prompt": prompt}

    def patch_activation(self, prompt: str, layer: int, neuron_index: int, patch_value: float) -> PatchResult:
        orig = _mock_activation(layer, neuron_index, prompt)
        return PatchResult(orig, patch_value, round(patch_value - orig, 4), " Paris", " Berlin", layer, neuron_index, patch_value)

    def get_residual_stream(self, prompt: str) -> List[Dict[str, Any]]:
        return [{"layer": i, "norm": round(1.7 + i * 0.31, 4)} for i in range(self.spec.num_layers + 1)]


# ─────────────────────────────────────────────────────────────────────────────
# DeepSeek Adapter
# ─────────────────────────────────────────────────────────────────────────────

class DeepSeekAdapter(ModelAdapter):
    """Adapter for DeepSeek model family (DeepSeek-V2, DeepSeek-R1)."""

    def __init__(self, variant: str = "deepseek-r1-1.5b", mock_mode: bool = False) -> None:
        configs = {
            "deepseek-r1-1.5b": ModelSpec("deepseek-r1-1.5b", "deepseek", 28, 16, 1536, 8960, 102400, 131072, "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", mock_mode=mock_mode),
            "deepseek-v2-7b":   ModelSpec("deepseek-v2-7b",   "deepseek", 28, 28, 4096, 11008, 102400, 4096, "deepseek-ai/deepseek-moe-16b-base",           mock_mode=mock_mode),
        }
        super().__init__(configs.get(variant, configs["deepseek-r1-1.5b"]))
        _force_simulated(self, "DeepSeek")

    def get_activations(self, prompt: str, layer: int, neuron_index: Optional[int] = None) -> List[ActivationResult]:
        n_indices = [neuron_index] if neuron_index is not None else list(range(8))
        return [ActivationResult(layer=layer, token_index=0, neuron_index=n,
                                 activation_value=_mock_activation(layer, n, prompt),
                                 context_prompt=prompt) for n in n_indices]

    def get_attention_patterns(self, prompt: str, layer: int) -> List[AttentionPattern]:
        return _mock_attention_patterns(self.spec.num_heads, prompt, layer)

    def get_logits(self, prompt: str) -> Dict[str, Any]:
        return {**_mock_logits(), "prompt": prompt}

    def patch_activation(self, prompt: str, layer: int, neuron_index: int, patch_value: float) -> PatchResult:
        orig = _mock_activation(layer, neuron_index, prompt)
        return PatchResult(orig, patch_value, round(patch_value - orig, 4), " Paris", " Tokyo", layer, neuron_index, patch_value)

    def get_residual_stream(self, prompt: str) -> List[Dict[str, Any]]:
        return [{"layer": i, "norm": round(1.6 + i * 0.27, 4)} for i in range(self.spec.num_layers + 1)]
