"""Model Adapter Registry.

Resolves model name strings to the correct adapter class and instantiates it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Type

from .adapter_base import ModelAdapter
from .gpt2_adapter import GPT2Adapter
from .model_adapters import (
    DeepSeekAdapter, GemmaAdapter, LlamaAdapter, MistralAdapter, QwenAdapter
)
from .transformer_lens_adapter import TransformerLensAdapter

_REGISTRY: Dict[str, Dict[str, Any]] = {
    # GPT-2 family
    "gpt2":           {"cls": GPT2Adapter,    "variant": "small"},
    "gpt2-small":     {"cls": GPT2Adapter,    "variant": "small"},
    "gpt2-medium":    {"cls": GPT2Adapter,    "variant": "medium"},
    "gpt2-large":     {"cls": GPT2Adapter,    "variant": "large"},
    # TransformerLens (v39.6)
    "tl-gpt2":        {"cls": TransformerLensAdapter, "variant": "gpt2-small"},
    "tl-gpt2-small":  {"cls": TransformerLensAdapter, "variant": "gpt2-small"},
    "tl-gemma-2b":    {"cls": TransformerLensAdapter, "variant": "gemma-2b"},
    # Gemma family
    "gemma-2b":       {"cls": GemmaAdapter,   "variant": "gemma-2b"},
    "gemma-7b":       {"cls": GemmaAdapter,   "variant": "gemma-7b"},
    # Llama family
    "tinyllama":      {"cls": LlamaAdapter,   "variant": "tinyllama"},
    "llama-3-8b":     {"cls": LlamaAdapter,   "variant": "llama-3-8b"},
    "llama-3-70b":    {"cls": LlamaAdapter,   "variant": "llama-3-70b"},
    # Qwen family
    "qwen-2-1.5b":    {"cls": QwenAdapter,    "variant": "qwen-2-1.5b"},
    "qwen-2-7b":      {"cls": QwenAdapter,    "variant": "qwen-2-7b"},
    "qwen-2.5-7b":    {"cls": QwenAdapter,    "variant": "qwen-2.5-7b"},
    # Mistral family
    "mistral-7b":     {"cls": MistralAdapter, "variant": "mistral-7b"},
    "mixtral-8x7b":   {"cls": MistralAdapter, "variant": "mixtral-8x7b"},
    # DeepSeek family
    "deepseek-r1-1.5b": {"cls": DeepSeekAdapter, "variant": "deepseek-r1-1.5b"},
    "deepseek-v2-7b":   {"cls": DeepSeekAdapter, "variant": "deepseek-v2-7b"},
}

_FAMILY_GROUPS: Dict[str, List[str]] = {
    "gpt2":     ["gpt2", "gpt2-small", "gpt2-medium", "gpt2-large"],
    "gemma":    ["gemma-2b", "gemma-7b"],
    "llama":    ["tinyllama", "llama-3-8b", "llama-3-70b"],
    "qwen":     ["qwen-2-1.5b", "qwen-2-7b", "qwen-2.5-7b"],
    "mistral":  ["mistral-7b", "mixtral-8x7b"],
    "deepseek": ["deepseek-r1-1.5b", "deepseek-v2-7b"],
}


class ModelAdapterRegistry:
    """Resolves model name to the correct ModelAdapter class and instantiates it."""

    def list_adapters(self) -> List[Dict[str, Any]]:
        results = []
        for family, variants in _FAMILY_GROUPS.items():
            for v in variants:
                entry = _REGISTRY[v]
                results.append({
                    "model_id": v,
                    "family": family,
                    "hf_repo_id": entry["cls"](_variant_kw(entry), mock_mode=True).spec.hf_repo_id,
                    "num_layers": entry["cls"](_variant_kw(entry), mock_mode=True).spec.num_layers,
                    "d_model":    entry["cls"](_variant_kw(entry), mock_mode=True).spec.d_model,
                })
        return results

    def get_adapter(self, model_id: str, mock_mode: bool = False) -> ModelAdapter:
        entry = _REGISTRY.get(model_id.lower())
        if not entry:
            raise ValueError(f"Unknown model '{model_id}'. Available: {list(_REGISTRY)}")
        cls: Type[ModelAdapter] = entry["cls"]
        return cls(_variant_kw(entry), mock_mode=mock_mode)  # type: ignore[arg-type]


def _variant_kw(entry: Dict[str, Any]) -> str:
    return entry.get("variant", "small")
