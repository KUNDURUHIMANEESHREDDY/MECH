"""Model Adapter Registry.

Resolves model name strings to the correct adapter class and instantiates it.

Derived from the spec tables, not maintained beside them
--------------------------------------------------------
This used to hand-maintain a ``_REGISTRY`` of ``{id: {"cls": ..., "variant": ...}}``
alongside a ``_FAMILY_GROUPS`` listing, while `model_adapters.py` held its own
``LLAMA`` / ``QWEN`` / ... spec dicts. Two authorities for one fact, which is
the arrangement that lets a rename in one place silently break the other.

It already had. The adapter rewrite replaced ``configs.get(variant,
configs[default])`` -- a silent fallback -- with a hard ``KeyError``, and every
registry entry naming a variant that no longer existed started raising:
``'tinyllama' is not a configured Llama variant``. Four tests caught it; the
registry had broken far more than four ids.

So the family entries are now built from the spec dicts themselves. Adding a
model to ``LLAMA`` makes it resolvable, with no second edit, and it cannot drift.

Aliases
-------
Older callers use shorthand ids. Only ids whose referent is genuinely the *same
model* are aliased. ``tinyllama`` always meant TinyLlama-1.1B -- it is the only
TinyLlama release -- so ``tinyllama -> tinyllama-1.1b`` is an honest alias.

Nothing else is aliased, deliberately. ``qwen-2-1.5b`` is Qwen2-1.5B and
``qwen2.5-1.5b`` is Qwen2.5-1.5B: different models with different weights.
Pointing one at the other would hand back the second model's numbers under the
first model's name, which is exactly the identity mismatch this registry exists
to prevent.

Ids dropped by the adapter rewrite, and so no longer resolvable:
``llama-3-8b``, ``llama-3-70b``, ``qwen-2-1.5b``, ``qwen-2-7b``,
``qwen-2.5-7b``, ``mixtral-8x7b``, ``deepseek-v2-7b``. Each is a distinct model,
not a renamed one, so re-adding them means adding real specs with correct
architecture numbers -- not pointing them at a neighbour. See
``test_registry_no_longer_resolves_ids_it_never_had_specs_for`` so the loss
stays visible.
"""

from __future__ import annotations

from typing import Any, Dict, List, Type

from .adapter_base import ModelAdapter
from .gpt2_adapter import GPT2Adapter
from .model_adapters import (
    DEEPSEEK, GEMMA, LLAMA, MISTRAL, QWEN,
    DeepSeekAdapter, GemmaAdapter, LlamaAdapter, MistralAdapter, QwenAdapter,
)
from .transformer_lens_adapter import TransformerLensAdapter

#: family -> (spec table, adapter class). Single source of truth for the five
#: transformer families.
_FAMILY_SPECS: Dict[str, Any] = {
    "gemma": GEMMA,
    "llama": LLAMA,
    "qwen": QWEN,
    "mistral": MISTRAL,
    "deepseek": DEEPSEEK,
}

_FAMILY_CLASSES: Dict[str, Type[ModelAdapter]] = {
    "gemma": GemmaAdapter,
    "llama": LlamaAdapter,
    "qwen": QwenAdapter,
    "mistral": MistralAdapter,
    "deepseek": DeepSeekAdapter,
}

# GPT-2 and TransformerLens are not spec-table driven, so they stay declared.
_STATIC_REGISTRY: Dict[str, Dict[str, Any]] = {
    "gpt2":           {"cls": GPT2Adapter,    "variant": "small"},
    "gpt2-small":     {"cls": GPT2Adapter,    "variant": "small"},
    "gpt2-medium":    {"cls": GPT2Adapter,    "variant": "medium"},
    "gpt2-large":     {"cls": GPT2Adapter,    "variant": "large"},
    "tl-gpt2":        {"cls": TransformerLensAdapter, "variant": "gpt2-small"},
    "tl-gpt2-small":  {"cls": TransformerLensAdapter, "variant": "gpt2-small"},
    "tl-gemma-2b":    {"cls": TransformerLensAdapter, "variant": "gemma-2b"},
}

_STATIC_GROUPS: Dict[str, List[str]] = {
    "gpt2": ["gpt2", "gpt2-small", "gpt2-medium", "gpt2-large"],
}


def _shorthand_aliases() -> Dict[str, str]:
    """Legacy public id -> current spec key, only where the model is the same."""
    families: Dict[str, str] = {}
    for family, specs in _FAMILY_SPECS.items():
        for key in specs:
            families[key] = family

    aliases: Dict[str, str] = {}
    for alias, target in (("tinyllama", "tinyllama-1.1b"),):
        if target not in families:
            # The spec table changed shape again. Better to drop the alias than
            # to point it somewhere that might not be the same model.
            continue
        aliases[alias] = target
    return aliases


def _build() -> tuple:
    registry: Dict[str, Dict[str, Any]] = dict(_STATIC_REGISTRY)
    groups: Dict[str, List[str]] = {
        family: list(ids) for family, ids in _STATIC_GROUPS.items()
    }

    for family, specs in _FAMILY_SPECS.items():
        adapter_cls = _FAMILY_CLASSES[family]
        for spec_key in specs:
            registry[spec_key] = {"cls": adapter_cls, "variant": spec_key}
        groups.setdefault(family, []).extend(specs)

    for alias, target in _shorthand_aliases().items():
        registry[alias] = dict(registry[target])
        groups[_family_of(target)].append(alias)

    return registry, groups


def _family_of(spec_key: str) -> str:
    for family, specs in _FAMILY_SPECS.items():
        if spec_key in specs:
            return family
    return "unknown"


_REGISTRY, _FAMILY_GROUPS = _build()


class ModelAdapterRegistry:
    """Resolves model name to the correct ModelAdapter class and instantiates it."""

    def list_adapters(self) -> List[Dict[str, Any]]:
        results = []
        for family, variants in _FAMILY_GROUPS.items():
            for v in variants:
                entry = _REGISTRY[v]
                # Constructed once per entry. The previous version built three
                # separate adapters per row to read three fields off the spec,
                # which is three times the work for the same answer.
                adapter = entry["cls"](_variant_kw(entry), mock_mode=True)
                results.append({
                    "model_id": v,
                    "family": family,
                    "hf_repo_id": adapter.spec.hf_repo_id,
                    "num_layers": adapter.spec.num_layers,
                    "d_model": adapter.spec.d_model,
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