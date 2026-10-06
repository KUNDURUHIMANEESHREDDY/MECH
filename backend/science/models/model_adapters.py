"""Gemma, Llama, Qwen, Mistral, and DeepSeek adapters.

What this used to be
--------------------
A class per family, each with five methods that called a shared mock helper
*unconditionally*. It never read `self._model`, so `mock_mode=False` -- which
would have attempted a real multi-gigabyte weight load -- changed nothing about
the returned data. Real architecture metadata sat beside fabricated activations,
attention patterns, logits and residuals:

    AttentionPattern(..., pattern_matrix=[[round(random.uniform(0.05, 0.3), 3) ...]])
    attn_entropy=round(1.1 + h * 0.07, 4)
    _mock_logits -> {"top_tokens": [{"token": " Paris", "prob": 0.80},
                                    {"token": " France", "prob": 0.12}]}

Cross-model comparisons built on those numbers -- "causal similarity 0.91 between
GPT-2 and Gemma" -- were fiction with a real model name attached. An earlier fix
forced `mock_mode` on in every constructor, which stopped the fiction but left
five families unable to measure anything at all.

What it is now
--------------
The five methods are implemented in `hf_adapter.HFAdapterMixin` against
`transformers`, and these classes are thin specs over it. `mock_mode=False`
loads the family's real weights and every method measures against them.

Two deliberate constraints:

* **No weights bundled, no default family.** These are multi-gigabyte downloads.
  `mock_mode=True` still simulates -- that is what the flag means now -- but it
  is opt-in and the output says so on every method.
* **The spec table is a claim, and it is checked.** `config_agreement()` compares
  each declared field against the config that was actually loaded. When they
  disagree the config wins, because a mis-specified family silently producing
  measurements attributed to the wrong architecture is the exact failure this
  file is being rewritten to end. Note that Llama 2, Mistral and Qwen 2 use
  grouped-query attention, where `num_key_value_heads != num_attention_heads`;
  `num_heads` here means attention heads, which is what the interpretation code
  indexes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .adapter_base import (
    ActivationResult,
    AttentionPattern,
    ModelAdapter,
    ModelSpec,
    PatchResult,
)
from .hf_adapter import HFAdapterMixin

# ─────────────────────────────────────────────────────────────────────────
# Architecture specs
#
# Real published configurations, so `model_id` and `hf_repo_id` are not
# placeholders. `context_length` is the trained maximum, not a guess.
# ─────────────────────────────────────────────────────────────────────────

GEMMA: Dict[str, ModelSpec] = {
    "gemma-2b": ModelSpec(
        "gemma-2b", "gemma", 18, 8, 2048, 16384, 256000, 8192,
        "google/gemma-2b"),
    "gemma-7b": ModelSpec(
        "gemma-7b", "gemma", 28, 16, 3072, 24576, 256000, 8192,
        "google/gemma-7b"),
    "gemma-2-2b": ModelSpec(
        "gemma-2-2b", "gemma", 26, 8, 2304, 9216, 256000, 8192,
        "google/gemma-2-2b"),
}

LLAMA: Dict[str, ModelSpec] = {
    "llama-3.2-1b": ModelSpec(
        "llama-3.2-1b", "llama", 16, 32, 2048, 8192, 128256, 131072,
        "meta-llama/Llama-3.2-1B"),
    "llama-3.2-3b": ModelSpec(
        "llama-3.2-3b", "llama", 28, 24, 3072, 8192, 128256, 131072,
        "meta-llama/Llama-3.2-3B"),
    "llama-2-7b": ModelSpec(
        "llama-2-7b", "llama", 32, 32, 4096, 11008, 32000, 4096,
        "meta-llama/Llama-2-7b-hf"),
    "tinyllama-1.1b": ModelSpec(
        "tinyllama-1.1b", "llama", 22, 32, 2048, 5632, 32000, 2048,
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
}

QWEN: Dict[str, ModelSpec] = {
    "qwen2.5-0.5b": ModelSpec(
        "qwen2.5-0.5b", "qwen", 24, 14, 896, 4864, 151936, 32768,
        "Qwen/Qwen2.5-0.5B"),
    "qwen2.5-1.5b": ModelSpec(
        "qwen2.5-1.5b", "qwen", 28, 28, 1536, 8960, 151936, 32768,
        "Qwen/Qwen2.5-1.5B"),
    "qwen2-0.5b": ModelSpec(
        "qwen2-0.5b", "qwen", 24, 14, 896, 4864, 151936, 32768,
        "Qwen/Qwen2-0.5B"),
}

MISTRAL: Dict[str, ModelSpec] = {
    "mistral-7b": ModelSpec(
        "mistral-7b", "mistral", 32, 32, 4096, 14336, 32000, 32768,
        "mistralai/Mistral-7B-v0.3"),
    "ministral-8b": ModelSpec(
        "ministral-8b", "mistral", 32, 32, 4096, 14336, 32000, 32768,
        "mistralai/Ministral-8B-Instruct-2410"),
    "open-mistral-7b": ModelSpec(
        "open-mistral-7b", "mistral", 32, 32, 4096, 4096, 32000, 32768,
        "mistralai/open-mistral-7b-v0.3"),
}

DEEPSEEK: Dict[str, ModelSpec] = {
    "deepseek-r1-1.5b": ModelSpec(
        "deepseek-r1-1.5b", "deepseek", 28, 12, 1536, 8960, 151936, 131072,
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
    "deepseek-coder-1.3b": ModelSpec(
        "deepseek-coder-1.3b", "deepseek", 24, 16, 2048, 10944, 102272, 16384,
        "deepseek-ai/deepseek-coder-1.3b-base"),
}


def _unavailable(adapter: ModelAdapter, family: str) -> Dict[str, Any]:
    """The refusal every simulated method returns.

    `mock_mode=True` is now opt-in and honest, so a simulated result has to say
    so on the record rather than only in a constructor's docstring.
    """
    adapter.spec = ModelSpec(**{**adapter.spec.__dict__, "mock_mode": True})
    adapter.simulated = True
    adapter.simulation_reason = (
        f"mock_mode=True: the {family} adapter is simulating and no weights are "
        f"loaded. Every value below is generated, not measured. Pass "
        f"mock_mode=False to load {adapter.spec.hf_repo_id}."
    )
    return {
        "status": "unavailable",
        "provenance": "synthetic",
        "measured": False,
        "validation_eligible": False,
        "publication_eligible": False,
        "reason": adapter.simulation_reason,
    }


class _FamilyAdapter(HFAdapterMixin, ModelAdapter):
    """Shared constructor for the five families.

    One class rather than five near-identical ones: the difference between the
    families is a `ModelSpec`, and the methods are now real. The previous file
    had five classes whose only distinction was which dict they looked their
    spec up in.
    """

    #: Overridden by each family subclass.
    SPECS: Dict[str, ModelSpec] = {}
    FAMILY: str = "unknown"

    def __init__(self, variant: str, mock_mode: bool = False) -> None:
        if variant not in self.SPECS:
            raise KeyError(
                f"{variant!r} is not a configured {self.FAMILY} variant. "
                f"Available: {sorted(self.SPECS)}")

        # `mock_mode` has to reach the spec *before* the base constructor runs,
        # not after. `ModelAdapter.__init__` loads weights whenever
        # `spec.mock_mode` is false, and it is called on the line below.
        #
        # Passing the spec through unchanged meant `mock_mode=True` did not
        # prevent loading -- it only marked the result unavailable afterwards.
        # So constructing a "simulated" adapter downloaded and materialised the
        # real weights first, which is the opposite of what the flag says, and
        # was the reason four families in one process took the interpreter down
        # with an access violation. The unit suite has to be able to build every
        # adapter without touching a multi-gigabyte download.
        spec = self.SPECS[variant]
        if mock_mode and not spec.mock_mode:
            spec = ModelSpec(**{**spec.__dict__, "mock_mode": True})

        super().__init__(spec)
        self.variant = variant
        self.simulated = False
        self.simulation_reason: Optional[str] = None
        self._load_failure: Optional[str] = None

        if mock_mode:
            _unavailable(self, self.FAMILY)
            return

        # Load for real, and record the agreement between the spec table and the
        # config that was actually loaded.
        self._load_model()
        if not self.mock_mode:
            self.config_check = self.config_agreement()

    def get_activations(self, prompt: str, layer: int,
                        neuron_index: Optional[int] = None) -> Any:
        if self.spec.mock_mode:
            return _unavailable(self, self.FAMILY)
        return HFAdapterMixin.get_activations(self, prompt, layer, neuron_index)

    def get_attention_patterns(self, prompt: str, layer: int,
                               per_head: bool = False) -> Any:
        if self.spec.mock_mode:
            return _unavailable(self, self.FAMILY)
        return HFAdapterMixin.get_attention_patterns(self, prompt, layer,
                                                     per_head)

    def get_logits(self, prompt: str) -> Dict[str, Any]:
        if self.spec.mock_mode:
            return _unavailable(self, self.FAMILY)
        return HFAdapterMixin.get_logits(self, prompt)

    def patch_activation(self, prompt: str, layer: int, neuron_index: int,
                         patch_value: float) -> Any:
        if self.spec.mock_mode:
            return _unavailable(self, self.FAMILY)
        return HFAdapterMixin.patch_activation(self, prompt, layer,
                                               neuron_index, patch_value)

    def get_residual_stream(self, prompt: str) -> Dict[str, Any]:
        if self.spec.mock_mode:
            return _unavailable(self, self.FAMILY)
        return HFAdapterMixin.get_residual_stream(self, prompt)


class GemmaAdapter(_FamilyAdapter):
    """Google Gemma. `num_heads` counts attention heads, not KV heads."""

    SPECS = GEMMA
    FAMILY = "Gemma"

    def __init__(self, variant: str = "gemma-2b", mock_mode: bool = False) -> None:
        super().__init__(variant, mock_mode)


class LlamaAdapter(_FamilyAdapter):
    """Meta Llama. Grouped-query attention on 3.x and 2."""

    SPECS = LLAMA
    FAMILY = "Llama"

    def __init__(self, variant: str = "llama-3.2-1b",
                 mock_mode: bool = False) -> None:
        super().__init__(variant, mock_mode)


class QwenAdapter(_FamilyAdapter):
    """Alibaba Qwen."""

    SPECS = QWEN
    FAMILY = "Qwen"

    def __init__(self, variant: str = "qwen2.5-0.5b",
                 mock_mode: bool = False) -> None:
        super().__init__(variant, mock_mode)


class MistralAdapter(_FamilyAdapter):
    """Mistral AI. Windowed attention on v0.1/v0.2, sliding on Ministral."""

    SPECS = MISTRAL
    FAMILY = "Mistral"

    def __init__(self, variant: str = "mistral-7b",
                 mock_mode: bool = False) -> None:
        super().__init__(variant, mock_mode)


class DeepSeekAdapter(_FamilyAdapter):
    """DeepSeek. The R1 distill is a Qwen architecture; the coder is a Llama one."""

    SPECS = DEEPSEEK
    FAMILY = "DeepSeek"

    def __init__(self, variant: str = "deepseek-r1-1.5b",
                 mock_mode: bool = False) -> None:
        super().__init__(variant, mock_mode)


__all__ = [
    "GemmaAdapter", "LlamaAdapter", "QwenAdapter", "MistralAdapter",
    "DeepSeekAdapter",
    "GEMMA", "LLAMA", "QWEN", "MISTRAL", "DEEPSEEK",
]