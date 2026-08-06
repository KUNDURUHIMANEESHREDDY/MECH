"""GPT-2 Model Adapter (Small & Medium).

Uses real HuggingFace weights when available; falls back to structured mock data
for unit tests. Wires transformer hooks to expose activations, attention, and
activation patching via the unified ModelAdapter interface.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Dict, List, Optional

from .adapter_base import (
    ActivationResult, AttentionPattern, ModelAdapter, ModelSpec, PatchResult
)


def _gpt2_spec(variant: str = "small", mock_mode: bool = False) -> ModelSpec:
    configs = {
        "small":  ModelSpec("gpt2-small",  "gpt2", 12, 12, 768,  3072, 50257, 1024, "gpt2",        mock_mode=mock_mode),
        "medium": ModelSpec("gpt2-medium", "gpt2", 24, 16, 1024, 4096, 50257, 1024, "gpt2-medium", mock_mode=mock_mode),
        "large":  ModelSpec("gpt2-large",  "gpt2", 36, 20, 1280, 5120, 50257, 1024, "gpt2-large",  mock_mode=mock_mode),
    }
    return configs.get(variant, configs["small"])


def _mock_activation(layer: int, neuron_index: int, prompt: str) -> float:
    """Deterministic mock activation for reproducible tests."""
    seed = int(hashlib.sha256(f"{layer}_{neuron_index}_{prompt[:20]}".encode("utf-8")).hexdigest()[:8], 16)
    return round((seed % 1000) / 200.0 - 2.5, 4)


class GPT2Adapter(ModelAdapter):
    """Adapter for GPT-2 model family with real HuggingFace integration."""

    # Canonical top tokens for known prompts (for mock reproducibility)
    _KNOWN_TOP_TOKENS = {
        "The Eiffel Tower is in":  [("Paris", 0.82), ("France", 0.11), ("the", 0.04)],
        "Mary gave John the":      [("book", 0.51), ("ball", 0.18), ("gift", 0.14)],
        "When Mary and John went":  [("to", 0.61), ("home", 0.22), ("back", 0.09)],
    }

    def __init__(self, variant: str = "small", mock_mode: bool = False) -> None:
        super().__init__(_gpt2_spec(variant, mock_mode))
        self._hooks: List[Any] = []
        self._hook_outputs: Dict[int, Any] = {}

    # ------------------------------------------------------------------ #
    #  Real HuggingFace implementation                                     #
    # ------------------------------------------------------------------ #

    def _forward_with_hooks(self, prompt: str):
        """Run a real forward pass with registered hooks."""
        import torch
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True, output_attentions=True)
        return outputs

    # ------------------------------------------------------------------ #
    #  Unified interface                                                   #
    # ------------------------------------------------------------------ #

    def get_activations(
        self,
        prompt: str,
        layer: int,
        neuron_index: Optional[int] = None,
    ) -> List[ActivationResult]:
        if not self.spec.mock_mode and self._model is not None:
            outputs = self._forward_with_hooks(prompt)
            hidden = outputs.hidden_states[layer]          # [1, seq, d_model]
            results = []
            seq_len = hidden.shape[1]
            for tok_idx in range(seq_len):
                vec = hidden[0, tok_idx, :].tolist()
                indices = [neuron_index] if neuron_index is not None else range(min(8, len(vec)))
                for n_idx in indices:
                    results.append(ActivationResult(
                        layer=layer, token_index=tok_idx, neuron_index=n_idx,
                        activation_value=round(float(vec[n_idx]), 4),
                        context_prompt=prompt,
                    ))
            return results

        # Mock path
        n_indices = [neuron_index] if neuron_index is not None else list(range(8))
        return [
            ActivationResult(
                layer=layer, token_index=0, neuron_index=n,
                activation_value=_mock_activation(layer, n, prompt),
                context_prompt=prompt,
                top_k_tokens=[{"token": " Paris", "prob": 0.82}, {"token": " France", "prob": 0.11}],
            )
            for n in n_indices
        ]

    def get_attention_patterns(self, prompt: str, layer: int) -> List[AttentionPattern]:
        if not self.spec.mock_mode and self._model is not None:
            outputs = self._forward_with_hooks(prompt)
            attn = outputs.attentions[layer]               # [1, heads, seq, seq]
            tokens = self._tokenizer.tokenize(prompt)
            patterns = []
            for h in range(self.spec.num_heads):
                mat = attn[0, h, :, :].tolist()
                entropy = -sum(p * math.log(p + 1e-9) for row in mat for p in row)
                patterns.append(AttentionPattern(layer=layer, head=h, pattern_matrix=mat, tokens=tokens, attn_entropy=round(entropy, 4)))
            return patterns

        # Mock path
        seq_len = max(4, len(prompt.split()))
        patterns = []
        for h in range(self.spec.num_heads):
            mat = [[round(random.uniform(0.05, 0.35), 3) for _ in range(seq_len)] for _ in range(seq_len)]
            patterns.append(AttentionPattern(layer=layer, head=h, pattern_matrix=mat, tokens=prompt.split()[:seq_len], attn_entropy=round(1.2 + h * 0.08, 4)))
        return patterns

    def get_logits(self, prompt: str) -> Dict[str, Any]:
        if not self.spec.mock_mode and self._model is not None:
            import torch
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                logits = self._model(**inputs).logits[0, -1, :]
            top5 = logits.topk(5)
            return {
                "prompt": prompt,
                "top_tokens": [
                    {"token": self._tokenizer.decode([idx]), "logit": round(float(l), 3)}
                    for l, idx in zip(top5.values, top5.indices)
                ],
                "top_token": self._tokenizer.decode([top5.indices[0]]),
            }

        # Mock
        for prefix, tokens in self._KNOWN_TOP_TOKENS.items():
            if prefix in prompt:
                return {
                    "prompt": prompt,
                    "top_tokens": [{"token": t, "logit": p * 10, "prob": p} for t, p in tokens],
                    "top_token": tokens[0][0],
                }
        return {"prompt": prompt, "top_tokens": [{"token": " the", "logit": 4.5, "prob": 0.45}], "top_token": " the"}

    def patch_activation(self, prompt: str, layer: int, neuron_index: int, patch_value: float) -> PatchResult:
        """Patch a specific neuron in the MLP layer."""
        if not self.spec.mock_mode and self._model is not None:
            import torch
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            
            with torch.no_grad():
                orig_logits = self._model(**inputs).logits[0, -1, :]
            orig_top = orig_logits.topk(1)
            original_logit = float(orig_top.values[0])
            top_token_before = self._tokenizer.decode([orig_top.indices[0]])

            def patch_hook(module, input, output):
                if output.shape[-1] > neuron_index:
                    output[0, -1, neuron_index] = patch_value
                return output

            layer_module = self._model.transformer.h[layer].mlp
            handle = layer_module.register_forward_hook(patch_hook)

            try:
                with torch.no_grad():
                    patched_logits = self._model(**inputs).logits[0, -1, :]
                patched_top = patched_logits.topk(1)
                patched_logit_for_orig_token = float(patched_logits[orig_top.indices[0]])
                top_token_after = self._tokenizer.decode([patched_top.indices[0]])

                return PatchResult(
                    original_logit=original_logit,
                    patched_logit=patched_logit_for_orig_token,
                    delta=patched_logit_for_orig_token - original_logit,
                    top_token_before=top_token_before,
                    top_token_after=top_token_after,
                    layer=layer,
                    neuron_index=neuron_index,
                    patch_value=patch_value
                )
            finally:
                handle.remove()

        # Mock Path
        original = _mock_activation(layer, neuron_index, prompt)
        return PatchResult(
            original_logit=original, patched_logit=patch_value, delta=round(patch_value - original, 4),
            top_token_before=" Paris", top_token_after=" France" if abs(patch_value - original) > 1.0 else " Paris",
            layer=layer, neuron_index=neuron_index, patch_value=patch_value,
        )

    def patch_head_output(
        self, prompt: str, layer: int, head_index: int, patch_vector: Optional[List[float]] = None
    ) -> PatchResult:
        """Patch the output of a specific attention head (head-level causal intervention)."""
        if not self.spec.mock_mode and self._model is not None:
            import torch
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

            with torch.no_grad():
                orig_logits = self._model(**inputs).logits[0, -1, :]
            orig_top = orig_logits.topk(1)
            original_logit = float(orig_top.values[0])
            top_token_before = self._tokenizer.decode([orig_top.indices[0]])

            # GPT-2 Attention head output patching (hooking 'attn.c_proj' or internal 'attn')
            # For simplicity, we hook the entire attention block output and mask the specific head
            def head_patch_hook(module, input, output):
                # GPT-2 output shape: [batch, seq, d_model]
                # Head dimension: d_model / num_heads
                d_head = self.spec.d_model // self.spec.num_heads
                start = head_index * d_head
                end = (head_index + 1) * d_head

                if patch_vector is not None:
                    # Inject specific vector (e.g. mean ablation vector)
                    v = torch.tensor(patch_vector, device=output.device, dtype=output.dtype)
                    output[0, -1, start:end] = v
                else:
                    # Zero ablation
                    output[0, -1, start:end] = 0.0
                return output

            layer_module = self._model.transformer.h[layer].attn
            handle = layer_module.register_forward_hook(head_patch_hook)

            try:
                with torch.no_grad():
                    patched_logits = self._model(**inputs).logits[0, -1, :]
                patched_logit_for_orig_token = float(patched_logits[orig_top.indices[0]])
                top_token_after = self._tokenizer.decode([torch.argmax(patched_logits)])

                return PatchResult(
                    original_logit=original_logit,
                    patched_logit=patched_logit_for_orig_token,
                    delta=patched_logit_for_orig_token - original_logit,
                    top_token_before=top_token_before,
                    top_token_after=top_token_after,
                    layer=layer,
                    neuron_index=head_index, # overloaded as head index
                    patch_value=0.0 # overloaded as zero ablation
                )
            finally:
                handle.remove()

        # Mock Path
        return PatchResult(
            original_logit=0.82, patched_logit=0.45, delta=-0.37,
            top_token_before=" Paris", top_token_after=" France",
            layer=layer, neuron_index=head_index, patch_value=0.0
        )

    def run_isolated_circuit(self, prompt: str, head_list: List[str]) -> Dict[str, Any]:
        """Run a forward pass where only specified heads are active (all others are zero-ablated)."""
        if not self.spec.mock_mode and self._model is not None:
            import torch

            # Parse head list: "L9H6" -> (9, 6)
            active_heads = set()
            for h_str in head_list:
                try:
                    parts = h_str.replace("L", "").split("H")
                    active_heads.add((int(parts[0]), int(parts[1])))
                except Exception: continue

            handles = []
            d_head = self.spec.d_model // self.spec.num_heads

            def make_mask_hook(layer_idx):
                def mask_hook(module, input, output):
                    # output shape: [batch, seq, d_model]
                    for h_idx in range(self.spec.num_heads):
                        if (layer_idx, h_idx) not in active_heads:
                            start = h_idx * d_head
                            end = (h_idx + 1) * d_head
                            output[0, :, start:end] = 0.0
                    return output
                return mask_hook

            # Register hooks for all attention layers
            for i in range(self.spec.num_layers):
                layer_module = self._model.transformer.h[i].attn
                handles.append(layer_module.register_forward_hook(make_mask_hook(i)))

            try:
                # Also optionally ablate MLPs if they aren't in the "circuit"
                # For IOI, MLPs are often included or excluded depending on paper
                # We'll just run with MLPs active for now unless requested
                inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
                with torch.no_grad():
                    logits = self._model(**inputs).logits[0, -1, :]

                top5 = logits.topk(5)
                return {
                    "top_token": self._tokenizer.decode([top5.indices[0]]),
                    "top_tokens": [
                        {"token": self._tokenizer.decode([idx]), "logit": round(float(l), 3)}
                        for l, idx in zip(top5.values, top5.indices)
                    ]
                }
            finally:
                for h in handles: h.remove()

        # Mock Path
        return {"top_token": " Mary", "top_tokens": [{"token": " Mary", "logit": 3.2}]}

    def get_residual_stream(self, prompt: str) -> List[Dict[str, Any]]:
        if not self.spec.mock_mode and self._model is not None:
            outputs = self._forward_with_hooks(prompt)
            return [
                {"layer": i, "norm": round(float(h[0, -1, :].norm()), 4)}
                for i, h in enumerate(outputs.hidden_states)
            ]
        return [{"layer": i, "norm": round(1.8 + i * 0.3, 4)} for i in range(self.spec.num_layers + 1)]
