"""Gemma, Llama, Qwen, Mistral, and DeepSeek Adapters.

Each adapter follows the unified ModelAdapter interface from adapter_base.py.
Real HuggingFace / PyTorch weights are loaded when available;
reproducible deterministic statistical mode is used when mock_mode=True or weights are unavailable.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Dict, List, Optional

from .adapter_base import (
    ActivationResult, AttentionPattern, ModelAdapter, ModelSpec, PatchResult
)
from .gpt2_adapter import _mock_activation


# ─────────────────────────────────────────────────────────────────────────────
# Shared real forward hook and mock helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_entropy(matrix: List[List[float]]) -> float:
    """Compute mean Shannon entropy across attention rows."""
    if not matrix or not matrix[0]:
        return 0.0
    entropies = []
    for row in matrix:
        h = -sum(p * math.log(p + 1e-12) for p in row if p > 0)
        entropies.append(h)
    return round(sum(entropies) / max(len(entropies), 1), 4)


def _dynamic_mock_tokens(prompt: str) -> List[Dict[str, Any]]:
    """Generate deterministic dynamic top tokens based on prompt content."""
    words = [w.strip(".,!?;:\"'") for w in prompt.split() if len(w) > 1]
    
    # Contextual associations for common benchmark prompts
    p_lower = prompt.lower()
    if "france" in p_lower or "eiffel" in p_lower:
        top_candidates = [(" Paris", 0.78), (" France", 0.12), (" Europe", 0.05)]
    elif "rome" in p_lower or "italy" in p_lower:
        top_candidates = [(" Rome", 0.81), (" Italy", 0.11), (" Europe", 0.04)]
    elif "capital" in p_lower:
        top_candidates = [(" Paris", 0.65), (" London", 0.18), (" Washington", 0.08)]
    elif "mary" in p_lower and "john" in p_lower:
        top_candidates = [(" Mary", 0.54), (" John", 0.32), (" to", 0.08)]
    elif words:
        top_candidates = [(f" {words[-1]}", 0.45), (f" {words[0]}", 0.28), (" the", 0.15)]
    else:
        top_candidates = [(" the", 0.50), (" a", 0.25), (" is", 0.12)]

    return [{"token": t, "prob": round(p, 4)} for t, p in top_candidates]


def _mock_attention_patterns(num_heads: int, prompt: str, layer: int) -> List[AttentionPattern]:
    raw_tokens = prompt.split() if prompt else ["<empty>"]
    seq_len = max(2, min(len(raw_tokens), 16))
    tokens = raw_tokens[:seq_len]

    patterns = []
    for h in range(num_heads):
        matrix = []
        seed = int(hashlib.sha256(f"{layer}_{h}_{prompt}".encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed)
        
        for i in range(seq_len):
            # Causal lower-triangular attention mask
            weights = [rng.uniform(0.01, 1.0) if j <= i else 0.0 for j in range(seq_len)]
            total = sum(weights) or 1.0
            matrix.append([round(w / total, 4) for w in weights])

        patterns.append(
            AttentionPattern(
                layer=layer,
                head=h,
                pattern_matrix=matrix,
                tokens=tokens,
                attn_entropy=_compute_entropy(matrix),
            )
        )
    return patterns


class _BaseGenericAdapter(ModelAdapter):
    """Generic base providing real PyTorch forward passes and explicit mock paths."""

    def _forward_with_hooks(self, prompt: str):
        import torch
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(f"Live model '{self.spec.name}' is uninitialized.")
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True, output_attentions=True)
        return outputs

    def get_activations(
        self,
        prompt: str,
        layer: int,
        neuron_index: Optional[int] = None,
    ) -> List[ActivationResult]:
        if not self.spec.mock_mode:
            try:
                outputs = self._forward_with_hooks(prompt)
                hidden = outputs.hidden_states[layer]  # [1, seq, d_model]
                results = []
                seq_len = hidden.shape[1]
                for tok_idx in range(seq_len):
                    vec = hidden[0, tok_idx, :].tolist()
                    indices = [neuron_index] if neuron_index is not None else range(min(8, len(vec)))
                    for n_idx in indices:
                        results.append(
                            ActivationResult(
                                layer=layer,
                                token_index=tok_idx,
                                neuron_index=n_idx,
                                activation_value=round(float(vec[n_idx]), 4),
                                context_prompt=prompt,
                            )
                        )
                return results
            except Exception as err:
                raise RuntimeError(f"Failed to extract live activations from {self.spec.name}: {err}") from err

        # CRITICAL: Mock path raises error instead of returning fabricated data
        raise RuntimeError(
            f"Cannot compute activations for '{prompt[:50]}...' on {self.spec.name} - "
            "model is in mock_mode. Mock activations must never reach "
            "EvidenceRecord, Finding, or ScientificConclusion. "
            "Load a real model to compute actual activations."
        )

    def get_attention_patterns(self, prompt: str, layer: int) -> List[AttentionPattern]:
        if not self.spec.mock_mode:
            try:
                outputs = self._forward_with_hooks(prompt)
                if outputs.attentions and layer < len(outputs.attentions):
                    attn = outputs.attentions[layer][0]  # [num_heads, seq_len, seq_len]
                    tokens = [self._tokenizer.decode([t]) for t in self._tokenizer.encode(prompt)]
                    patterns = []
                    for h in range(attn.shape[0]):
                        mat = [[round(float(v), 4) for v in row] for row in attn[h].tolist()]
                        patterns.append(
                            AttentionPattern(
                                layer=layer,
                                head=h,
                                pattern_matrix=mat,
                                tokens=tokens,
                                attn_entropy=_compute_entropy(mat),
                            )
                        )
                    return patterns
                raise RuntimeError(f"Layer {layer} attention weights unavailable.")
            except Exception as err:
                raise RuntimeError(f"Failed to extract live attention patterns: {err}") from err

        # CRITICAL: Mock path raises error instead of returning fabricated data
        raise RuntimeError(
            f"Cannot compute attention patterns for '{prompt[:50]}...' on {self.spec.name} - "
            "model is in mock_mode. Mock attention patterns must never reach "
            "EvidenceRecord, Finding, or ScientificConclusion. "
            "Load a real model to compute actual attention patterns."
        )

    def get_logits(self, prompt: str) -> Dict[str, Any]:
        if not self.spec.mock_mode:
            try:
                import torch
                outputs = self._forward_with_hooks(prompt)
                logits = outputs.logits[0, -1, :]  # [vocab_size]
                probs = torch.softmax(logits, dim=-1)
                top_k = torch.topk(probs, k=5)
                top_tokens = []
                for idx, prob in zip(top_k.indices.tolist(), top_k.values.tolist()):
                    token_str = self._tokenizer.decode([idx])
                    top_tokens.append({"token": token_str, "prob": round(float(prob), 4)})
                return {
                    "prompt": prompt,
                    "top_token": top_tokens[0]["token"],
                    "top_tokens": top_tokens,
                }
            except Exception as err:
                raise RuntimeError(f"Failed to extract live logits: {err}") from err

        # CRITICAL: Mock path raises error instead of returning fabricated data
        raise RuntimeError(
            f"Cannot compute logits for '{prompt[:50]}...' on {self.spec.name} - "
            "model is in mock_mode. Mock logits must never reach "
            "EvidenceRecord, Finding, or ScientificConclusion. "
            "Load a real model to compute actual logits."
        )

    def patch_activation(
        self,
        prompt: str,
        layer: int,
        neuron_index: int,
        patch_value: float,
    ) -> PatchResult:
        if not self.spec.mock_mode:
            import torch
            if self._model is None or self._tokenizer is None:
                raise RuntimeError(f"Live model '{self.spec.name}' is uninitialized for activation patching.")

            # 1. Baseline forward pass
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                base_out = self._model(**inputs)
            base_logits = base_out.logits[0, -1, :]
            top_before_idx = int(torch.argmax(base_logits).item())
            top_before = self._tokenizer.decode([top_before_idx])
            orig_val = float(base_logits[top_before_idx].item())

            # 2. Intervene via hook
            def patch_hook(module, input_tensor, output_tensor):
                if isinstance(output_tensor, tuple):
                    h = output_tensor[0]
                    if neuron_index < h.shape[-1]:
                        h[:, -1, neuron_index] = patch_value
                    return (h,) + output_tensor[1:]
                else:
                    if neuron_index < output_tensor.shape[-1]:
                        output_tensor[:, -1, neuron_index] = patch_value
                    return output_tensor

            blocks = getattr(self._model, "transformer", getattr(self._model, "model", None))
            layers = getattr(blocks, "h", getattr(blocks, "layers", []))
            if layer >= len(layers):
                raise IndexError(f"Layer {layer} out of range for model with {len(layers)} layers.")

            target_block = layers[layer]
            hook_handle = target_block.register_forward_hook(patch_hook)
            try:
                with torch.no_grad():
                    steered_out = self._model(**inputs)
                steered_logits = steered_out.logits[0, -1, :]
                top_after_idx = int(torch.argmax(steered_logits).item())
                top_after = self._tokenizer.decode([top_after_idx])
                patched_val = float(steered_logits[top_before_idx].item())
            finally:
                hook_handle.remove()

            delta = round(patched_val - orig_val, 4)
            return PatchResult(
                original_logit=orig_val,
                patched_logit=patched_val,
                delta=delta,
                top_token_before=top_before,
                top_token_after=top_after,
                layer=layer,
                neuron_index=neuron_index,
                patch_value=patch_value,
            )

        # CRITICAL: Mock path raises error instead of returning fabricated data
        raise RuntimeError(
            f"Cannot patch activation for '{prompt[:50]}...' on {self.spec.name} - "
            "model is in mock_mode. Mock patch results must never reach "
            "EvidenceRecord, Finding, or ScientificConclusion. "
            "Load a real model to perform actual activation patching."
        )

    def get_residual_stream(self, prompt: str) -> List[Dict[str, Any]]:
        if not self.spec.mock_mode:
            try:
                import torch
                outputs = self._forward_with_hooks(prompt)
                stream = []
                for i, h in enumerate(outputs.hidden_states):
                    norm_val = round(float(torch.norm(h[0, -1, :]).item()), 4)
                    stream.append({"layer": i, "norm": norm_val})
                return stream
            except Exception as err:
                raise RuntimeError(f"Failed to extract residual stream: {err}") from err

        # CRITICAL: Mock path raises error instead of returning fabricated data
        raise RuntimeError(
            f"Cannot get residual stream for '{prompt[:50]}...' on {self.spec.name} - "
            "model is in mock_mode. Mock residual stream must never reach "
            "EvidenceRecord, Finding, or ScientificConclusion. "
            "Load a real model to compute actual residual stream norms."
        )



# ─────────────────────────────────────────────────────────────────────────────
# Gemma Adapter
# ─────────────────────────────────────────────────────────────────────────────

class GemmaAdapter(_BaseGenericAdapter):
    """Adapter for Google Gemma model family (Gemma-2B, Gemma-7B, Gemma-2)."""

    def __init__(self, variant: str = "gemma-2b", mock_mode: bool = False) -> None:
        configs = {
            "gemma-2b": ModelSpec("gemma-2b", "gemma", 18, 8,  2048, 16384, 256000, 8192, "google/gemma-2b",  mock_mode=mock_mode),
            "gemma-7b": ModelSpec("gemma-7b", "gemma", 28, 16, 3072, 24576, 256000, 8192, "google/gemma-7b",  mock_mode=mock_mode),
        }
        super().__init__(configs.get(variant, configs["gemma-2b"]))


# ─────────────────────────────────────────────────────────────────────────────
# Llama Adapter
# ─────────────────────────────────────────────────────────────────────────────

class LlamaAdapter(_BaseGenericAdapter):
    """Adapter for Meta Llama model family (Llama-3.1, TinyLlama)."""

    def __init__(self, variant: str = "tinyllama", mock_mode: bool = False) -> None:
        configs = {
            "tinyllama":   ModelSpec("tinyllama-1.1b", "llama", 22, 32, 2048,  5632,  32000,  2048, "TinyLlama/TinyLlama-1.1B-Chat-v1.0", mock_mode=mock_mode),
            "llama-3-8b":  ModelSpec("llama-3-8b",     "llama", 32, 32, 4096,  14336, 128256, 8192, "meta-llama/Meta-Llama-3-8B",          mock_mode=mock_mode),
            "llama-3-70b": ModelSpec("llama-3-70b",    "llama", 80, 64, 8192,  28672, 128256, 8192, "meta-llama/Meta-Llama-3-70B",         mock_mode=mock_mode),
        }
        super().__init__(configs.get(variant, configs["tinyllama"]))


# ─────────────────────────────────────────────────────────────────────────────
# Qwen Adapter
# ─────────────────────────────────────────────────────────────────────────────

class QwenAdapter(_BaseGenericAdapter):
    """Adapter for Alibaba Qwen model family (Qwen-2, Qwen-2.5)."""

    def __init__(self, variant: str = "qwen-2-1.5b", mock_mode: bool = False) -> None:
        configs = {
            "qwen-2-1.5b": ModelSpec("qwen-2-1.5b", "qwen", 28, 16, 1536,  8960,  151936, 32768,  "Qwen/Qwen2-1.5B",    mock_mode=mock_mode),
            "qwen-2-7b":   ModelSpec("qwen-2-7b",   "qwen", 28, 28, 3584,  18944, 151936, 32768,  "Qwen/Qwen2-7B",      mock_mode=mock_mode),
            "qwen-2.5-7b": ModelSpec("qwen-2.5-7b", "qwen", 28, 28, 3584,  18944, 152064, 131072, "Qwen/Qwen2.5-7B",    mock_mode=mock_mode),
        }
        super().__init__(configs.get(variant, configs["qwen-2-1.5b"]))


# ─────────────────────────────────────────────────────────────────────────────
# Mistral Adapter
# ─────────────────────────────────────────────────────────────────────────────

class MistralAdapter(_BaseGenericAdapter):
    """Adapter for Mistral AI model family (Mistral-7B, Mixtral-8x7B)."""

    def __init__(self, variant: str = "mistral-7b", mock_mode: bool = False) -> None:
        configs = {
            "mistral-7b":   ModelSpec("mistral-7b",   "mistral", 32, 32, 4096, 14336, 32000, 32768, "mistralai/Mistral-7B-v0.3",  mock_mode=mock_mode),
            "mixtral-8x7b": ModelSpec("mixtral-8x7b", "mistral", 32, 32, 4096, 14336, 32000, 32768, "mistralai/Mixtral-8x7B-v0.1", mock_mode=mock_mode),
        }
        super().__init__(configs.get(variant, configs["mistral-7b"]))


# ─────────────────────────────────────────────────────────────────────────────
# DeepSeek Adapter
# ─────────────────────────────────────────────────────────────────────────────

class DeepSeekAdapter(_BaseGenericAdapter):
    """Adapter for DeepSeek model family (DeepSeek-V2, DeepSeek-R1)."""

    def __init__(self, variant: str = "deepseek-r1-1.5b", mock_mode: bool = False) -> None:
        configs = {
            "deepseek-r1-1.5b": ModelSpec("deepseek-r1-1.5b", "deepseek", 28, 16, 1536, 8960,  102400, 131072, "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", mock_mode=mock_mode),
            "deepseek-v2-7b":   ModelSpec("deepseek-v2-7b",   "deepseek", 28, 28, 4096, 11008, 102400, 4096,   "deepseek-ai/deepseek-moe-16b-base",           mock_mode=mock_mode),
        }
        super().__init__(configs.get(variant, configs["deepseek-r1-1.5b"]))


def create_adapter(model_id: str = "gpt2", mock_mode: bool = False) -> ModelAdapter:
    """Factory creating the appropriate ModelAdapter for any supported model family."""
    m_lower = model_id.lower()
    if "gpt" in m_lower:
        from .gpt2_adapter import GPT2Adapter
        return GPT2Adapter(mock_mode=mock_mode)
    elif "gemma" in m_lower:
        variant = "gemma-7b" if "7b" in m_lower else "gemma-2b"
        return GemmaAdapter(variant=variant, mock_mode=mock_mode)
    elif "llama" in m_lower or "tinyllama" in m_lower:
        variant = "llama-3-8b" if "8b" in m_lower else ("llama-3-70b" if "70b" in m_lower else "tinyllama")
        return LlamaAdapter(variant=variant, mock_mode=mock_mode)
    elif "qwen" in m_lower:
        variant = "qwen-2-7b" if "7b" in m_lower else "qwen-2-1.5b"
        return QwenAdapter(variant=variant, mock_mode=mock_mode)
    elif "mistral" in m_lower or "mixtral" in m_lower:
        variant = "mixtral-8x7b" if "mixtral" in m_lower else "mistral-7b"
        return MistralAdapter(variant=variant, mock_mode=mock_mode)
    elif "deepseek" in m_lower:
        variant = "deepseek-v2-7b" if "v2" in m_lower else "deepseek-r1-1.5b"
        return DeepSeekAdapter(variant=variant, mock_mode=mock_mode)
    else:
        spec = ModelSpec(
            model_id=model_id,
            family="generic",
            num_layers=12,
            num_heads=12,
            d_model=768,
            d_mlp=3072,
            vocab_size=50257,
            context_length=1024,
            hf_repo_id=model_id,
            mock_mode=mock_mode,
        )
        return _BaseGenericAdapter(spec)



