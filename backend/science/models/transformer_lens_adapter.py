"""TransformerLens Model Adapter.

Wraps HookedTransformer to provide high-fidelity mechanistic interpretability.
Enables cross-validation against the mature TransformerLens ecosystem.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from .adapter_base import (
    ActivationResult, AttentionPattern, ModelAdapter, ModelSpec, PatchResult
)


class TransformerLensAdapter(ModelAdapter):
    """Adapter for TransformerLens HookedTransformer models."""

    def __init__(self, model_name: str = "gpt2-small", mock_mode: bool = False) -> None:
        # Map model name to spec
        spec = ModelSpec(
            model_id=f"tl-{model_name}",
            family=model_name.split("-")[0],
            num_layers=12, num_heads=12, d_model=768, d_mlp=3072,
            vocab_size=50257, context_length=1024,
            hf_repo_id=model_name,
            mock_mode=mock_mode
        )
        super().__init__(spec)
        self._tl_model = None
        #: Set when a genuine load failure forced the mock fallback, so the
        #: degradation can be reported rather than hidden.
        self.load_error: Optional[str] = None

    def _load_model(self) -> None:
        """Loads a HookedTransformer."""
        if self.spec.mock_mode: return

        # Resolve through MECH's compat shim: an unsupported transformer-lens
        # release (4.x removed HookedTransformer) must not degrade into a silent
        # mock. Record why we could not load, so callers can report provenance.
        try:
            from backend.interpretability.tl_compat import resolve_hooked_transformer

            hooked_transformer = resolve_hooked_transformer()
        except ImportError as exc:
            raise ImportError(
                f"TransformerLensAdapter cannot load {self.spec.hf_repo_id!r} "
                f"because the installed transformer-lens is unsupported:\n{exc}"
            ) from exc

        try:
            self._tl_model = hooked_transformer.from_pretrained(self.spec.hf_repo_id)
        except Exception as exc:
            # Keep the historical mock fallback for genuine load failures
            # (no network, no cached weights), but make the reason visible
            # instead of degrading silently.
            self.spec.mock_mode = True
            self.load_error = f"{type(exc).__name__}: {exc}"

    def get_activations(self, prompt: str, layer: int, neuron_index: Optional[int] = None) -> List[ActivationResult]:
        if self.spec.mock_mode or self._tl_model is None:
            return [] # Mock data handled by base if needed

        # Real TL implementation
        logits, cache = self._tl_model.run_with_cache(prompt)
        # Assuming post-MLP activations
        act_name = f"blocks.{layer}.mlp.hook_post"
        acts = cache[act_name][0, -1, :] # [d_mlp]

        results = []
        indices = [neuron_index] if neuron_index is not None else range(min(8, len(acts)))
        for idx in indices:
            results.append(ActivationResult(
                layer=layer, token_index=-1, neuron_index=idx,
                activation_value=float(acts[idx]),
                context_prompt=prompt
            ))
        return results

    def get_attention_patterns(self, prompt: str, layer: int) -> List[AttentionPattern]:
        if self.spec.mock_mode or self._tl_model is None:
            return []

        logits, cache = self._tl_model.run_with_cache(prompt)
        attn = cache[f"blocks.{layer}.attn.hook_pattern"][0] # [heads, seq, seq]
        tokens = self._tl_model.to_str_tokens(prompt)

        patterns = []
        for h in range(self.spec.num_heads):
            mat = attn[h].tolist()
            patterns.append(AttentionPattern(
                layer=layer, head=h, pattern_matrix=mat,
                tokens=tokens, attn_entropy=0.0 # calculate if needed
            ))
        return patterns

    def get_logits(self, prompt: str) -> Dict[str, Any]:
        if self.spec.mock_mode or self._tl_model is None:
            # Match GPT2Adapter mock behavior for alignment testing
            from .gpt2_adapter import GPT2Adapter
            mock_adapter = GPT2Adapter(variant="small", mock_mode=True)
            return mock_adapter.get_logits(prompt)

        logits = self._tl_model(prompt)[0, -1, :]
        top5 = logits.topk(5)
        return {
            "prompt": prompt,
            "top_token": self._tl_model.to_string(top5.indices[0]),
            "top_tokens": [
                {"token": self._tl_model.to_string(idx), "logit": float(l)}
                for l, idx in zip(top5.values, top5.indices)
            ]
        }

    def patch_activation(self, prompt: str, layer: int, neuron_index: int, patch_value: float) -> PatchResult:
        """Patch a specific neuron in the MLP layer using HookedTransformer hooks."""
        if self.spec.mock_mode or self._tl_model is None:
            # Match GPT2Adapter mock behavior for alignment testing
            from .gpt2_adapter import GPT2Adapter
            mock_adapter = GPT2Adapter(variant="small", mock_mode=True)
            return mock_adapter.patch_activation(prompt, layer, neuron_index, patch_value)

        # Real TL implementation
        # 1. Original Logits
        orig_logits = self._tl_model(prompt)[0, -1, :]
        orig_top_idx = orig_logits.argmax()
        original_logit = float(orig_logits[orig_top_idx])
        top_token_before = self._tl_model.to_string(orig_top_idx)

        # 2. Define Patching Hook
        def patch_hook(value, hook):
            value[0, -1, neuron_index] = patch_value
            return value

        # 3. Run with Hook
        hook_name = f"blocks.{layer}.mlp.hook_post"
        with self._tl_model.hooks(fwd_hooks=[(hook_name, patch_hook)]):
            patched_logits = self._tl_model(prompt)[0, -1, :]

        patched_logit_for_orig_token = float(patched_logits[orig_top_idx])
        top_token_after = self._tl_model.to_string(patched_logits.argmax())

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

    def patch_head_output(
        self, prompt: str, layer: int, head_index: int, patch_vector: Optional[List[float]] = None
    ) -> PatchResult:
        """Patch the output of a specific attention head (head-level causal intervention)."""
        if self.spec.mock_mode or self._tl_model is None:
            from .gpt2_adapter import GPT2Adapter
            mock_adapter = GPT2Adapter(variant="small", mock_mode=True)
            return mock_adapter.patch_head_output(prompt, layer, head_index, patch_vector)

        import torch
        # 1. Original Logits
        orig_logits = self._tl_model(prompt)[0, -1, :]
        orig_top_idx = orig_logits.argmax()
        original_logit = float(orig_logits[orig_top_idx])
        top_token_before = self._tl_model.to_string(orig_top_idx)

        # 2. Define Head Patching Hook
        def head_patch_hook(value, hook):
            # value shape: [batch, seq, heads, d_head]
            if patch_vector is not None:
                v = torch.tensor(patch_vector, device=value.device, dtype=value.dtype)
                value[0, -1, head_index, :] = v
            else:
                value[0, -1, head_index, :] = 0.0
            return value

        # 3. Run with Hook
        hook_name = f"blocks.{layer}.attn.hook_z"
        with self._tl_model.hooks(fwd_hooks=[(hook_name, head_patch_hook)]):
            patched_logits = self._tl_model(prompt)[0, -1, :]

        patched_logit_for_orig_token = float(patched_logits[orig_top_idx])
        top_token_after = self._tl_model.to_string(patched_logits.argmax())

        return PatchResult(
            original_logit=original_logit,
            patched_logit=patched_logit_for_orig_token,
            delta=patched_logit_for_orig_token - original_logit,
            top_token_before=top_token_before,
            top_token_after=top_token_after,
            layer=layer,
            neuron_index=head_index,
            patch_value=0.0
        )

    def get_residual_stream(self, prompt: str) -> List[Dict[str, Any]]:
        if self.spec.mock_mode or self._tl_model is None:
            # Match GPT2Adapter mock behavior for alignment testing
            return [{"layer": i, "norm": round(1.8 + i * 0.3, 4)} for i in range(self.spec.num_layers + 1)]

        logits, cache = self._tl_model.run_with_cache(prompt)
        # TL resid_post includes embedding as blocks.0.hook_resid_pre or similar
        # We'll map to 0..n_layers
        res = []
        for i in range(self.spec.num_layers):
            res.append({"layer": i, "norm": float(cache[f"blocks.{i}.hook_resid_post"].norm())})

        # Add final LN/output norm if needed to match n_layers + 1
        res.append({"layer": self.spec.num_layers, "norm": float(cache["ln_final.hook_norm"].norm()) if "ln_final.hook_norm" in cache else 0.0})
        return res
