"""GPT-2 Model Adapter (Small & Medium).

Uses real HuggingFace weights when available; falls back to structured mock data
for unit tests. Wires transformer hooks to expose activations, attention, and
activation patching via the unified ModelAdapter interface.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from typing import Any, Dict, List, Optional

from .adapter_base import (
    ActivationResult, AttentionPattern, LiveUnavailable, ModelAdapter, ModelSpec,
    PatchResult,
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


def _parse_head_label(label: str, n_layers: int, n_heads: int):
    """``"L9H6"`` -> ``(9, 6)``, or ``None`` if it is not a real head.

    Returns ``None`` for anything that is not exactly ``L<layer>H<head>`` and
    for indices outside this model's shape. Out-of-range matters as much as
    malformed: ``"L99H99"`` parses cleanly as a tuple, and the caller would
    then request a layer that does not exist, whose mask slice is silently
    never written -- shrinking the circuit without saying so.
    """
    match = re.fullmatch(r"L(\d+)H(\d+)", label)
    if match is None:
        return None
    layer, head = int(match.group(1)), int(match.group(2))
    if not (0 <= layer < n_layers and 0 <= head < n_heads):
        return None
    return (layer, head)


class GPT2Adapter(ModelAdapter):
    """Adapter for GPT-2 model family with real HuggingFace integration."""

    # Canonical top tokens for known prompts (for mock reproducibility)
    _KNOWN_TOP_TOKENS = {
        "The Eiffel Tower is in":  [("Paris", 0.82), ("France", 0.11), ("the", 0.04)],
        "Mary gave John the":      [("book", 0.51), ("ball", 0.18), ("gift", 0.14)],
        "When Mary and John went":  [("to", 0.61), ("home", 0.22), ("back", 0.09)],
    }

    # IOI name set used by reproducibility pipelines.
    #
    # Deliberately unused for answer synthesis. A `_match_ioi_prompt` parser
    # existed here that parsed "When {a} and {b} went to the store, {a} gave a
    # drink to" and returned the indirect object with logit 8.2. Three tests
    # demanded it. It was removed rather than merged: returning a correct IOI
    # answer from a mock makes the fixture indistinguishable from a real
    # forward pass on the exact benchmark the platform reports a score for.
    # The name list stays for callers that construct prompts, which is honest;
    # only answer synthesis is refused.
    _IOI_NAMES = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"]

    def __init__(self, variant: str = "small", mock_mode: bool = False) -> None:
        super().__init__(_gpt2_spec(variant, mock_mode))
        self._hooks: List[Any] = []
        self._hook_outputs: Dict[int, Any] = {}
        self._fingerprint: Optional[str] = None

    # ------------------------------------------------------------------ #
    #  Model identity                                                      #
    # ------------------------------------------------------------------ #

    def model_fingerprint(self) -> Optional[str]:
        """A content hash of the weights currently loaded, or `None`.

        Derived from tensor values, not from `self.spec.model_id`. A label is
        an assertion about provenance; a hash is evidence of it. The distinction
        is the whole point: a swapped checkpoint keeps its model id, keeps its
        shapes, and passes every tensor-level check, while the fingerprint
        changes. That is precisely the case this exists to catch -- activations
        captured under one checkpoint and patched under another are shape-
        compatible, so nothing downstream notices.

        Cached after the first call: the weights are fixed for the life of the
        adapter, and rehashing ~124M parameters on every result would be a
        real cost for no additional guarantee. A caller that mutates weights in
        place must invalidate the cache itself, which no current code path
        does -- recorded here rather than defended against.

        `None` in mock mode, where there are no weights to describe. Returning
        a string derived from the mock's construction would be worse than
        returning nothing: it would fingerprint a fiction.
        """
        if self._model is None:
            return None
        if self._fingerprint is not None:
            return self._fingerprint

        import torch  # local: this module imports torch lazily by design

        digest = hashlib.sha256()
        for name, param in sorted(self._model.named_parameters()):
            digest.update(name.encode("utf-8"))
            tensor = param.detach().to("cpu")
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.to(torch.float32).contiguous().numpy().tobytes())

        self._fingerprint = f"sha256:{digest.hexdigest()[:32]}"
        return self._fingerprint

    def same_model_as(self, other: Any) -> Optional[bool]:
        """Whether two adapters hold the same weights. `None` if unknowable."""
        mine, theirs = self.model_fingerprint(), other.model_fingerprint()
        if mine is None or theirs is None:
            return None
        return mine == theirs

    # ------------------------------------------------------------------ #
    #  Real HuggingFace implementation                                     #
    # ------------------------------------------------------------------ #

    def _forward_with_hooks(self, prompt: str):
        """Run a real forward pass with registered hooks.

        `output_attentions=True` is not sufficient on its own. With the sdpa
        attention path (the default for GPT-2 in current transformers), torch
        returns a tuple of `None` -- one per layer -- instead of the attention
        weights. Nothing raises; `outputs.attentions[layer]` is simply None,
        and the caller fails later with an unrelated error deep inside its own
        arithmetic. This silently disabled every attention-based measurement:
        the induction-heads benchmark could not run at all.

        Forcing the eager implementation materialises the weights, which is
        what "give me the attention matrix" is asking for. Done per call on the
        config rather than by rebuilding the model, and restored afterwards so
        the choice does not leak into unrelated forward passes.
        """
        import torch
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        config = self._model.config
        previous = getattr(config, "_attn_implementation", None)
        if previous != "eager":
            config._attn_implementation = "eager"
        try:
            with torch.no_grad():
                outputs = self._model(**inputs, output_hidden_states=True,
                                      output_attentions=True)
        finally:
            if previous is not None:
                config._attn_implementation = previous
        return outputs

    def _forward_with_hooks_ids(self, input_ids):
        """Forward pass from pre-tokenised ids, with attentions materialised.

        Callers that construct sequences from token ids need this: passing text
        to the tokenizer again can shift the token boundary, and an off-by-one
        in the sequence period silently invalidates every position-indexed
        measurement made against it.

        Shares the eager-attention handling with `_forward_with_hooks`; see
        that method for why sdpa is not sufficient.
        """
        import torch
        inputs = {"input_ids": input_ids.to(self._model.device)}
        config = self._model.config
        previous = getattr(config, "_attn_implementation", None)
        if previous != "eager":
            config._attn_implementation = "eager"
        try:
            with torch.no_grad():
                return self._model(**inputs, output_hidden_states=True,
                                   output_attentions=True)
        finally:
            if previous is not None:
                config._attn_implementation = previous

    def _attentions_or_reason(
        self, outputs: Any, layer: int
    ) -> tuple[Optional[Any], Optional[str]]:
        """The attention tensor for a layer, or (None, why it is unavailable).

        Callers that want attentions must handle the absent case explicitly.
        A None here means "not measured", which is different from a measured
        zero, and conflating the two is how a stub ends up reporting a result.
        """
        attentions = getattr(outputs, "attentions", None)
        if attentions is None:
            return None, (
                "The model returned no attentions tuple at all."
            )
        if layer >= len(attentions):
            return None, (
                f"Layer {layer} is out of range; the model returned "
                f"{len(attentions)} layers."
            )
        tensor = attentions[layer]
        if tensor is None:
            return None, (
                f"Layer {layer} attention is None. The sdpa attention path "
                f"does not materialise attention weights; eager is required."
            )
        return tensor, None

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
            attn, unavailable = self._attentions_or_reason(outputs, layer)
            if attn is None:
                # Previously this indexed into a None tuple and raised
                # "'NoneType' object is not subscriptable" from inside this
                # function, with nothing to say that attention had not been
                # available at all.
                raise RuntimeError(
                    f"Attention patterns unavailable for layer {layer}: "
                    f"{unavailable}"
                )
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
                "provenance": "live",
                "top_tokens": [
                    {"token": self._tokenizer.decode([idx]), "logit": round(float(l), 3)}
                    for l, idx in zip(top5.values, top5.indices)
                ],
                "top_token": self._tokenizer.decode([top5.indices[0]]),
            }

        # Mock mode. These are fixed fixtures, not measurements: they are
        # labelled "seeded" and carry validation/publication ineligibility so
        # they can never be mistaken for evidence. Only the three exact
        # prefixes below are recognised -- deliberately no name-parsing
        # heuristic, because synthesising a plausible IOI answer for arbitrary
        # names would be a fake result dressed as a finding.
        #
        # Rejected on this branch: a `_match_ioi_prompt` regex that parsed
        # "When Xavier and Yolanda ... gave a drink to" and returned
        # "Yolanda" with logit 8.2. Three tests demanded it. It would have made
        # the mock indistinguishable from a real forward pass on exactly the
        # benchmark the platform claims to measure, which is a worse failure
        # than an honest fixture.
        for prefix, tokens in self._KNOWN_TOP_TOKENS.items():
            if prefix in prompt:
                return {
                    "prompt": prompt,
                    "provenance": "seeded",
                    "validation_eligible": False,
                    "publication_eligible": False,
                    "reason": ("Fixed mock fixture; no weights were run. "
                               "Not evidence."),
                    "top_tokens": [{"token": t, "logit": p * 10, "prob": p} for t, p in tokens],
                    "top_token": tokens[0][0],
                }
        return {
            "prompt": prompt,
            "provenance": "seeded",
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": ("No mock fixture matches this prompt and no weights "
                       "were run. Not evidence."),
            "top_tokens": [{"token": " the", "logit": 4.5, "prob": 0.45}],
            "top_token": " the",
        }

    def mlp_patch_logits(
        self,
        clean_prompt: str,
        target_prompt: str,
        layer: int,
    ) -> Dict[str, Any]:
        """Causal-mediation measurement: patch one layer's whole MLP output.

        Runs three forward passes and returns the last-position logits of each:

        1. ``clean_prompt`` with a capture hook on ``h[layer].mlp``
        2. ``target_prompt`` unmodified
        3. ``target_prompt`` with ``h[layer].mlp`` output replaced by the value
           captured from pass 1

        A caller recovers a clean/corrupted logit difference in pass 3 to get
        the fraction of the effect that flows through that layer's MLP. This is
        the intervention Hanna et al. use to localise greater-than computation,
        and it is why the method exists: the previous greater-than pipeline
        reported a patch effect from a hardcoded ``{7: 0.82, 8: 0.91, 9: 0.78}``
        table keyed to the paper's own answer, so its "measurement" could never
        disagree with the paper it was reproducing.

        Raises ``LiveUnavailable`` rather than returning numbers when no weights
        are loaded. There is no fixture for this measurement.
        """
        if self.spec.mock_mode or self._model is None:
            raise LiveUnavailable(
                "mlp_patch_logits requires loaded weights: it is a forward-pass "
                "measurement and has no fixture."
            )

        import torch

        blocks = self._model.transformer.h
        if not 0 <= layer < len(blocks):
            raise ValueError(
                f"layer {layer} out of range for a model with {len(blocks)} blocks"
            )

        clean_inputs = self._tokenizer(clean_prompt, return_tensors="pt").to(
            self._model.device
        )
        target_inputs = self._tokenizer(target_prompt, return_tensors="pt").to(
            self._model.device
        )

        captured: Dict[str, Any] = {}

        def capture_hook(_module, _inp, output):
            captured["mlp"] = output.detach()
            return output

        mlp = blocks[layer].mlp
        handle = mlp.register_forward_hook(capture_hook)
        try:
            with torch.no_grad():
                clean_logits = self._model(**clean_inputs).logits[0, -1, :]
        finally:
            handle.remove()

        if "mlp" not in captured:
            raise LiveUnavailable("MLP capture hook did not fire; no value to patch.")
        donor = captured["mlp"]

        # A patch is only valid where the two sequences line up positionally.
        # Clean and target are normally the same length (both are
        # "The event lasted from NNNN to NN"), but silently broadcasting a
        # misaligned donor would produce a plausible number describing the
        # wrong computation.
        clean_len = clean_inputs["input_ids"].shape[1]
        target_len = target_inputs["input_ids"].shape[1]
        if clean_len != target_len:
            raise LiveUnavailable(
                f"cannot patch across unequal sequence lengths "
                f"(clean={clean_len}, target={target_len}); the positions would "
                f"not correspond"
            )

        with torch.no_grad():
            target_logits = self._model(**target_inputs).logits[0, -1, :]

        def patch_hook(_module, _inp, _output):
            return donor

        handle = mlp.register_forward_hook(patch_hook)
        try:
            with torch.no_grad():
                patched_logits = self._model(**target_inputs).logits[0, -1, :]
        finally:
            handle.remove()

        return {
            "clean_logits": clean_logits,
            "target_logits": target_logits,
            "patched_logits": patched_logits,
            "provenance": "live",
        }

    def logit_lens(self, prompt: str) -> Dict[str, Any]:
        """Apply the model's own unembedding to every intermediate residual state.

        This is the logit lens proper: take the hidden state after each block,
        push it through the model's final layer norm and unembedding matrix, and
        read off what token the model would predict if it stopped there. Nothing
        about the result is assumed -- in particular the argmax at each layer can
        and does disagree with the final layer, which is the interesting part.

        Returns per-layer ``top_token``, ``top_logit``, Shannon ``entropy`` (in
        nats, over the full vocabulary) and the residual ``norm``.

        Raises `LiveUnavailable` without weights rather than reporting a
        fabricated sweep. The previous implementation of the logit-lens pipeline
        returned `expected if progress > 0.60 else " the"` -- the correct answer
        hardcoded in for every layer past 60% depth, which made the sweep converge
        by construction, with entropy from `3.5 * exp(-2 * layer / n)` and a
        convergence layer of `int(n_layers * 0.65)` that is always the same
        integer regardless of the model or the prompt.
        """
        if self.spec.mock_mode or self._model is None:
            raise LiveUnavailable(
                "logit_lens requires loaded weights: it is a forward-pass "
                "measurement and has no fixture."
            )

        import torch

        outputs = self._forward_with_hooks(prompt)
        final_norm = self._model.transformer.ln_f
        unembed = self._model.lm_head
        final_logits = outputs.logits[0, -1, :]

        def _entropy_and_top(logits):
            # log_softmax, not softmax-then-log. GPT-2's intermediate states
            # produce logit spans of ~70, and float32 softmax underflows roughly
            # 50,000 of the 50,257 probabilities to exact zero. Taking log of a
            # clamped zero and multiplying by that zero is a NaN waiting to
            # happen; working in log space keeps the zeros finite.
            logp = torch.log_softmax(logits, dim=-1)
            entropy = float(-(logp.exp() * logp).sum())
            top = int(torch.argmax(logits))
            return entropy, top, float(logits[top])

        layers: List[Dict[str, Any]] = []
        with torch.no_grad():
            # Every intermediate state goes through the lens proper: final layer
            # norm, then unembed.
            for index, hidden in enumerate(outputs.hidden_states[:-1]):
                vector = hidden[0, -1, :]
                entropy, top, top_logit = _entropy_and_top(unembed(final_norm(vector)))
                layers.append({
                    "layer": index,
                    "top_token": self._tokenizer.decode([top]),
                    "top_logit": round(top_logit, 4),
                    "entropy": round(entropy, 4),
                    "norm": round(float(vector.norm()), 4),
                })

            # The last entry is the model's own output, read from `logits`
            # rather than recomputed. transformers collects the final hidden
            # state into `hidden_states` before `ln_f` is applied in some
            # versions and after it in others, so pushing it through the lens
            # again can silently disagree with the real forward pass -- it did,
            # by 84 logits. Taking the value the model actually produced makes
            # the last row a genuine check on every row above it.
            entropy, top, top_logit = _entropy_and_top(final_logits)
            layers.append({
                "layer": len(outputs.hidden_states) - 1,
                "top_token": self._tokenizer.decode([top]),
                "top_logit": round(top_logit, 4),
                "entropy": round(entropy, 4),
                "norm": round(float(outputs.hidden_states[-1][0, -1, :].norm()), 4),
                "is_model_output": True,
            })

        return {
            "prompt": prompt,
            "provenance": "live",
            "layers": layers,
            "final_top_token": layers[-1]["top_token"],
        }

    def patch_activation(self, prompt: str, layer: int, neuron_index: int, patch_value: float) -> PatchResult:
        """Overwrite one coordinate of one layer's MLP output.

        What this writes, precisely: the method registers a **forward** hook on
        `transformer.h[layer].mlp`, so `output` is the module's return value --
        after `c_proj`, `d_model` wide, sitting in the residual stream. Measured:

            MLP forward-hook output shape  : (1, seq, 768)   # d_model
            internal MLP hidden (4*d_model): 3072

        So `neuron_index` is a **residual-stream coordinate, not an MLP neuron**.
        A GPT-2 MLP neuron lives in the 3072-wide post-GELU hidden and is not
        reachable through this hook. Both an ablation of a neuron and an
        overwrite of a residual channel produce a finite number; only one is
        what a caller asking for "a neuron" means. The parameter name is kept
        for compatibility with existing callers, and
        `test_activation_patch_placement.py` pins the distinction so it cannot
        quietly regress.

        An out-of-range index raises rather than returning an unchanged
        result. The previous guard skipped the write and returned a well-formed
        `PatchResult` with `delta == 0.0`, which is indistinguishable from an
        intervention that genuinely had no effect.
        """
        if not self.spec.mock_mode and self._model is not None:
            import torch
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

            width = self.spec.d_model
            if not isinstance(neuron_index, int) or not 0 <= neuron_index < width:
                raise IndexError(
                    f"neuron_index {neuron_index!r} is outside the residual "
                    f"stream of width {width} (valid: 0..{width - 1}). This "
                    f"method patches the MLP output at transformer.h[{layer}]"
                    f".mlp, not the {4 * width}-wide internal hidden. "
                    f"Refusing rather than returning delta=0.0, which would "
                    f"be indistinguishable from an intervention with no effect."
                )

            with torch.no_grad():
                # Eager, matching every other intervention on this adapter.
                # `patch_activation` was the one site that did not force it, so
                # its `delta` was computed under sdpa while `patch_head_output`
                # and `run_isolated_circuit` computed theirs under eager. Two
                # methods on the same adapter therefore reported numbers from
                # different attention kernels, and comparing them was comparing
                # across an unrecorded change of implementation. Measured
                # divergence on a layer-5 residual-coordinate patch: 0.0625.
                previous_impl = getattr(
                    self._model.config, "_attn_implementation", None)
                self._model.config._attn_implementation = "eager"
                try:
                    orig_logits = self._model(**inputs).logits[0, -1, :]
                finally:
                    self._model.config._attn_implementation = previous_impl
            orig_top = orig_logits.topk(1)
            original_logit = float(orig_top.values[0])
            top_token_before = self._tokenizer.decode([orig_top.indices[0]])

            def patch_hook(module, input, output):
                # Copy rather than mutate in place: the returned tensor may be
                # referenced elsewhere, and an in-place write from a forward
                # hook is invisible to anything holding that reference.
                patched = output.clone()
                patched[0, -1, neuron_index] = patch_value
                return patched

            layer_module = self._model.transformer.h[layer].mlp
            handle = layer_module.register_forward_hook(patch_hook)

            try:
                previous_impl = getattr(
                    self._model.config, "_attn_implementation", None)
                self._model.config._attn_implementation = "eager"
                try:
                    with torch.no_grad():
                        patched_logits = self._model(
                            **inputs).logits[0, -1, :]
                finally:
                    self._model.config._attn_implementation = previous_impl
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

    def capture_head_outputs(self, prompt: str) -> Dict[int, Dict[int, List[float]]]:
        """Capture every attention head's output vector in one forward pass.

        Returns ``{layer: {head_index: [floats]}}`` for the last token position.

        GPT-2 concatenates all head outputs into one tensor before projecting, so
        the input to ``attn.c_proj`` is exactly the concatenated per-head outputs
        and head ``h`` occupies the slice
        ``[h * d_head : (h + 1) * d_head]`` where ``d_head = d_model / num_heads``.

        This exists because there was no way to get a real head vector. ACDC
        worked around that by passing a *head* index to APIs whose parameter is
        ``neuron_index``, which silently indexed something else entirely:

          * ``get_activations(layer, neuron_index=head)`` reads
            ``hidden_states[layer][0, tok, :][head]`` -- a dimension of the
            residual stream (d_model = 768), not a head.
          * ``patch_activation(layer, neuron_index=head)`` writes
            ``transformer.h[layer].mlp`` output at index ``head`` -- MLP neuron
            ``head`` out of 3072, not a head out of 12.

        Both succeeded and returned plausible numbers, because 768 and 3072 are
        both larger than 12. That is what made the bug survivable: ACDC appeared
        to run on real weights while measuring neither the heads it named nor a
        consistent quantity across candidates. Pruning on that signal is pruning
        on noise.

        Raises `LiveUnavailable` without weights rather than returning zeros.
        """
        if self.spec.mock_mode or self._model is None:
            raise LiveUnavailable(
                "capture_head_outputs requires loaded weights: per-head output "
                "vectors are a forward-pass measurement and have no fixture."
            )

        import torch

        captured: Dict[int, Any] = {}
        hooks = []

        def make_hook(layer: int):
            def hook(_module, inp, _out):
                captured[layer] = inp[0].detach()
            return hook

        for layer in range(self.spec.num_layers):
            hooks.append(
                self._model.transformer.h[layer].attn.c_proj.register_forward_hook(
                    make_hook(layer)
                )
            )

        try:
            self._forward_with_hooks(prompt)
        finally:
            for handle in hooks:
                handle.remove()

        d_head = self.spec.d_model // self.spec.num_heads
        out: Dict[int, Dict[int, List[float]]] = {}
        for layer, merged in captured.items():
            last = merged[0, -1, :]
            out[layer] = {
                head: [round(float(v), 6) for v in last[head * d_head:(head + 1) * d_head]]
                for head in range(self.spec.num_heads)
            }
        return out

    def _forward_with_grad(self, prompt: str) -> Any:
        """Forward pass with autograd enabled.

        `_forward_with_hooks` wraps the model call in `torch.no_grad()` because
        every measurement it serves is a read-only observation and building the
        graph for 12 layers of GPT-2 wastes memory for nothing. A gradient is the
        one thing that genuinely needs the graph, so this is a separate entry
        point rather than a flag on the shared helper: flipping `no_grad` off
        for all callers would make a slow path the default and, worse, would let
        a future caller mutate the model by accident.

        Eager attention is forced for the same reason as elsewhere, so behaviour
        matches `_forward_with_hooks` exactly.
        """
        import torch
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        config = self._model.config
        previous = getattr(config, "_attn_implementation", None)
        if previous != "eager":
            config._attn_implementation = "eager"
        try:
            with torch.enable_grad():
                return self._model(**inputs, output_hidden_states=True,
                                   output_attentions=True)
        finally:
            if previous is not None:
                config._attn_implementation = previous

    def capture_head_outputs_with_grad(
        self,
        prompt: str,
        target_token_id: int,
        baseline_token_id: int,
    ) -> Dict[int, Dict[int, List[float]]]:
        """Per-head output vectors *and* d(metric)/d(head output) in one pass.

        The metric is the logit difference ``target - baseline`` at the final
        position. Returns ``{layer: {head: {"activation": [...],
        "grad": [...]}}}``.

        Why this exists
        ---------------
        Attribution patching needs ``d Metric / d z``, a gradient. The algorithm
        was substituting ``metric_delta / (|z_clean| + |z_corrupted|)`` -- a
        global metric change divided by an activation magnitude, which is not a
        derivative of anything.

        Worse, that expression is algebraically degenerate. When a component's
        clean and corrupted activations straddle zero, ``|c - r| == |c| + |r|``
        exactly, so

            attribution = |c - r| * metric_delta / (|c| + |r|) = metric_delta

        identically -- for every such component, regardless of how small its
        activation change was. Measured on gpt2-small over all 144 heads: 9 of
        the top 10 attributions were exactly ``metric_delta``, so the ranking
        among them was decided by rounding rather than by importance.

        The real gradient has no such degeneracy, so it is computed with
        `torch.autograd.grad`. The concatenated head outputs are captured with
        `requires_grad_()` on the forward hook input, the metric is built from
        the logits, and one backward pass yields every component's gradient.

        Raises `LiveUnavailable` without weights. There is no fixture for a
        gradient, and substituting one would defeat the purpose.
        """
        if self.spec.mock_mode or self._model is None:
            raise LiveUnavailable(
                "capture_head_outputs_with_grad requires loaded weights: a "
                "gradient is a backward-pass measurement and has no fixture."
            )

        import torch

        captured: Dict[int, Any] = {}
        hooks = []

        def make_hook(layer: int):
            def hook(_module, inp, _out):
                tensor = inp[0]
                if tensor.requires_grad is False:
                    tensor = tensor.detach().requires_grad_(True)
                captured[layer] = tensor
            return hook

        for layer in range(self.spec.num_layers):
            hooks.append(
                self._model.transformer.h[layer].attn.c_proj.register_forward_hook(
                    make_hook(layer)
                )
            )

        try:
            outputs = self._forward_with_grad(prompt)
            logits = outputs.logits[0, -1, :]
            metric = logits[target_token_id] - logits[baseline_token_id]
            # `grad` is taken with respect to the captured tensors directly, so
            # the backward pass stops there instead of propagating through the
            # remaining 12 layers into the embedding. The result is identical for
            # these leaves and materially cheaper.
            grads = torch.autograd.grad(
                metric, list(captured.values()), allow_unused=True)
            grad_by_layer = dict(zip(captured.keys(), grads))

            d_head = self.spec.d_model // self.spec.num_heads
            result: Dict[int, Dict[int, Dict[str, List[float]]]] = {}
            for layer, merged in captured.items():
                last = merged[0, -1, :]
                layer_grad = grad_by_layer.get(layer)
                grad_slice = (None if layer_grad is None
                              else layer_grad[0, -1, :])
                per_head: Dict[int, Dict[str, List[float]]] = {}
                for head in range(self.spec.num_heads):
                    lo, hi = head * d_head, (head + 1) * d_head
                    per_head[head] = {
                        # `.detach()` before `float()`: the captured tensor still
                        # carries requires_grad, and converting it to a Python
                        # scalar warns about unexpected behaviour.
                        "activation": [round(float(v), 6)
                                       for v in last[lo:hi].detach()],
                        "grad": ([] if grad_slice is None
                                 else [round(float(v), 8) for v in grad_slice[lo:hi]]),
                    }
                result[layer] = per_head
            return result
        finally:
            for handle in hooks:
                handle.remove()

    def patch_head_output(
        self, prompt: str, layer: int, head_index: int,
        patch_vector: Optional[List[float]] = None,
        score_token_id: Optional[int] = None,
        position: Optional[int] = None,
    ) -> PatchResult:
        """Patch the output of a specific attention head (head-level causal intervention).

        Scope
        -----
        `position` selects which sequence positions the write covers, and it
        applies to **both** modes:

            position=None  every position of every batch row (the default)
            position=<int> that position in every batch row

        It used to apply to neither consistently. Zero ablation wrote
        `x[..., start:end]` -- all positions -- while vector injection wrote
        `x[0, -1, start:end]`, batch 0's final position only. Both are reached
        by omitting an argument, so the difference was invisible at the call
        site and the two modes were not comparable:

            patch_vector=None      delta = -1.687500
            patch_vector=[0.0]*64  delta = -0.937500

        Defaults have not moved for zero ablation, which already meant all
        positions. They have for vector injection, which previously meant the
        last position -- see `test_patch_scope_and_indexing.py`.

        Validation
        ----------
        `layer`, `head_index` and the length of `patch_vector` are checked
        before anything is computed. An out-of-range head previously returned
        `delta: 0.0` and a short vector broadcast across the whole head.

        `score_token_id` selects which token's logit `original_logit` /
        `patched_logit` / `delta` refer to. Without it they report the logit of
        whatever token the *unpatched* run predicted, which is the wrong quantity
        for any clean/corrupted comparison: the interesting question is what
        happened to a token the unpatched run was not predicting. Path-patching
        a head into a corrupted run needs the target token's score, so ACDC
        passes it.
        """
        if not self.spec.mock_mode and self._model is not None:
            import torch
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

            # Both forward passes below must run the same attention kernel as
            # the capture that produced `patch_vector`.
            #
            # `_forward_with_hooks` forces eager so attention weights are
            # materialised; this method used the model default (sdpa), so a
            # captured vector and the tensor it was patched back into came from
            # two different kernels over the same weights. Measured at layer 6,
            # head 6: 87.5% of the 64 components differed, worst 3.7e-04 --
            # well above the capture's own 6-decimal rounding. Feeding a
            # captured vector back in therefore perturbed the tensor it came
            # from, and the scored logit moved by 0.0625.
            #
            # That constant kernel-disagreement term sits inside every
            # head-level causal result, because each one compares a capture
            # against a patched pass. Small, universal, and indistinguishable
            # from a weak causal effect.
            #
            # Validated before anything is computed. Both of these used to fail
            # late, and one failed silently:
            #
            #  * An out-of-range head computed a slice past the end of the
            #    tensor; `head_patch_hook`'s `if x.shape[-1] < end` guard then
            #    returned the input untouched and the caller received
            #    `delta: 0.0` -- identical to what ablating a causally inert
            #    head returns. Measured at head 12 (the model has 0..11):
            #    delta +0.000000, against +0.312500 for real head 11. A typo
            #    in a discovered head list read as "this head does nothing".
            #
            #  * A wrong-length vector broadcast: `x[0, -1, start:end] = v`
            #    accepts a length-1 `v` and fills all 64 components with it.
            #    Measured, `[0.5]` and `[0.5] * 64` both give -0.937500.
            _d_head = self.spec.d_model // self.spec.num_heads
            if not isinstance(layer, int) or not (0 <= layer < self.spec.num_layers):
                raise ValueError(
                    f"layer {layer!r} is out of range; this model has "
                    f"{self.spec.num_layers} layers "
                    f"(0..{self.spec.num_layers - 1})")
            if not isinstance(head_index, int) or not (
                    0 <= head_index < self.spec.num_heads):
                raise ValueError(
                    f"head {head_index!r} is out of range; layer {layer} has "
                    f"{self.spec.num_heads} heads "
                    f"(0..{self.spec.num_heads - 1}). Patching a head outside "
                    f"this range previously returned delta 0.0 rather than "
                    f"failing, which is indistinguishable from an inert head.")
            if patch_vector is not None and len(patch_vector) != _d_head:
                raise ValueError(
                    f"patch_vector has length {len(patch_vector)}; a GPT-2 "
                    f"head occupies {_d_head} components (d_model "
                    f"{self.spec.d_model} / {self.spec.num_heads} heads). A "
                    f"shorter vector would broadcast across the whole head "
                    f"rather than fail.")

            config = self._model.config
            previous_impl = getattr(config, "_attn_implementation", None)
            if previous_impl != "eager":
                config._attn_implementation = "eager"
            try:
                with torch.no_grad():
                    orig_logits = self._model(**inputs).logits[0, -1, :]
                orig_top = orig_logits.topk(1)
                scored_index = orig_top.indices[0] if score_token_id is None else score_token_id
                original_logit = float(orig_logits[scored_index])
                top_token_before = self._tokenizer.decode([orig_top.indices[0]])

                # GPT-2 Attention head output patching via attn.c_proj pre-hook.
                # (A post-hook on `attn` cannot work: its output is a tuple and
                # mutating it neither applies nor returns correctly.)
                def head_patch_hook(module, inp):
                    # Merged-head input shape: [batch, seq, d_model]
                    # Head dimension: d_model / num_heads
                    d_head = self.spec.d_model // self.spec.num_heads
                    start = head_index * d_head
                    end = (head_index + 1) * d_head
                    x = inp[0]
                    if x.shape[-1] < end:
                        return inp
                    x = x.clone()
                    if patch_vector is not None:
                        # Inject specific vector (e.g. mean ablation vector)
                        v = torch.tensor(patch_vector, device=x.device, dtype=x.dtype)
                    else:
                        v = 0.0
                    # One scope for both modes.
                    #
                    # They used to differ, and the difference was invisible at
                    # the call site because both are reached by omitting an
                    # argument: zero ablation wrote `x[..., start:end]` -- every
                    # position, every batch row -- while vector injection wrote
                    # `x[0, -1, start:end]`, batch 0's final position only.
                    #
                    # Measured, both meaning "zero head 3 of layer 5":
                    #     patch_vector=None      delta = -1.687500
                    #     patch_vector=[0.0]*64  delta = -0.937500
                    #
                    # So a caller comparing "head ablated" against "head set
                    # to v" was comparing two different experiments, and
                    # neither result said which it had run. `position=None`
                    # now means all positions for both, so the modes are
                    # comparable; pass an index to restrict deliberately.
                    if position is None:
                        x[..., start:end] = v
                    else:
                        x[:, position, start:end] = v
                    return (x,) + tuple(inp[1:])

                layer_module = self._model.transformer.h[layer].attn.c_proj
                handle = layer_module.register_forward_pre_hook(head_patch_hook)

                try:
                    with torch.no_grad():
                        patched_logits = self._model(**inputs).logits[0, -1, :]
                    patched_logit_for_orig_token = float(patched_logits[scored_index])
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
            finally:
                # Restore the caller's attention implementation. Left as eager,
                # this would silently change every later measurement that does
                # not force its own -- the same class of bug in reverse.
                if previous_impl is not None:
                    config._attn_implementation = previous_impl

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

            # Parse head list: "L9H6" -> (9, 6).
            #
            # Unreadable or out-of-range labels were previously dropped by a
            # bare `except: continue`. Measured consequence: ["garbage"] and
            # ["L99H99"] both returned a top logit identical to an *empty*
            # circuit, and ["L9H6", "garbage"] returned exactly ["L9H6"] alone.
            # The caller got a complete, plausible result describing a circuit
            # they had not asked for, with nothing to distinguish the two.
            #
            # Now every rejected label is named in `rejected_heads`, and a
            # circuit with no readable heads at all is refused rather than
            # measured.
            n_layers = self.spec.num_layers
            n_heads = self.spec.num_heads
            active_heads = set()
            rejected_heads: List[str] = []

            for h_str in head_list:
                label = str(h_str).strip().upper()
                parsed = _parse_head_label(label, n_layers, n_heads)
                if parsed is None:
                    rejected_heads.append(str(h_str))
                else:
                    active_heads.add(parsed)

            # A refusal is warranted only when the caller *supplied* heads and
            # none could be read. An explicitly empty `head_list` is a
            # different request -- isolate nothing, ablate every head -- and is
            # a well-defined measurement, so it runs and is labelled as such.
            # Conflating the two turned a legitimate control condition into an
            # error.
            if head_list and not active_heads:
                return {
                    "status": "unavailable",
                    "provenance": "unavailable",
                    "measured": False,
                    "validation_eligible": False,
                    "publication_eligible": False,
                    "rejected_heads": rejected_heads,
                    "reason": (
                        "No head in the requested circuit could be read against "
                        f"this model ({n_layers} layers, {n_heads} heads per "
                        f"layer), so there was no circuit to isolate and nothing "
                        f"was measured. Rejected: {rejected_heads}."
                    ),
                }

            handles = []
            d_head = self.spec.d_model // self.spec.num_heads

            def make_mask_hook(layer_idx):
                def mask_hook(module, inp):
                    # Merged-head c_proj input shape: [batch, seq, d_model].
                    # Pre-hook replaces the input so the patched tensor flows
                    # through (a post-hook on `attn` sees a tuple output).
                    x = inp[0]
                    x = x.clone()
                    for h_idx in range(self.spec.num_heads):
                        if (layer_idx, h_idx) not in active_heads:
                            start = h_idx * d_head
                            end = (h_idx + 1) * d_head
                            if x.shape[-1] >= end:
                                x[:, :, start:end] = 0.0
                    return (x,) + tuple(inp[1:])
                return mask_hook

            # Register hooks for all attention layers
            for i in range(n_layers):
                layer_module = self._model.transformer.h[i].attn.c_proj
                handles.append(layer_module.register_forward_pre_hook(make_mask_hook(i)))

            # Pin the attention kernel to the one `capture_head_outputs` uses.
            # Without this the isolation ran under sdpa while the activation it
            # is compared against was captured under eager -- the same
            # constant kernel-disagreement term fixed for `patch_head_output`.
            config = self._model.config
            previous_impl = getattr(config, "_attn_implementation", None)
            if previous_impl != "eager":
                config._attn_implementation = "eager"
            try:
                # Also optionally ablate MLPs if they aren't in the "circuit"
                # For IOI, MLPs are often included or excluded depending on paper
                # We'll just run with MLPs active for now unless requested
                inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
                with torch.no_grad():
                    logits = self._model(**inputs).logits[0, -1, :]

                top5 = logits.topk(5)
                result = {
                    "status": "ok",
                    "top_token": self._tokenizer.decode([top5.indices[0]]),
                    "top_tokens": [
                        {"token": self._tokenizer.decode([idx]), "logit": round(float(l), 3)}
                        for l, idx in zip(top5.values, top5.indices)
                    ],
                    # The circuit that actually ran, so a partially-readable
                    # request is attributable rather than silently different.
                    "active_heads": sorted(
                        "L%dH%d" % (l, h) for l, h in active_heads
                    ),
                    "circuit_isolated": True,
                    "all_heads_ablated": not active_heads,
                }
                if rejected_heads:
                    result["rejected_heads"] = rejected_heads
                    result["reason"] = (
                        "Some requested heads could not be read against this "
                        f"model and were excluded from the circuit that ran: "
                        f"{rejected_heads}. `active_heads` names what was "
                        f"actually isolated."
                    )
                return result
            finally:
                for h in handles: h.remove()
                if previous_impl is not None:
                    config._attn_implementation = previous_impl

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
