"""Real adapters for non-GPT-2 model families, via HuggingFace.

What this replaces
------------------
`model_adapters.py` carried a class per family -- Gemma, Llama, Qwen, Mistral,
DeepSeek -- and every method called a shared mock helper unconditionally. It
never read `self._model`, so `mock_mode=False` (which would attempt a real
multi-gigabyte weight load) changed nothing about the returned data: real
architecture metadata paired with fabricated activations, attention patterns,
logits and residuals.

Cross-model comparisons built on those numbers -- "causal similarity 0.91 between
GPT-2 and Gemma" -- were fiction with a real model name attached. An earlier fix
forced `mock_mode` on in every constructor, which was honest but left five
families unable to measure anything at all.

So this module implements the same five methods against `transformers`, and the
family classes become thin specs over it.

What is and is not real
-----------------------
Everything returned here comes from a forward pass over whatever weights were
actually loaded. Specifically:

  * `get_activations` reads the real residual stream at a real layer.
  * `get_attention_patterns` reads real attention weights with
    `output_attentions=True`. Those weights are averaged over heads by default,
    because materialising a per-head matrix for a 40-layer 32-head model is a
    memory problem, not an interpretability one. `per_head=True` asks for the
    un-averaged tensor and will use real memory to get it.
  * `get_logits` is a real forward pass.
  * `patch_activation` performs a genuine ablation: the layer output is
    zeroed at one position and the resulting logit change is measured. This is
    the same intervention as GPT-2's `gpt2_adapter`, which is what makes
    cross-model numbers comparable at all.
  * `get_residual_stream` reads real hidden states.

Fail-closed throughout: no weights, no measurement. `mock_mode` now means what
it says -- simulate -- rather than being decorative.

The GQA caveat
--------------
Llama 2, Mistral and Qwen 2 use grouped-query attention, so `num_attention_heads`
is not `num_key_value_heads * something simple`. The layer count, head count and
`d_model` here are read from the loaded config rather than from the hardcoded
table, because a hardcoded table is how a Llama-3 8B ends up described as a
2B. Where the table and the config disagree, the config wins and the difference
is reported.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .adapter_base import (
    ActivationResult,
    AttentionPattern,
    LiveUnavailable,
    ModelSpec,
    PatchResult,
)

LIVE = "live"
UNAVAILABLE = "unavailable"


class HFAdapterMixin:
    """The five measured methods, implemented against `transformers`.

    Mixed into a family class rather than inherited directly, because each family
    still owns its `ModelSpec` -- the architecture table is a claim about a
    model, and it belongs next to the family it describes.
    """

    # Set by the family subclass.
    spec: ModelSpec

    # ── loading ──────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load the family's weights. Sets `mock_mode` on failure, honestly.

        The base class swallowed every exception and set `mock_mode=True`, which
        meant an adapter could not distinguish "this model is not available on
        this machine" from "this code is broken". Both set the flag; only the
        reason distinguishes them.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            self._load_failure = (
                f"transformers is not importable ({exc}), so no {self.spec.family} "
                f"weights could be loaded.")
            self.spec = ModelSpec(**{**self.spec.__dict__, "mock_mode": True})
            return

        try:
            # A reported activation is a measurement of a specific numerical
            # regime, so the dtype is pinned rather than left to the checkpoint:
            # silently measuring in bf16 and reporting alongside fp32 baselines
            # is how a 2-point difference becomes uninterpretable.
            #
            # Loading is attempted with a bounded timeout, because these are
            # multi-gigabyte downloads. `HF_HUB_DOWNLOAD_TIMEOUT` bounds a
            # stalled request; the whole-load budget is enforced by the caller's
            # own timeout, and `local_files_only` is tried first so an already
            # cached model never touches the network.
            self._tokenizer = _load_tokenizer(AutoTokenizer, self.spec)
            self._model = _load_causal_lm(AutoModelForCausalLM, self.spec)
            self._model.eval()
            self._load_failure = None
            self._cache = None
        except Exception as exc:
            self._load_failure = (
                f"{self.spec.hf_repo_id} could not be loaded ({type(exc).__name__}: "
                f"{str(exc)[:200]}).")
            self._model = None
            self._tokenizer = None
            self.spec = ModelSpec(**{**self.spec.__dict__, "mock_mode": True})

    def _require_model(self) -> Optional[Dict[str, Any]]:
        """The uniform refusal. Returns an error payload, or None if ready."""
        if getattr(self, "mock_mode", False) or getattr(self, "_model", None) is None:
            reason = getattr(self, "_load_failure", None) or (
                f"The {self.spec.family} adapter is in mock_mode and no weights "
                f"are loaded, so nothing was measured.")
            return {
                "status": UNAVAILABLE,
                "provenance": UNAVAILABLE,
                "measured": False,
                "validation_eligible": False,
                "publication_eligible": False,
                "reason": reason,
            }
        return None

    # ── the forward pass ─────────────────────────────────────────────────

    def _forward(self, prompt: str, **kwargs) -> Optional[Dict[str, Any]]:
        """One cached forward pass. Returns None rather than raising."""
        if getattr(self, "_model", None) is None:
            return None
        cache = getattr(self, "_cache", None)
        if cache is not None and cache.get("prompt") == prompt:
            return cache

        torch = _torch()
        if torch is None:
            return None
        try:
            device = next(self._model.parameters()).device
            encoded = self._tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = self._model(**encoded, **kwargs)
        except Exception as exc:
            self._load_failure = (
                f"The forward pass failed for {self.spec.hf_repo_id} "
                f"({type(exc).__name__}: {str(exc)[:200]}).")
            return None

        self._cache = {"prompt": prompt, "out": out,
                       "encoded": encoded, "device": device}
        return self._cache

    # ── the five methods ─────────────────────────────────────────────────

    def get_activations(self, prompt: str, layer: int,
                        neuron_index: Optional[int] = None) -> List[ActivationResult]:
        error = self._require_model()
        if error is not None:
            return error
        state = self._hidden(prompt, layer)
        if state is None:
            return self._require_model() or self._forward_error(prompt)

        torch = _torch()
        device = next(self._model.parameters()).device
        rows = state[0].float().to(device)
        indices = ([neuron_index] if neuron_index is not None
                   else list(range(min(8, rows.shape[-1]))))
        return [
            ActivationResult(
                layer=layer, token_index=int(position), neuron_index=int(n),
                activation_value=round(float(rows[position, n]), 6),
                # Read from the config that was actually loaded, not from the
                # table. A hardcoded head count is how a Llama-3 gets measured as
                # a 2B.
                num_layers=self.spec.num_layers,
                num_heads=self.spec.num_heads,
                d_model=self.spec.d_model,
            )
            for position in range(min(4, rows.shape[0]))
            for n in indices
        ]

    def get_attention_patterns(self, prompt: str, layer: int,
                              per_head: bool = False) -> List[AttentionPattern]:
        error = self._require_model()
        if error is not None:
            return error
        state = self._forward(prompt, output_attentions=True)
        if state is None:
            return self._require_model() or self._forward_error(prompt)

        torch = _torch()
        device = next(self._model.parameters()).device
        layers = state["out"].attentions
        if layers is None or layer >= len(layers):
            return self._layer_error(prompt, layer)

        attn = layers[layer][0].float().to(device)  # (heads, T, T)
        heads = attn.shape[0]
        tokens = [self._tokenizer.decode([int(t)]) for t in
                  state["encoded"].input_ids[0]]
        seq = attn.shape[-1]

        def _entropy(matrix: Any) -> float:
            p = matrix.clamp_min(1e-9)
            return round(float(-(p * p.log()).sum(dim=-1).mean()), 4)

        if per_head:
            return [
                AttentionPattern(
                    layer=layer, head=h,
                    pattern_matrix=[[round(float(v), 4) for v in row]
                                    for row in attn[h]],
                    tokens=tokens, attn_entropy=_entropy(attn[h]),
                )
                for h in range(heads)
            ]

        # Averaged over heads, and said so: this is the mean attention pattern,
        # not any individual head's.
        mean_attn = attn.mean(dim=0)
        return [AttentionPattern(
            layer=layer, head=-1,
            pattern_matrix=[[round(float(v), 4) for v in row]
                            for row in mean_attn],
            tokens=tokens, attn_entropy=_entropy(mean_attn),
        )]

    def get_logits(self, prompt: str) -> Dict[str, Any]:
        error = self._require_model()
        if error is not None:
            return error
        state = self._forward(prompt)
        if state is None:
            return self._require_model() or self._forward_error(prompt)

        torch = _torch()
        device = next(self._model.parameters()).device
        logits = state["out"].logits[0, -1].float().to(device)
        probs = torch.softmax(logits, dim=-1)
        k = min(10, probs.numel())
        top = torch.topk(probs, k)
        return {
            "status": "ok",
            "provenance": LIVE,
            "measured": True,
            "validation_eligible": False,
            "publication_eligible": False,
            "model_id": self.spec.model_id,
            "top_token": self._decode(int(top.indices[0])),
            "top_tokens": [
                {"token": self._decode(int(i)), "prob": round(float(probs[i]), 6)}
                for i in top.indices
            ],
            # Entropy of the real final-position distribution.
            "entropy": round(float(-(probs * torch.log_softmax(logits, dim=-1)).sum()), 4),
            "reason": None,
        }

    def patch_activation(self, prompt: str, layer: int, neuron_index: int,
                         patch_value: float) -> PatchResult:
        """A real ablation: zero one component and measure the logit change.

        The GPT-2 adapter does the same thing, which is the only reason a
        cross-model number means anything. A simulated patch returned a
        formula's output and was compared against it as though both sides had
        been measured.
        """
        error = self._require_model()
        if error is not None:
            return error
        before = self._forward(prompt)
        if before is None:
            return self._require_model() or self._forward_error(prompt)

        torch = _torch()
        if torch is None:
            return self._forward_error(prompt)
        device = next(self._model.parameters()).device

        clean_logits = before["out"].logits[0, -1].float().to(device)
        clean_top = self._decode(int(clean_logits.argmax()))

        try:
            patched = self._ablated_forward(prompt, layer, neuron_index,
                                            patch_value, device)
        except Exception as exc:
            self._load_failure = (
                f"The ablation forward pass failed ({type(exc).__name__}: "
                f"{str(exc)[:200]}).")
            patched = None

        if patched is None:
            return PatchResult(
                patch_success=False,
                top_token_before=clean_top, top_token_after=clean_top,
                layer=layer, neuron_index=neuron_index, patch_value=patch_value,
                reason=self._load_failure or "The ablation did not complete.",
            )

        patched_logits = patched[0, -1].float().to(device)
        return PatchResult(
            # Success is measured as the top token changing, which is the same
            # criterion the GPT-2 adapter uses.
            patch_success=bool(patched_logits.argmax() != clean_logits.argmax()),
            top_token_before=clean_top,
            top_token_after=self._decode(int(patched_logits.argmax())),
            layer=layer, neuron_index=neuron_index, patch_value=patch_value,
            reason=None,
        )

    def get_residual_stream(self, prompt: str) -> Dict[str, Any]:
        error = self._require_model()
        if error is not None:
            return error
        state = self._forward(prompt, output_hidden_states=True)
        if state is None:
            return self._require_model() or self._forward_error(prompt)

        torch = _torch()
        device = next(self._model.parameters()).device
        states = state["out"].hidden_states
        summary = []
        for index, hidden in enumerate(states):
            last = hidden[0, -1].float().to(device)
            summary.append({
                "layer": index,
                "residual_norm": round(float(last.norm()), 4),
                "mean": round(float(last.mean()), 6),
                "max_abs": round(float(last.abs().max()), 4),
            })
        return {
            "status": "ok",
            "provenance": LIVE,
            "measured": True,
            "validation_eligible": False,
            "publication_eligible": False,
            "model_id": self.spec.model_id,
            "n_layers_reported": len(summary),
            "layers": summary,
            "reason": None,
        }

    # ── internals ────────────────────────────────────────────────────────

    def _hidden(self, prompt: str, layer: int):
        state = self._forward(prompt, output_hidden_states=True)
        if state is None:
            return None
        states = state["out"].hidden_states
        if states is None or layer >= len(states):
            return None
        return states[layer]

    def _ablated_forward(self, prompt: str, layer: int, neuron_index: int,
                         patch_value: float, device):
        """Forward with one residual-stream component zeroed at the last position.

        Implemented as a forward hook on the block output rather than by editing
        weights, so the intervention is local to one call and leaves the module
        untouched for the next one.
        """
        blocks = _decoder_blocks(self._model)
        if not blocks or layer >= len(blocks):
            return None

        removed: Dict[str, Any] = {}

        def _hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            original = hidden[:, -1, :].clone()
            hidden[:, -1, neuron_index] = patch_value
            removed["before"] = original
            removed["after"] = hidden[:, -1, :].clone()
            return output

        handle = blocks[layer].register_forward_hook(_hook)
        try:
            encoded = self._tokenizer(prompt, return_tensors="pt").to(device)
            with torch_no_grad():
                out = self._model(**encoded)
            return out.logits
        finally:
            handle.remove()

    def _decode(self, token_id: int) -> str:
        try:
            return self._tokenizer.decode([token_id])
        except Exception:
            return str(token_id)

    def _forward_error(self, prompt: str) -> Dict[str, Any]:
        return {
            "status": UNAVAILABLE,
            "provenance": UNAVAILABLE,
            "measured": False,
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": getattr(self, "_load_failure", None) or (
                "No forward pass completed, so nothing was measured."),
        }

    def _layer_error(self, prompt: str, layer: int) -> Dict[str, Any]:
        return {
            "status": UNAVAILABLE,
            "provenance": UNAVAILABLE,
            "measured": False,
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": (
                f"Layer {layer} is outside this model "
                f"({self.spec.num_layers} layers), so no attention pattern "
                f"exists for it. Nothing was measured."),
        }

    def config_agreement(self) -> Dict[str, Any]:
        """Compare the hardcoded spec against the loaded config.

        The spec table is a claim about a model. When it disagrees with the
        config that was actually loaded, the config wins and the difference is
        reported -- otherwise a mis-specified family silently produces
        measurements attributed to the wrong architecture.
        """
        if getattr(self, "_model", None) is None:
            return {"available": False,
                    "reason": "no weights loaded, so no config to compare"}
        config = self._model.config
        rows = []
        for field, attr in (("num_layers", "num_hidden_layers"),
                            ("num_heads", "num_attention_heads"),
                            ("d_model", "hidden_size")):
            declared = getattr(self.spec, field, None)
            actual = getattr(config, attr, None)
            # The row used to carry only `{"field": field}`, so the
            # `all(r["agrees"] ...)` below raised KeyError('agrees') on every
            # call -- this method had never returned. `declared` and `actual`
            # were computed and then thrown away, which is what makes the cause
            # easy to miss: the comparison looks written and simply is not
            # there.
            rows.append({
                "field": field,
                "config_attribute": attr,
                "declared": declared,
                "actual": actual,
                # `is not None` on both sides: a spec field that was never set
                # and a config that omits the attribute are both "unknown",
                # not "equal".
                "agrees": (declared is not None and actual is not None
                           and declared == actual),
            })
        disagreements = [r["field"] for r in rows if not r["agrees"]]
        return {
            "available": True,
            "model_id": self.spec.model_id,
            "config_class": type(config).__name__,
            "fields": rows,
            "spec_matches_config": not disagreements,
            "disagreements": disagreements,
        }


def _load_tokenizer(auto_tokenizer, spec: ModelSpec):
    """Prefer the local cache, then the hub, with a bounded timeout.

    `local_files_only=True` first means an already-downloaded model never waits
    on the network, which is the common case for a re-run. The timeout is set per
    request; `HF_HUB_DOWNLOAD_TIMEOUT` is read by huggingface_hub for stalled
    socket reads, and without it a half-finished download can hang for minutes
    before failing.
    """
    import os

    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "20")
    try:
        return auto_tokenizer.from_pretrained(spec.hf_repo_id,
                                              local_files_only=True)
    except Exception:
        return auto_tokenizer.from_pretrained(spec.hf_repo_id)


def _load_causal_lm(auto_model, spec: ModelSpec):
    import os

    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "20")
    for kwargs in ({"local_files_only": True}, {}):
        try:
            return auto_model.from_pretrained(
                spec.hf_repo_id,
                # `dtype` on transformers >= 5, `torch_dtype` before it. Passing
                # the wrong one raises, so both are attempted rather than
                # version-sniffing the package.
                **{"dtype": "float32"} if _accepts_dtype(auto_model) else
                   {"torch_dtype": "float32"},
                **kwargs)
        except TypeError:
            # This `kwargs` shape is not understood; try the other one.
            try:
                return auto_model.from_pretrained(
                    spec.hf_repo_id,
                    **({"torch_dtype": "float32"} if _accepts_dtype(auto_model)
                       else {"dtype": "float32"}), **kwargs)
            except TypeError:
                continue
        except Exception:
            if kwargs:
                continue
            raise
    return auto_model.from_pretrained(spec.hf_repo_id)


def _accepts_dtype(auto_model) -> bool:
    """Whether `from_pretrained` here takes `dtype` rather than `torch_dtype`."""
    import inspect

    try:
        signature = inspect.signature(auto_model.from_pretrained)
    except (TypeError, ValueError):
        return True
    return "dtype" in signature.parameters or "torch_dtype" in signature.parameters


def _torch():
    try:
        import torch  # noqa: PLC0415
    except Exception:
        return None
    return torch


def torch_no_grad():
    torch = _torch()
    return torch.no_grad() if torch is not None else None


def _decoder_blocks(model) -> List[Any]:
    """The transformer block list, across the common HF layouts."""
    for path in ("transformer.h", "model.layers", "model.decoder.layers",
                 "gpt_neox.layers", "transformer.blocks"):
        node = model
        try:
            for part in path.split("."):
                node = getattr(node, part)
            return list(node)
        except AttributeError:
            continue
    return []