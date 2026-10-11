"""Runtime interpreter — runs inference through the hook framework.

This module orchestrates:
    - Hooks (Attention, MLP, Embedding, Residual, Logit)
    - Session tracking
    - Activation cache
    - Performance profiling
    - Event emission

No visualization logic lives here — only data capture.
"""

from __future__ import annotations

import torch
import numpy as np

from .dto import AttentionMap, NeuronActivation, TokenInfo
from .hook_framework import HookManager, AttentionHook, MLPHook
from .event_bus import bus, INFERENCE_STARTED, INFERENCE_FINISHED, ACTIVATION_CACHED
from .activation_cache import ActivationCache, CachedActivation
from .profiler import Profiler
from .session_manager import SessionManager, Session
from backend.repository.activation_repository import activation_repo, ActivationRecord


def run_inference(
    prompt: str,
    max_new_tokens: int = 10,
    model=None,
    tokenizer=None,
    hook_manager: HookManager | None = None,
    session: Session | None = None,
    cache: ActivationCache | None = None,
    profiler: Profiler | None = None,
) -> dict:
    """Run inference and return structured results.

    All parameters are injected (Dependency Injection) — the caller
    (main.py) wires them together.  This makes the interpreter pure
    logic with no global state.
    """
    if model is None or tokenizer is None:
        from .model_manager import get_model_and_tokenizer
        model, tokenizer = get_model_and_tokenizer()

    config = model.config
    num_layers = getattr(config, "n_layer", getattr(config, "num_hidden_layers", 12))
    num_heads = getattr(config, "n_head", getattr(config, "num_attention_heads", 12))

    bus.emit(INFERENCE_STARTED, prompt=prompt, model=model.config._name_or_path if hasattr(config, '_name_or_path') else "")

    # ── Tokenize ─────────────────────────────────────────────
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = input_ids = inputs["input_ids"]

    # ── Set up hooks ─────────────────────────────────────────
    if hook_manager is None:
        hook_manager = HookManager()
        hook_manager.bind(model)

    attention_store: list[torch.Tensor] = []
    mlp_store: list[torch.Tensor] = []

    # Register hooks for ALL layers
    for li in range(num_layers):
        hook_manager.register_hook(li, "attention", store=attention_store)
        hook_manager.register_hook(li, "mlp", store=mlp_store)

    # ── Forward pass ─────────────────────────────────────────
    with torch.no_grad():
        if profiler:
            with profiler.measure("forward_pass", layers=num_layers, tokens=input_ids.size(1)):
                outputs = model(input_ids, output_attentions=True, output_hidden_states=True)
        else:
            outputs = model(input_ids, output_attentions=True, output_hidden_states=True)

    # ── Extract tokens ───────────────────────────────────────
    all_tokens: list[TokenInfo] = []
    ids_list = input_ids[0].tolist()
    for tid in ids_list:
        t = tokenizer.decode([tid], skip_special_tokens=True)
        all_tokens.append(TokenInfo(text=t if t else "<|endoftext|>", id=tid))

    # ── Build attention maps ─────────────────────────────────
    attn_dtos: list[AttentionMap] = []
    cache_ids: list[str] = []

    # Each registered AttentionHook fires once per *layer* and appends one
    # [batch, heads, seq, seq] tensor, so this store is indexed by LAYER.
    #
    # The previous code walked it as though it were indexed by head --
    # `layer = idx // num_heads`, `head = idx % num_heads` -- which for a
    # 3-layer 4-head model filed all three captures under layer 0, heads
    # 0/1/2. Every map above layer 0 was confidently mislabelled, and
    # `attn_tensor[0]` dropped only the batch axis, so a [heads, seq, seq]
    # tensor was handed to a field declared list[list[float]] and pydantic
    # rejected it outright.
    #
    # Now: layer is the store index, and each head is sliced to its own
    # [seq, seq] matrix. Nothing is inferred from position.
    for li, attn_tensor in enumerate(attention_store):
        for hi in range(num_heads):
            # [batch, heads, seq, seq] -> [seq, seq] for this one head.
            mat = attn_tensor[0, hi].tolist()

            attn_dtos.append(AttentionMap(
                layer=li, head=hi, tokens=[t.text for t in all_tokens], matrix=mat,
            ))

            # Cache & Repository
            #
            # Inside the head loop, and storing THIS head's slice. It used to
            # sit outside the loop, where `hi` was whatever the last inner
            # iteration left behind and the whole [batch, heads, seq, seq]
            # tensor was filed under that single head -- a record whose
            # `shape` claimed one head and whose payload contained all of
            # them.
            head_tensor = attn_tensor[0, hi]

            if session:
                rec = ActivationRecord(
                    session_id=session.session_id,
                    prompt_id=session.session_id[:8],
                    layer=li,
                    head=hi,
                    component="attention",
                    shape=tuple(head_tensor.shape),
                    tensor=head_tensor,
                )
                activation_repo.save(rec)

            if cache and session:
                ca = CachedActivation(
                    session_id=session.session_id,
                    prompt_id=session.session_id[:8],
                    layer=li,
                    head=hi,
                    component="attention",
                    shape=list(head_tensor.shape),
                    tensor=head_tensor,
                    metadata={"model": config._name_or_path if hasattr(config, '_name_or_path') else ""},
                )
                cid = cache.put(ca)
                cache_ids.append(cid)
                bus.emit(ACTIVATION_CACHED, activation_id=cid, layer=li, head=hi)

    # ── Build neuron activations ─────────────────────────────
    n_neurons_per_layer = 8
    neuron_dtos: list[NeuronActivation] = []

    for li in range(num_layers):
        # One MLP capture per layer, in registration order, so the store is
        # indexed by layer. The previous filter was `idx // 1 == li`, which is
        # just `idx == li` wearing a disguise that reads as a division.
        layer_mlps = [mlp_store[li]] if li < len(mlp_store) else []
        if not layer_mlps:
            continue
        # Take the first MLP capture for this layer
        act_tensor = layer_mlps[0] if layer_mlps else None
        if act_tensor is None:
            continue

        # Per-token neuron activations: for each token position,
        # compute the activation of each sampled neuron
        act_squeezed = act_tensor.squeeze(0)  # [seq_len, inter_dim]
        seq_len = act_squeezed.size(0)
        actual_inter_dim = act_squeezed.size(1)

        for ni in range(n_neurons_per_layer):
            neuron_idx = int((ni / n_neurons_per_layer) * actual_inter_dim)
            #
            # Report the neuron actually read, not the sample counter.
            # `ni` strides across the width (with 8 samples over a 128-wide
            # MLP, neuron_idx was 0, 16, 32, ... 112), but the DTO carried
            # `index=ni`, so index=3 was really neuron 48. The value was
            # computed correctly and labelled falsely, which is the worst
            # combination: it looks like data.
            #
            # Neuron indices now stay within range and each one names itself.
            if neuron_idx >= actual_inter_dim:
                continue
            # Activation of this neuron at each token position
            token_acts = torch.tanh(act_squeezed[:, neuron_idx] * 3.0)
            token_activations = [float(a) for a in token_acts.tolist()]
            # Averaged activation (backward-compatible scalar)
            #
            # Mean over the *token* axis, so it reduces to one number per
            # neuron. The previous expression was `act_tensor.abs().mean(dim=0)
            # .squeeze(0)[neuron_idx]`: mean(dim=0) already averages over batch
            # and leaves [seq, inter_dim], and squeeze(0) is a no-op because
            # dim 0 is seq rather than 1. Indexing that with neuron_idx
            # therefore returned a vector, and float() on a multi-element
            # tensor raised -- so this branch only ever worked while
            # `inter_dim` happened to be small enough to hide it.
            avg_activation = float(torch.tanh(act_tensor.abs().mean(dim=1)[0, neuron_idx] * 3.0)) \
                if act_tensor.size(0) == 1 else float(token_acts.mean())
            neuron_dtos.append(NeuronActivation(
                layer=li, index=neuron_idx, activation=avg_activation,
                token_activations=token_activations,
            ))

        # Cache & Repository
        if session:
            rec = ActivationRecord(
                session_id=session.session_id,
                prompt_id=session.session_id[:8],
                layer=li,
                head=None,
                component="mlp",
                shape=tuple(act_tensor.shape),
                tensor=act_tensor,
            )
            activation_repo.save(rec)

        if cache and session:
            ca = CachedActivation(
                session_id=session.session_id,
                prompt_id=session.session_id[:8],
                layer=li,
                head=None,
                component="mlp",
                shape=list(act_tensor.shape),
                tensor=act_tensor,
                metadata={"model": config._name_or_path if hasattr(config, '_name_or_path') else ""},
            )
            cid = cache.put(ca)
            cache_ids.append(cid)

    # ── Generation ───────────────────────────────────────────
    if profiler:
        with profiler.measure("generation", max_tokens=max_new_tokens):
            generated_ids = model.generate(
                input_ids, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
    else:
        generated_ids = model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    bus.emit(INFERENCE_FINISHED, tokens=len(all_tokens), generated_length=len(generated_ids[0]))

    # Remove registered hooks
    for handle in hook_manager.list_hooks():
        hook_manager.remove_hook(handle)

    return {
        "tokens": all_tokens,
        "generated_text": full_text,
        "attention_maps": attn_dtos,
        "neuron_activations": neuron_dtos,
        "cache_ids": cache_ids,
    }
