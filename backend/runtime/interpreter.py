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

    for idx, attn_tensor in enumerate(attention_store):
        # attn_tensor shape: [batch, heads, seq, seq]
        li = idx // num_heads
        hi = idx % num_heads
        mat = attn_tensor[0].numpy().tolist()  # [heads, seq, seq] → [seq, seq]

        attn_dtos.append(AttentionMap(
            layer=li, head=hi, tokens=[t.text for t in all_tokens], matrix=mat,
        ))

        # Cache & Repository
        if session:
            rec = ActivationRecord(
                session_id=session.session_id,
                prompt_id=session.session_id[:8],
                layer=li,
                head=hi,
                component="attention",
                shape=tuple(attn_tensor.shape),
                tensor=attn_tensor,
            )
            activation_repo.save(rec)

        if cache and session:
            ca = CachedActivation(
                session_id=session.session_id,
                prompt_id=session.session_id[:8],
                layer=li,
                head=hi,
                component="attention",
                shape=list(attn_tensor.shape),
                tensor=attn_tensor,
                metadata={"model": config._name_or_path if hasattr(config, '_name_or_path') else ""},
            )
            cid = cache.put(ca)
            cache_ids.append(cid)
            bus.emit(ACTIVATION_CACHED, activation_id=cid, layer=li, head=hi)

    # ── Build neuron activations ─────────────────────────────
    n_neurons_per_layer = 8
    neuron_dtos: list[NeuronActivation] = []

    for li in range(num_layers):
        # Each MLP hook stores one tensor per layer
        layer_mlps = [t for idx, t in enumerate(mlp_store) if idx // 1 == li]
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
                layer=li, index=ni, activation=avg_activation,
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
