"""Shared live causal-measurement primitives for GPT-2 Small.

All functions in this module measure the loaded HuggingFace GPT-2 model with
real forward passes — zero-ablation and clean-to-corrupted activation
injection through forward hooks.  Nothing here is seeded, hardcoded, or
copied from a reference circuit list.

Threading: hook-based interventions are serialized with a module RLock so
concurrent Society/API workers cannot interleave hooks on the shared model.
Results are deterministic for fixed weights/prompts, so a small in-memory
cache keyed by (operation, prompt, parameters) avoids re-measuring the same
intervention twice in one process lifetime.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Set, Tuple


class LiveUnavailable(Exception):
    """Raised when no live model is connected; callers must fail closed."""


_MEASURE_LOCK = threading.RLock()
_CACHE: Dict[str, Any] = {}


def _engine():
    from backend.services import gpt2_engine as engine
    if not engine.is_available():
        raise LiveUnavailable("torch/transformers not installed")
    loaded = engine.load()
    if not isinstance(loaded, dict) or loaded.get("status") != "loaded":
        raise LiveUnavailable(f"model load failed: {loaded}")
    return engine


def dims() -> Tuple[int, int, int]:
    """Return (n_layers, n_heads, head_dim) from the live model config."""
    engine = _engine()
    n_layers = int(engine._model.config.n_layer)
    n_heads = int(engine._model.config.n_head)
    return n_layers, n_heads, int(engine._model.config.n_embd // n_heads)


def single_token_names(names: List[str]) -> Dict[str, Optional[int]]:
    """Map each name to its single leading-space token id, or None.

    A name that does not tokenize to exactly one token cannot be used as an
    IOI comparison target; callers must fail closed for such prompts rather
    than guessing token splits.
    """
    engine = _engine()
    out: Dict[str, Optional[int]] = {}
    for name in names:
        ids = engine._tokenizer.encode(" " + str(name).lstrip())
        out[name] = int(ids[-1]) if len(ids) == 1 else None
    return out


def clean_prompt(subject: str, io_name: str) -> str:
    return (f"When {subject} and {io_name} went to the store, "
            f"{subject} gave a bottle to")


def corrupted_prompt(subject: str, io_name: str) -> str:
    return (f"When {subject} and {io_name} went to the store, "
            f"{io_name} gave a bottle to")


def _inputs(engine: Any, prompt: str) -> Dict[str, Any]:
    device = next(engine._model.parameters()).device
    inputs = engine._tokenizer(prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def _logit_diff(engine: Any, logits: Any, io_id: int, subj_id: int) -> float:
    return float(logits[io_id].item()) - float(logits[subj_id].item())


def baseline(prompt: str, io_id: int, subj_id: int) -> Dict[str, Any]:
    """Clean forward pass: top-1 token and IO-minus-subject logit difference."""
    key = f"baseline|{prompt}|{io_id}|{subj_id}"
    if key in _CACHE:
        return dict(_CACHE[key])
    engine = _engine()
    import torch
    with _MEASURE_LOCK:
        with torch.no_grad():
            out = engine._model(**_inputs(engine, prompt))
        logits = out.logits[0, -1].detach().cpu()
        top1 = engine._decode(int(torch.argmax(logits)))
        result = {
            "top1": top1,
            "logit_diff": _logit_diff(engine, logits, io_id, subj_id),
        }
    _CACHE[key] = dict(result)
    return result


def ablate(prompt: str, io_id: int, subj_id: int,
           ablate_heads: Set[Tuple[int, int]]) -> float:
    """Logit difference with the given (layer, head) outputs zero-ablated.

    An empty set measures the clean baseline through the same code path.
    """
    key = (f"ablate|{prompt}|{io_id}|{subj_id}|"
           f"{','.join(f'{l}H{h}' for l, h in sorted(ablate_heads))}")
    if key in _CACHE:
        return float(_CACHE[key])
    engine = _engine()
    import torch
    _, n_heads, hd = dims()
    hooks = []
    with _MEASURE_LOCK:
        for layer, head in sorted(ablate_heads):
            def hook(mod: Any, inp: Any, _h: int = head) -> Any:
                x = inp[0].clone()
                x[..., _h * hd:(_h + 1) * hd] = 0.0
                return (x,) + tuple(inp[1:])
            hooks.append(engine._model.transformer.h[layer]
                         .attn.c_proj.register_forward_pre_hook(hook))
        try:
            with torch.no_grad():
                out = engine._model(**_inputs(engine, prompt))
            value = _logit_diff(engine, out.logits[0, -1].detach().cpu(),
                                io_id, subj_id)
        finally:
            for handle in hooks:
                handle.remove()
    _CACHE[key] = value
    return value


def capture(prompt: str) -> Tuple[Any, Dict[int, Any]]:
    """Forward pass capturing per-layer merged attention outputs.

    Returns (last-token logits, {layer: c_proj input tensor}).
    """
    key = f"capture|{prompt}"
    if key in _CACHE:
        logits, caps = _CACHE[key]
        return logits.clone(), {layer: t.clone() for layer, t in caps.items()}
    engine = _engine()
    import torch
    caps: Dict[int, Any] = {}
    hooks = []
    with _MEASURE_LOCK:
        n_layers = int(engine._model.config.n_layer)
        for layer in range(n_layers):
            def fhook(mod: Any, inp: Any, out: Any, _l: int = layer) -> None:
                caps[_l] = inp[0].detach().cpu().clone()
            hooks.append(engine._model.transformer.h[layer]
                         .attn.c_proj.register_forward_hook(fhook))
        try:
            with torch.no_grad():
                out = engine._model(**_inputs(engine, prompt))
            logits = out.logits[0, -1].detach().cpu().clone()
        finally:
            for handle in hooks:
                handle.remove()
    _CACHE[key] = (logits.clone(),
                   {layer: t.clone() for layer, t in caps.items()})
    return logits, caps


def inject(prompt: str, io_id: int, subj_id: int,
           clean_caps: Dict[int, Any],
           active_heads: Set[Tuple[int, int]]) -> Dict[str, Any]:
    """Run `prompt` with clean attention slices injected for active heads.

    Used for clean-to-corrupted circuit isolation: `prompt` is normally the
    corrupted variant while `clean_caps` come from the clean variant.
    """
    key = (f"inject|{prompt}|{io_id}|{subj_id}|"
           f"{','.join(f'{l}H{h}' for l, h in sorted(active_heads))}")
    if key in _CACHE:
        return dict(_CACHE[key])
    engine = _engine()
    import torch
    _, _, hd = dims()
    hooks = []
    with _MEASURE_LOCK:
        by_layer: Dict[int, List[int]] = {}
        for layer, head in sorted(active_heads):
            by_layer.setdefault(layer, []).append(head)
        for layer, heads in by_layer.items():
            clean = clean_caps[layer]
            def hook(mod: Any, inp: Any, _x: Any = clean,
                     _hs: Tuple[int, ...] = tuple(heads)) -> Any:
                x = inp[0].clone()
                src = _x.to(x.device)
                for hh in _hs:
                    x[..., hh * hd:(hh + 1) * hd] = src[..., hh * hd:(hh + 1) * hd]
                return (x,) + tuple(inp[1:])
            hooks.append(engine._model.transformer.h[layer]
                         .attn.c_proj.register_forward_pre_hook(hook))
        try:
            with torch.no_grad():
                out = engine._model(**_inputs(engine, prompt))
            logits = out.logits[0, -1].detach().cpu()
            result = {
                "top1": engine._decode(int(torch.argmax(logits))),
                "logit_diff": _logit_diff(engine, logits, io_id, subj_id),
            }
        finally:
            for handle in hooks:
                handle.remove()
    _CACHE[key] = dict(result)
    return result


def head_label(layer: int, head: int) -> str:
    return f"L{layer}H{head}"


def parse_head(label: str) -> Optional[Tuple[int, int]]:
    try:
        body = str(label).strip().upper().lstrip("L")
        layer_s, head_s = body.split("H")
        return int(layer_s), int(head_s)
    except Exception:
        return None
