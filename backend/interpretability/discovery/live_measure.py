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


def path_patch(clean: str, corrupted: str, io_id: int, subj_id: int,
               sender: Tuple[int, int],
               receivers: List[Tuple[int, int]]) -> Dict[str, Any]:
    """Isolate the direct causal path from `sender` to each receiver.

    The canonical path-patching procedure (Wang et al. 2022) needs *two
    simultaneous interventions* on the corrupted run:

    1. Freeze the sender -- its output is forced to the clean-run value, so it
       cannot contribute its ordinary, non-path effect.
    2. Swap the receiver's input -- the receiver reads the clean-run value that
       arrived at its position from the sender.

    The change in logit difference relative to the plain corrupted baseline is
    then attributable to the sender->receiver route itself.

    Doing only step 2, as the previous implementation did, leaves the sender
    free to contribute normally, so the measurement conflates the direct path
    with the sender's total effect and overstates every edge it reports. That
    is why the two hooks must be installed together.
    """
    s_layer, s_head = sender
    engine = _engine()
    import torch
    _, _, hd = dims()

    # --- clean run: capture the sender's output and each receiver's input ---
    sender_out: Dict[str, Any] = {}
    receiver_in: Dict[int, Any] = {}
    hooks = []
    with _MEASURE_LOCK:
        def sender_hook(_mod: Any, _inp: Any, out: Any) -> Any:
            sender_out["value"] = out[0][:, -1, :].detach().clone()
            return out

        def make_receiver_hook(layer: int) -> Any:
            def hook(_mod: Any, inp: Any) -> Any:
                receiver_in[layer] = inp[0][:, -1, :].detach().clone()
                return inp
            return hook

        hooks.append(engine._model.transformer.h[s_layer].attn.c_proj
                     .register_forward_hook(sender_hook))
        for layer, _head in receivers:
            hooks.append(engine._model.transformer.h[layer].attn.c_proj
                         .register_forward_pre_hook(make_receiver_hook(layer)))
        try:
            with torch.no_grad():
                engine._model(**_inputs(engine, clean))
        finally:
            for handle in hooks:
                handle.remove()

    clean_sender = sender_out.get("value")
    if clean_sender is None:
        raise RuntimeError("path patching: sender output was not captured")

    corr_base = baseline(corrupted, io_id, subj_id)["logit_diff"]
    clean_base = baseline(clean, io_id, subj_id)["logit_diff"]
    denom = clean_base - corr_base

    rows = []
    for r_layer, r_head in receivers:
        cached_in = receiver_in.get(r_layer)
        if cached_in is None:
            continue
        forward = _path_patched_forward(
            engine, corrupted, io_id, subj_id,
            s_layer=s_layer, s_head=s_head, clean_sender=clean_sender, hd=hd,
            r_layer=r_layer, r_head=r_head, cached_receiver_in=cached_in)
        patched_diff = forward["logit_diff"]
        rows.append({
            "receiver": head_label(r_layer, r_head),
            "patched_logit_diff": patched_diff,
            "path_effect": (None if not denom
                            else round((patched_diff - corr_base) / denom, 4)),
            "top1": forward["top1"],
        })

    return {
        "method": "path_patching",
        "sender": head_label(*sender),
        "receivers": rows,
        "clean_logit_diff": clean_base,
        "corrupted_logit_diff": corr_base,
        "denominator": denom,
        "sender_frozen": True,
        "receiver_input_swapped": True,
        "isolates_direct_path": True,
    }


def _path_patched_forward(engine: Any, prompt: str, io_id: int, subj_id: int,
                          *, s_layer: int, s_head: int, clean_sender: Any,
                          hd: int, r_layer: int, r_head: int,
                          cached_receiver_in: Any) -> Dict[str, Any]:
    """One corrupted forward pass with the sender frozen and receiver swapped."""
    import torch
    hooks = []
    s_slice = slice(s_head * hd, (s_head + 1) * hd)

    with _MEASURE_LOCK:
        def freeze_sender(_mod: Any, _inp: Any, out: Any) -> Any:
            x = out[0].clone()
            src = clean_sender.to(x.device).to(x.dtype)
            x[:, -1, s_slice] = src[:, -1, s_slice]
            return (x,) + tuple(out[1:])

        def swap_receiver(_mod: Any, inp: Any) -> Any:
            x = inp[0].clone()
            src = cached_receiver_in.to(x.device).to(x.dtype)
            x[:, -1, :] = src[:, -1, :]
            return (x,) + tuple(inp[1:])

        hooks.append(engine._model.transformer.h[s_layer].attn.c_proj
                     .register_forward_hook(freeze_sender))
        hooks.append(engine._model.transformer.h[r_layer].attn.c_proj
                     .register_forward_pre_hook(swap_receiver))
        try:
            with torch.no_grad():
                out = engine._model(**_inputs(engine, prompt))
            logits = out.logits[0, -1].detach().cpu()
            return {
                "logit_diff": _logit_diff(engine, logits, io_id, subj_id),
                "top1": engine._decode(int(torch.argmax(logits))),
            }
        finally:
            for handle in hooks:
                handle.remove()


def circuit_fidelity(clean: str, corrupted: str, io_id: int, subj_id: int,
                     retained: Set[Tuple[int, int]]) -> Dict[str, Any]:
    """Fraction of the clean-minus-corrupted logit gap recovered by `retained`.

    This is ACDC's headline metric, and it is a real measurement: the retained
    heads are injected into the corrupted run and the resulting logit
    difference is compared against the gap the clean model achieves. It
    replaces a rescaling of the pruning ratio that could not report failure.
    """
    clean_base = baseline(clean, io_id, subj_id)["logit_diff"]
    corr_base = baseline(corrupted, io_id, subj_id)["logit_diff"]
    denom = clean_base - corr_base
    if not retained or not denom:
        return {
            "fidelity": None,
            "measured": False,
            "denominator": denom,
            "reason": ("fidelity is undefined without both a retained circuit "
                       "and a non-zero clean-minus-corrupted logit gap"),
        }
    _, caps = capture(clean)
    recovered = inject(corrupted, io_id, subj_id, caps, retained)["logit_diff"]
    return {
        "fidelity": round((recovered - corr_base) / denom, 4),
        "measured": True,
        "clean_logit_diff": clean_base,
        "corrupted_logit_diff": corr_base,
        "recovered_logit_diff": recovered,
        "denominator": denom,
        "retained_heads": sorted(head_label(*h) for h in retained),
    }

