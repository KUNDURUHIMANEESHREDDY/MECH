"""
GPT-2 Service — wraps the 8-step pipeline from gpt2_steps.py into
JSON-serialisable methods callable by the dispatcher.

All methods return plain dicts/lists so they can pass through the
JSON-lines IPC protocol unchanged.
"""

from __future__ import annotations

import os
import sys
import warnings
import traceback
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore")

# ── lazy singletons ──────────────────────────────────────────────────────────
_model   = None   # HookedTransformer
_cache   = None   # last ActivationCache
_tokens  = None   # last token tensor
_prompt  = None   # last prompt string
_device  = "cpu"

def _get_model():
    global _model, _device
    if _model is not None:
        return _model

    # ── path-shadow fix ──────────────────────────────────────────────────────
    # backend/datasets/ is a project sub-package that shadows the real
    # `datasets` PyPI package needed by TransformerLens.  Remove any sys.path
    # entry whose resolved name ends with "backend" before importing so that
    # Python finds the real package instead of our local folder.
    import os as _os
    _shadow_paths = [
        p for p in sys.path
        if _os.path.basename(_os.path.normpath(p)).lower() == "backend"
    ]
    for _p in _shadow_paths:
        sys.path.remove(_p)
    try:
        import torch
        from transformer_lens import HookedTransformer
    finally:
        # Restore the paths so the rest of the sidecar still works.
        for _p in _shadow_paths:
            sys.path.insert(0, _p)
    # ────────────────────────────────────────────────────────────────────────

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = HookedTransformer.from_pretrained("gpt2", device=_device)
    return _model


# ── public methods ────────────────────────────────────────────────────────────

def load_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load GPT-2 Small (Steps 2 + 3).
    Returns architecture facts.
    """
    try:
        m = _get_model()
        return {
            "status": "loaded",
            "model_name": "GPT-2 Small",
            "n_layers": m.cfg.n_layers,
            "n_heads":  m.cfg.n_heads,
            "d_model":  m.cfg.d_model,
            "d_mlp":    m.cfg.d_mlp,
            "device":   _device,
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "trace": traceback.format_exc(limit=4)}


def run_prompt(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single prompt and return top-5 predictions + Paris logit info (Step 4).
    payload: { prompt: str }
    """
    global _cache, _tokens, _prompt
    try:
        import torch
        m = _get_model()
        prompt = str(payload.get("prompt", "The capital of France is"))
        _prompt = prompt

        toks = m.to_tokens(prompt)
        _tokens = toks

        with torch.no_grad():
            logits, cache = m.run_with_cache(toks)
        _cache = cache

        last = logits[0, -1, :]
        top5 = last.topk(5)
        top5_out = [
            {"token": m.to_string([int(i)]).strip(), "logit": float(v)}
            for v, i in zip(top5.values, top5.indices)
        ]

        # Paris vs capitals
        capitals = [" Paris", " London", " Berlin", " Madrid", " Rome"]
        cap_logits = {}
        for cap in capitals:
            try:
                tid = m.to_single_token(cap)
                cap_logits[cap.strip()] = float(last[tid])
            except Exception:
                pass

        paris_id   = m.to_single_token(" Paris")
        paris_rank = int((last > last[paris_id]).sum()) + 1

        str_tokens = m.to_str_tokens(prompt)

        return {
            "status": "ok",
            "prompt": prompt,
            "str_tokens": str_tokens,
            "top5": top5_out,
            "capitals": cap_logits,
            "paris_rank": paris_rank,
            "paris_logit": float(last[paris_id]),
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "trace": traceback.format_exc(limit=4)}


def get_activations(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return shapes of cached residual, attention, and MLP activations (Step 5).
    Optionally return the actual tensors as nested lists (expensive for big models).
    payload: { layer: int, include_values: bool }
    """
    try:
        if _cache is None:
            return {"status": "error", "error": "No cache — run a prompt first."}
        layer  = int(payload.get("layer", 0))
        m      = _get_model()
        n_layers = m.cfg.n_layers
        if layer >= n_layers:
            layer = 0

        resid = _cache["resid_post", layer]
        attn  = _cache["pattern",   layer]
        mlp   = _cache["mlp_post",  layer]

        out = {
            "status": "ok",
            "layer": layer,
            "resid_shape": list(resid.shape),
            "attn_shape":  list(attn.shape),
            "mlp_shape":   list(mlp.shape),
        }
        if payload.get("include_values"):
            out["resid_values"] = resid[0].tolist()
            out["mlp_values"]   = mlp[0].tolist()
        return out
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_attention_head(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return attention weight matrix for a specific layer/head (Step 6).
    payload: { layer: int, head: int }
    """
    try:
        if _cache is None:
            return {"status": "error", "error": "No cache — run a prompt first."}
        m     = _get_model()
        layer = int(payload.get("layer", 10))
        head  = int(payload.get("head",  7))
        layer = min(layer, m.cfg.n_layers - 1)
        head  = min(head,  m.cfg.n_heads  - 1)

        mat = _cache["pattern", layer][0, head].tolist()
        str_tokens = m.to_str_tokens(_prompt) if _prompt else []

        return {
            "status": "ok",
            "layer": layer,
            "head":  head,
            "matrix": mat,
            "str_tokens": str_tokens,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def patch_head(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Zero-ablate one attention head and return logit difference change (Step 7).
    payload: { layer: int, head: int, pos_token: str, neg_token: str }
    """
    try:
        if _tokens is None:
            return {"status": "error", "error": "No tokens — run a prompt first."}
        import torch
        m     = _get_model()
        layer = int(payload.get("layer", 10))
        head  = int(payload.get("head",  7))
        pos   = str(payload.get("pos_token", " Paris"))
        neg   = str(payload.get("neg_token", " London"))

        layer = min(layer, m.cfg.n_layers - 1)
        head  = min(head,  m.cfg.n_heads  - 1)

        pos_id = m.to_single_token(pos)
        neg_id = m.to_single_token(neg)

        def ld(lgts):
            return float(lgts[0, -1, pos_id] - lgts[0, -1, neg_id])

        with torch.no_grad():
            clean_logits = m(_tokens)
        clean_ld = ld(clean_logits)

        def zero_hook(z, hook):
            z[:, :, head, :] = 0.0
            return z

        with torch.no_grad():
            patched_logits = m.run_with_hooks(
                _tokens,
                fwd_hooks=[(f"blocks.{layer}.attn.hook_z", zero_hook)],
            )
        patched_ld = ld(patched_logits)
        delta = patched_ld - clean_ld

        return {
            "status":     "ok",
            "layer":      layer,
            "head":       head,
            "pos_token":  pos,
            "neg_token":  neg,
            "clean_ld":   clean_ld,
            "patched_ld": patched_ld,
            "delta":      delta,
            "direction":  "hurts" if delta < 0 else "helps",
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "trace": traceback.format_exc(limit=4)}


def run_ioi(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run one IOI experiment — clean vs corrupted (Step 8).
    payload: { io_name: str, subj_name: str }
    """
    try:
        import torch
        m   = _get_model()
        io  = str(payload.get("io_name",   "Mary"))
        subj = str(payload.get("subj_name", "John"))

        clean = f"When {io} and {subj} went to the store, {subj} gave the bag to"
        corr  = f"When {subj} and {io} went to the store, {io} gave the bag to"

        io_id   = m.to_single_token(f" {io}")
        subj_id = m.to_single_token(f" {subj}")

        def run(prompt):
            toks = m.to_tokens(prompt)
            with torch.no_grad():
                lgts = m(toks)
            last = lgts[0, -1, :]
            ld   = float(last[io_id] - last[subj_id])
            top1 = m.to_string([int(last.argmax())]).strip()
            return ld, top1

        clean_ld, clean_top  = run(clean)
        corr_ld,  corr_top   = run(corr)

        return {
            "status":          "ok",
            "io_name":         io,
            "subj_name":       subj,
            "clean_prompt":    clean,
            "clean_top1":      clean_top,
            "clean_ld":        clean_ld,
            "corrupted_prompt": corr,
            "corrupted_top1":  corr_top,
            "corrupted_ld":    corr_ld,
            "ioi_pass":        clean_ld > 0,
            "corrupted_pass":  corr_ld  < 0,
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "trace": traceback.format_exc(limit=4)}
