"""Real GPT-2 inference + dynamic architecture/neuron introspection.

All layer counts, neuron counts, weight stats, and activations come from the
loaded HuggingFace GPT-2 model — nothing is hardcoded per neuron/layer.
"""
from __future__ import annotations

import math
import threading
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ML_AVAILABLE = True
except Exception:
    ML_AVAILABLE = False
    np = None  # type: ignore
    torch = None  # type: ignore

_lock = threading.Lock()
_model: Any = None
_tokenizer: Any = None
_cache: Dict[str, Any] = {}


def is_available() -> bool:
    return ML_AVAILABLE


def _ensure_loaded() -> Optional[Dict[str, Any]]:
    if not ML_AVAILABLE:
        return {"status": "error", "error": "torch/transformers not installed"}
    if _model is None:
        r = load()
        if r.get("status") != "loaded":
            return r
    return None


def load() -> Dict[str, Any]:
    global _model, _tokenizer
    if not ML_AVAILABLE:
        return {"status": "error", "error": "torch/transformers not installed"}
    if _model is not None:
        return info()
    with _lock:
        if _model is not None:
            return info()
        try:
            try:
                _tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
                _model = AutoModelForCausalLM.from_pretrained(
                    "gpt2", local_files_only=True, attn_implementation="eager"
                )
            except Exception:
                # Fall back to download if local cache is missing
                _tokenizer = AutoTokenizer.from_pretrained("gpt2")
                _model = AutoModelForCausalLM.from_pretrained(
                    "gpt2", attn_implementation="eager"
                )
            _model.config.output_attentions = True
            _model.config.output_hidden_states = True
            _model.eval()
        except Exception as e:
            return {"status": "error", "error": str(e)}
        return info()


def _cfg() -> Any:
    return _model.config


def _d_mlp() -> int:
    """MLP width from the actual c_fc weight — not a hardcoded multiplier."""
    w = _model.transformer.h[0].mlp.c_fc.weight
    # Conv1D weight is [nx, nf] = [d_model, d_mlp]
    return int(w.shape[1])


def _d_model() -> int:
    return int(_cfg().n_embd)


def _n_layers() -> int:
    return int(_cfg().n_layer)


def _n_heads() -> int:
    return int(_cfg().n_head)


def _head_dim() -> int:
    return _d_model() // _n_heads()


def info() -> Dict[str, Any]:
    n_params = sum(p.numel() for p in _model.parameters())
    return {
        "status": "loaded",
        "model_name": "gpt2",
        "model_type": getattr(_cfg(), "model_type", "gpt2"),
        "n_layers": _n_layers(),
        "n_heads": _n_heads(),
        "d_model": _d_model(),
        "d_mlp": _d_mlp(),
        "d_head": _head_dim(),
        "vocab_size": int(_cfg().vocab_size),
        "n_positions": int(getattr(_cfg(), "n_positions", 1024)),
        "n_params": int(n_params),
        "n_params_human": _human_params(n_params),
        "device": str(next(_model.parameters()).device),
        "dtype": str(next(_model.parameters()).dtype),
        "activation": "gelu_new",
        "architecture": "decoder-only transformer (GPT-2)",
    }


def _human_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    return str(n)


def _decode(id_: int) -> str:
    return _tokenizer.decode([id_]).replace("Ġ", " ")


def _token_logit(token: str, logits) -> float:
    ids = _tokenizer.encode(" " + token.lstrip())
    return float(logits[ids[-1]].item())


def _tensor_stats(t: "torch.Tensor") -> Dict[str, float]:
    t = t.detach().float().reshape(-1)
    if t.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "l2": 0.0, "abs_mean": 0.0}
    return {
        "mean": round(float(t.mean()), 6),
        "std": round(float(t.std(unbiased=False)), 6),
        "min": round(float(t.min()), 6),
        "max": round(float(t.max()), 6),
        "l2": round(float(torch.linalg.vector_norm(t)), 6),
        "abs_mean": round(float(t.abs().mean()), 6),
    }


def _module_param_summary(module: Any) -> Dict[str, Any]:
    """Summarize every parameter of a module from live weights."""
    params = []
    total = 0
    for name, p in module.named_parameters(recurse=True):
        n = int(p.numel())
        total += n
        params.append(
            {
                "name": name,
                "shape": list(p.shape),
                "numel": n,
                "requires_grad": bool(p.requires_grad),
                "stats": _tensor_stats(p.data),
            }
        )
    return {"n_params": total, "parameters": params}


# ---------------------------------------------------------------------------
# Architecture tree (fully dynamic from loaded model)
# ---------------------------------------------------------------------------

def architecture() -> Dict[str, Any]:
    err = _ensure_loaded()
    if err:
        return err

    layers = []
    for li in range(_n_layers()):
        block = _model.transformer.h[li]
        d_mlp = int(block.mlp.c_fc.weight.shape[1])
        layers.append(
            {
                "layer_index": li,
                "label": f"blocks.{li}",
                "path": f"transformer.h.{li}",
                "num_attention_heads": _n_heads(),
                "num_mlp_neurons": d_mlp,
                "residual_stream_dim": _d_model(),
                "components": [
                    {
                        "id": "ln_1",
                        "type": "layernorm",
                        "path": f"transformer.h.{li}.ln_1",
                        "dim": _d_model(),
                    },
                    {
                        "id": "attn",
                        "type": "attention",
                        "path": f"transformer.h.{li}.attn",
                        "n_heads": _n_heads(),
                        "d_head": _head_dim(),
                        "d_model": _d_model(),
                    },
                    {
                        "id": "ln_2",
                        "type": "layernorm",
                        "path": f"transformer.h.{li}.ln_2",
                        "dim": _d_model(),
                    },
                    {
                        "id": "mlp",
                        "type": "mlp",
                        "path": f"transformer.h.{li}.mlp",
                        "d_model": _d_model(),
                        "d_mlp": d_mlp,
                        "num_neurons": d_mlp,
                        "activation": "gelu_new",
                    },
                ],
                "n_params": sum(p.numel() for p in block.parameters()),
            }
        )

    modules = [
        {
            "id": "wte",
            "label": "Token Embeddings",
            "path": "transformer.wte",
            "type": "embedding",
            "shape": list(_model.transformer.wte.weight.shape),
            "n_params": int(_model.transformer.wte.weight.numel()),
        },
        {
            "id": "wpe",
            "label": "Position Embeddings",
            "path": "transformer.wpe",
            "type": "embedding",
            "shape": list(_model.transformer.wpe.weight.shape),
            "n_params": int(_model.transformer.wpe.weight.numel()),
        },
        {
            "id": "blocks",
            "label": "Transformer Blocks",
            "type": "block_stack",
            "n_layers": _n_layers(),
            "layers": layers,
        },
        {
            "id": "ln_f",
            "label": "Final LayerNorm",
            "path": "transformer.ln_f",
            "type": "layernorm",
            "dim": _d_model(),
            "n_params": sum(p.numel() for p in _model.transformer.ln_f.parameters()),
        },
        {
            "id": "lm_head",
            "label": "LM Head",
            "path": "lm_head",
            "type": "linear",
            "shape": list(_model.lm_head.weight.shape),
            "n_params": int(_model.lm_head.weight.numel()),
            "tied_with_wte": _model.lm_head.weight.data_ptr()
            == _model.transformer.wte.weight.data_ptr(),
        },
    ]

    base = info()
    return {
        **base,
        "status": "ok",
        "modules": modules,
        "layers": layers,
        "has_cache": bool(_cache),
        "cached_prompt": _cache.get("prompt"),
        "cached_tokens": _cache.get("str_tokens"),
    }


# ---------------------------------------------------------------------------
# Forward pass with MLP + residual caches
# ---------------------------------------------------------------------------

def _forward(prompt: str) -> Dict[str, Any]:
    global _cache
    inputs = _tokenizer(prompt, return_tensors="pt")
    device = next(_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    mlp_pre: Dict[int, Any] = {}
    mlp_post: Dict[int, Any] = {}
    hooks = []

    for li, block in enumerate(_model.transformer.h):
        # c_fc output = pre-activation (before GELU)
        hooks.append(
            block.mlp.c_fc.register_forward_hook(
                lambda mod, inp, out, i=li: mlp_pre.__setitem__(i, out.detach())
            )
        )
        # c_proj input = post-GELU activations (the actual neuron firing)
        hooks.append(
            block.mlp.c_proj.register_forward_hook(
                lambda mod, inp, out, i=li: mlp_post.__setitem__(
                    i, inp[0].detach() if isinstance(inp, tuple) else inp.detach()
                )
            )
        )

    try:
        with torch.no_grad():
            out = _model(**inputs, output_attentions=True, output_hidden_states=True)
    finally:
        for h in hooks:
            h.remove()

    seq_ids = inputs["input_ids"][0].tolist()
    _cache = {
        "prompt": prompt,
        "ids": seq_ids,
        "str_tokens": [_decode(i) for i in seq_ids],
        "logits": out.logits[0, -1].detach().cpu(),
        "attentions": [a[0].detach().cpu().numpy() for a in out.attentions],
        "hidden": [h[0].detach().cpu().numpy() for h in out.hidden_states],
        "mlp_pre": {i: t[0].cpu() for i, t in mlp_pre.items()},   # [seq, d_mlp]
        "mlp_post": {i: t[0].cpu() for i, t in mlp_post.items()},  # [seq, d_mlp]
    }
    return _cache


def run_prompt(prompt: str) -> Dict[str, Any]:
    err = _ensure_loaded()
    if err:
        return err
    c = _forward(prompt)
    logits = c["logits"]
    probs = torch.softmax(logits, dim=-1)
    topk = torch.topk(probs, 16)
    top5 = [
        {"token": _decode(int(i)), "logit": round(float(logits[i]), 4), "prob": round(float(probs[i]), 6)}
        for i in topk.indices[:5]
    ]
    top16 = [
        {"token": _decode(int(i)), "logit": round(float(logits[i]), 4), "prob": round(float(probs[i]), 6)}
        for i in topk.indices[:16]
    ]
    return {
        "status": "ok",
        "prompt": prompt,
        "str_tokens": c["str_tokens"],
        "token_ids": c["ids"],
        "top5": top5,
        "top16": top16,
        "next_token": _decode(int(torch.argmax(logits))),
        "n_layers": _n_layers(),
        "d_mlp": _d_mlp(),
        "d_model": _d_model(),
    }


# ---------------------------------------------------------------------------
# Layer detail
# ---------------------------------------------------------------------------

def layer_detail(layer: int) -> Dict[str, Any]:
    err = _ensure_loaded()
    if err:
        return err
    layer = max(0, min(_n_layers() - 1, int(layer)))
    block = _model.transformer.h[layer]
    d_mlp = int(block.mlp.c_fc.weight.shape[1])

    # Weight stats from live tensors
    c_fc_w = block.mlp.c_fc.weight.data  # [d_model, d_mlp]
    c_proj_w = block.mlp.c_proj.weight.data  # [d_mlp, d_model]
    c_attn_w = block.attn.c_attn.weight.data
    c_attn_proj = block.attn.c_proj.weight.data

    # Per-neuron L2 norms of input weights (for heatmap / ranking)
    in_norms = torch.linalg.vector_norm(c_fc_w, dim=0).cpu()  # [d_mlp]
    out_norms = torch.linalg.vector_norm(c_proj_w, dim=1).cpu()  # [d_mlp]
    bias = block.mlp.c_fc.bias.data.cpu() if block.mlp.c_fc.bias is not None else None

    # Activation summary if cache present
    act_summary = None
    top_active = []
    if _cache and "mlp_post" in _cache and layer in _cache["mlp_post"]:
        acts = _cache["mlp_post"][layer]  # [seq, d_mlp]
        last = acts[-1]
        mean_abs = acts.abs().mean(dim=0)
        act_summary = {
            "seq_len": int(acts.shape[0]),
            "mean_abs_over_seq": _tensor_stats(mean_abs),
            "last_token_stats": _tensor_stats(last),
            "fraction_active_last": round(float((last.abs() > 1e-3).float().mean()), 4),
        }
        str_tokens = _cache.get("str_tokens") or []
        best_idx = acts.argmax(dim=0).tolist()  # [d_mlp] token index per neuron
        top_idx = torch.topk(last.abs(), k=min(32, d_mlp)).indices.tolist()
        top_active = [
            {
                "neuron_index": int(i),
                "activation": round(float(last[i]), 6),
                "abs_activation": round(float(last[i].abs()), 6),
                "in_weight_l2": round(float(in_norms[i]), 6),
                "out_weight_l2": round(float(out_norms[i]), 6),
                "top_token_index": int(best_idx[i]),
                "top_token": str_tokens[best_idx[i]] if best_idx[i] < len(str_tokens) else None,
                "top_token_activation": round(float(acts[best_idx[i], i]), 6),
            }
            for i in top_idx
        ]

    # Attention heads summary from live weights
    heads = []
    hd = _head_dim()
    # c_attn is [d_model, 3*d_model] = Q|K|V
    attn_layer = None
    str_tokens = []
    if _cache:
        if "attentions" in _cache and layer < len(_cache["attentions"]):
            attn_layer = _cache["attentions"][layer]
        str_tokens = _cache.get("str_tokens") or []
    for h in range(_n_heads()):
        q = c_attn_w[:, h * hd : (h + 1) * hd]
        k = c_attn_w[:, _d_model() + h * hd : _d_model() + (h + 1) * hd]
        v = c_attn_w[:, 2 * _d_model() + h * hd : 2 * _d_model() + (h + 1) * hd]
        o = c_attn_proj[h * hd : (h + 1) * hd, :]
        top_token = None
        top_token_index = None
        top_attention_weight = None
        if attn_layer is not None:
            attn = attn_layer[h]  # [seq, seq], rows=query, cols=key
            if attn is not None and attn.ndim == 2 and attn.shape[0] > 0:
                last_row = attn[-1]
                best = int(last_row.argmax())
                top_token_index = best
                top_token = str_tokens[best] if best < len(str_tokens) else None
                top_attention_weight = round(float(last_row[best]), 6)
        heads.append(
            {
                "head_index": h,
                "label": f"L{layer}H{h}",
                "d_head": hd,
                "q_weight_l2": round(float(torch.linalg.vector_norm(q)), 6),
                "k_weight_l2": round(float(torch.linalg.vector_norm(k)), 6),
                "v_weight_l2": round(float(torch.linalg.vector_norm(v)), 6),
                "o_weight_l2": round(float(torch.linalg.vector_norm(o)), 6),
                "top_token_index": top_token_index,
                "top_token": top_token,
                "top_attention_weight": top_attention_weight,
            }
        )

    return {
        "status": "ok",
        "layer": layer,
        "path": f"transformer.h.{layer}",
        "num_attention_heads": _n_heads(),
        "num_mlp_neurons": d_mlp,
        "residual_stream_dim": _d_model(),
        "n_params": sum(p.numel() for p in block.parameters()),
        "modules": {
            "ln_1": _module_param_summary(block.ln_1),
            "attn": _module_param_summary(block.attn),
            "ln_2": _module_param_summary(block.ln_2),
            "mlp": _module_param_summary(block.mlp),
        },
        "mlp": {
            "d_mlp": d_mlp,
            "c_fc_shape": list(c_fc_w.shape),
            "c_proj_shape": list(c_proj_w.shape),
            "in_weight_l2_stats": _tensor_stats(in_norms),
            "out_weight_l2_stats": _tensor_stats(out_norms),
            "bias_stats": _tensor_stats(bias) if bias is not None else None,
        },
        "attention_heads": heads,
        "activation_summary": act_summary,
        "top_active_neurons": top_active,
        "has_activations": act_summary is not None,
        "prompt": _cache.get("prompt") if _cache else None,
        "str_tokens": _cache.get("str_tokens") if _cache else None,
    }


# ---------------------------------------------------------------------------
# Neuron listing (paginated, sorted by real metrics)
# ---------------------------------------------------------------------------

def list_neurons(
    layer: int,
    component: str = "mlp",
    page: int = 0,
    page_size: int = 128,
    sort_by: str = "index",
    order: str = "asc",
    q: str = "",
) -> Dict[str, Any]:
    err = _ensure_loaded()
    if err:
        return err

    layer = max(0, min(_n_layers() - 1, int(layer)))
    page = max(0, int(page))
    page_size = max(1, min(512, int(page_size)))
    component = (component or "mlp").lower()

    block = _model.transformer.h[layer]

    if component == "mlp":
        c_fc_w = block.mlp.c_fc.weight.data  # [d_model, d_mlp]
        c_proj_w = block.mlp.c_proj.weight.data  # [d_mlp, d_model]
        bias = block.mlp.c_fc.bias.data if block.mlp.c_fc.bias is not None else None
        total = int(c_fc_w.shape[1])
        in_norms = torch.linalg.vector_norm(c_fc_w.float(), dim=0).cpu()
        out_norms = torch.linalg.vector_norm(c_proj_w.float(), dim=1).cpu()

        acts_last = None
        acts_mean = None
        best_idx = None
        str_tokens = []
        if _cache and "mlp_post" in _cache and layer in _cache["mlp_post"]:
            acts = _cache["mlp_post"][layer].float()
            acts_last = acts[-1]
            acts_mean = acts.mean(dim=0)
            best_idx = acts.argmax(dim=0).tolist()  # [total] token index per neuron
            str_tokens = _cache.get("str_tokens") or []

        records = []
        for i in range(total):
            rec = {
                "neuron_index": i,
                "label": f"L{layer}.mlp.N{i}",
                "component": "mlp",
                "in_weight_l2": float(in_norms[i]),
                "out_weight_l2": float(out_norms[i]),
                "bias": float(bias[i]) if bias is not None else 0.0,
                "activation": float(acts_last[i]) if acts_last is not None else None,
                "mean_activation": float(acts_mean[i]) if acts_mean is not None else None,
                "top_token": str_tokens[best_idx[i]] if best_idx is not None and best_idx[i] < len(str_tokens) else None,
                "top_token_activation": round(float(acts[best_idx[i], i]), 6) if best_idx is not None else None,
            }
            records.append(rec)
    elif component in ("resid", "residual"):
        total = _d_model()
        acts_last = None
        if _cache and "hidden" in _cache:
            # hidden[0]=embed, hidden[i+1]=after block i
            h = torch.tensor(_cache["hidden"][layer + 1][-1])
            acts_last = h
        records = []
        for i in range(total):
            records.append(
                {
                    "neuron_index": i,
                    "label": f"L{layer}.resid.D{i}",
                    "component": "resid",
                    "in_weight_l2": None,
                    "out_weight_l2": None,
                    "bias": None,
                    "activation": float(acts_last[i]) if acts_last is not None else None,
                    "mean_activation": None,
                }
            )
    else:
        return {"status": "error", "error": f"Unknown component: {component}"}

    # Optional text filter (index substring)
    if q:
        q = str(q).strip().lower()
        records = [r for r in records if q in str(r["neuron_index"]) or q in r["label"].lower()]

    # Sort
    key_map = {
        "index": lambda r: r["neuron_index"],
        "in_weight_l2": lambda r: r["in_weight_l2"] if r["in_weight_l2"] is not None else -1,
        "out_weight_l2": lambda r: r["out_weight_l2"] if r["out_weight_l2"] is not None else -1,
        "activation": lambda r: abs(r["activation"]) if r["activation"] is not None else -1,
        "bias": lambda r: abs(r["bias"]) if r["bias"] is not None else -1,
        "mean_activation": lambda r: abs(r["mean_activation"]) if r["mean_activation"] is not None else -1,
    }
    key_fn = key_map.get(sort_by, key_map["index"])
    reverse = order.lower() == "desc"
    records.sort(key=key_fn, reverse=reverse)

    total_filtered = len(records)
    start = page * page_size
    end = min(start + page_size, total_filtered)
    page_recs = records[start:end]

    # Round for JSON
    for r in page_recs:
        for k in ("in_weight_l2", "out_weight_l2", "bias", "activation", "mean_activation"):
            if r[k] is not None:
                r[k] = round(float(r[k]), 6)

    return {
        "status": "ok",
        "layer": layer,
        "component": component,
        "page": page,
        "page_size": page_size,
        "total_neurons": total,
        "total_filtered": total_filtered,
        "sort_by": sort_by,
        "order": order,
        "has_activations": acts_last is not None if component == "mlp" else acts_last is not None,
        "neurons": page_recs,
        "prompt": _cache.get("prompt") if _cache else None,
    }


# ---------------------------------------------------------------------------
# Single neuron deep inspection (real weights + activations)
# ---------------------------------------------------------------------------

def neuron_detail(
    layer: int,
    neuron_index: int,
    component: str = "mlp",
    top_k_weights: int = 16,
) -> Dict[str, Any]:
    err = _ensure_loaded()
    if err:
        return err

    layer = max(0, min(_n_layers() - 1, int(layer)))
    component = (component or "mlp").lower()
    top_k_weights = max(1, min(64, int(top_k_weights)))
    block = _model.transformer.h[layer]

    if component == "mlp":
        c_fc_w = block.mlp.c_fc.weight.data.float()  # [d_model, d_mlp]
        c_proj_w = block.mlp.c_proj.weight.data.float()  # [d_mlp, d_model]
        d_mlp = int(c_fc_w.shape[1])
        neuron_index = max(0, min(d_mlp - 1, int(neuron_index)))

        in_w = c_fc_w[:, neuron_index]  # [d_model]
        out_w = c_proj_w[neuron_index, :]  # [d_model]
        bias = (
            float(block.mlp.c_fc.bias.data[neuron_index])
            if block.mlp.c_fc.bias is not None
            else 0.0
        )

        # Top positive / negative input connections (from residual stream dims)
        top_in_pos = torch.topk(in_w, k=min(top_k_weights, in_w.numel()))
        top_in_neg = torch.topk(-in_w, k=min(top_k_weights, in_w.numel()))
        top_out_pos = torch.topk(out_w, k=min(top_k_weights, out_w.numel()))
        top_out_neg = torch.topk(-out_w, k=min(top_k_weights, out_w.numel()))

        def _pack(idx, vals, negate=False):
            return [
                {
                    "dim": int(i),
                    "weight": round(float(-v if negate else v), 6),
                }
                for i, v in zip(idx.tolist(), vals.tolist())
            ]

        # Activations across tokens if available
        per_token = []
        hist = None
        act_stats = None
        if _cache and "mlp_post" in _cache and layer in _cache["mlp_post"]:
            acts = _cache["mlp_post"][layer][:, neuron_index].float()  # [seq]
            pre = None
            if layer in _cache.get("mlp_pre", {}):
                pre = _cache["mlp_pre"][layer][:, neuron_index].float()
            tokens = _cache.get("str_tokens") or [f"t{i}" for i in range(len(acts))]
            for ti, tok in enumerate(tokens):
                per_token.append(
                    {
                        "token_index": ti,
                        "token": tok,
                        "activation": round(float(acts[ti]), 6),
                        "pre_activation": round(float(pre[ti]), 6) if pre is not None else None,
                    }
                )
            act_stats = _tensor_stats(acts)
            # Simple histogram over sequence (and denser if we only have seq)
            vals = acts.numpy()
            counts, edges = np.histogram(vals, bins=12)
            hist = {
                "bins": [round(float(e), 4) for e in edges.tolist()],
                "counts": [int(c) for c in counts.tolist()],
            }

        # Cosine similarity to other neurons in same layer (sample for speed)
        nearest = _nearest_mlp_neurons(layer, neuron_index, k=8)

        return {
            "status": "ok",
            "model_name": "gpt2",
            "layer": layer,
            "neuron_index": neuron_index,
            "component": "mlp",
            "id": f"L{layer}.mlp.N{neuron_index}",
            "path": f"transformer.h.{layer}.mlp",
            "d_mlp": d_mlp,
            "d_model": _d_model(),
            "bias": round(bias, 6),
            "in_weight_stats": _tensor_stats(in_w),
            "out_weight_stats": _tensor_stats(out_w),
            "in_weight_l2": round(float(torch.linalg.vector_norm(in_w)), 6),
            "out_weight_l2": round(float(torch.linalg.vector_norm(out_w)), 6),
            "top_input_weights_positive": _pack(top_in_pos.indices, top_in_pos.values),
            "top_input_weights_negative": _pack(top_in_neg.indices, top_in_neg.values, negate=True),
            "top_output_weights_positive": _pack(top_out_pos.indices, top_out_pos.values),
            "top_output_weights_negative": _pack(top_out_neg.indices, top_out_neg.values, negate=True),
            "per_token_activations": per_token,
            "activation_stats": act_stats,
            "activation_histogram": hist,
            "nearest_neurons": nearest,
            "has_activations": bool(per_token),
            "prompt": _cache.get("prompt") if _cache else None,
            "str_tokens": _cache.get("str_tokens") if _cache else None,
            "description": (
                f"MLP neuron {neuron_index} in block {layer}. "
                f"Reads from residual stream via c_fc[:, {neuron_index}] "
                f"(shape [{_d_model()}]), applies GELU, writes back via "
                f"c_proj[{neuron_index}, :] (shape [{_d_model()}])."
            ),
        }

    # Residual stream dimension
    neuron_index = max(0, min(_d_model() - 1, int(neuron_index)))
    per_token = []
    act_stats = None
    hist = None
    if _cache and "hidden" in _cache:
        h = _cache["hidden"][layer + 1]  # [seq, d_model]
        col = h[:, neuron_index]
        tokens = _cache.get("str_tokens") or [f"t{i}" for i in range(len(col))]
        for ti, tok in enumerate(tokens):
            per_token.append(
                {
                    "token_index": ti,
                    "token": tok,
                    "activation": round(float(col[ti]), 6),
                    "pre_activation": None,
                }
            )
        t = torch.tensor(col)
        act_stats = _tensor_stats(t)
        counts, edges = np.histogram(col, bins=12)
        hist = {
            "bins": [round(float(e), 4) for e in edges.tolist()],
            "counts": [int(c) for c in counts.tolist()],
        }

    return {
        "status": "ok",
        "model_name": "gpt2",
        "layer": layer,
        "neuron_index": neuron_index,
        "component": "resid",
        "id": f"L{layer}.resid.D{neuron_index}",
        "path": f"transformer.h.{layer} residual dim {neuron_index}",
        "d_model": _d_model(),
        "bias": None,
        "in_weight_stats": None,
        "out_weight_stats": None,
        "in_weight_l2": None,
        "out_weight_l2": None,
        "top_input_weights_positive": [],
        "top_input_weights_negative": [],
        "top_output_weights_positive": [],
        "top_output_weights_negative": [],
        "per_token_activations": per_token,
        "activation_stats": act_stats,
        "activation_histogram": hist,
        "nearest_neurons": [],
        "has_activations": bool(per_token),
        "prompt": _cache.get("prompt") if _cache else None,
        "str_tokens": _cache.get("str_tokens") if _cache else None,
        "description": (
            f"Residual stream dimension {neuron_index} after block {layer} "
            f"(d_model={_d_model()})."
        ),
    }


def _nearest_mlp_neurons(layer: int, neuron_index: int, k: int = 8) -> List[Dict[str, Any]]:
    """Find nearest MLP neurons by cosine similarity of concatenated in/out weights."""
    block = _model.transformer.h[layer]
    c_fc_w = block.mlp.c_fc.weight.data.float()  # [d_model, d_mlp]
    c_proj_w = block.mlp.c_proj.weight.data.float()  # [d_mlp, d_model]
    d_mlp = c_fc_w.shape[1]

    # Vector for each neuron: [in_w || out_w]
    # For speed, sample up to 512 candidates evenly + self
    step = max(1, d_mlp // 512)
    idxs = list(range(0, d_mlp, step))
    if neuron_index not in idxs:
        idxs.append(neuron_index)

    target_in = c_fc_w[:, neuron_index]
    target_out = c_proj_w[neuron_index, :]
    target = torch.cat([target_in, target_out])
    t_norm = torch.linalg.vector_norm(target)
    if t_norm < 1e-12:
        return []

    sims = []
    for j in idxs:
        if j == neuron_index:
            continue
        v = torch.cat([c_fc_w[:, j], c_proj_w[j, :]])
        vn = torch.linalg.vector_norm(v)
        if vn < 1e-12:
            continue
        sim = float(torch.dot(target, v) / (t_norm * vn))
        sims.append((j, sim))
    sims.sort(key=lambda x: -x[1])
    return [
        {"layer": layer, "neuron_index": j, "similarity": round(sim, 4), "component": "mlp"}
        for j, sim in sims[:k]
    ]


# ---------------------------------------------------------------------------
# Attention head detail
# ---------------------------------------------------------------------------

def head_detail(layer: int, head: int) -> Dict[str, Any]:
    err = _ensure_loaded()
    if err:
        return err
    layer = max(0, min(_n_layers() - 1, int(layer)))
    head = max(0, min(_n_heads() - 1, int(head)))
    hd = _head_dim()
    block = _model.transformer.h[layer]
    c_attn_w = block.attn.c_attn.weight.data.float()
    c_proj_w = block.attn.c_proj.weight.data.float()
    d = _d_model()

    q = c_attn_w[:, head * hd : (head + 1) * hd]
    k = c_attn_w[:, d + head * hd : d + (head + 1) * hd]
    v = c_attn_w[:, 2 * d + head * hd : 2 * d + (head + 1) * hd]
    o = c_proj_w[head * hd : (head + 1) * hd, :]

    matrix = None
    tokens = None
    if _cache and "attentions" in _cache:
        matrix = [
            [round(float(v), 4) for v in row]
            for row in _cache["attentions"][layer][head]
        ]
        tokens = _cache.get("str_tokens")

    return {
        "status": "ok",
        "layer": layer,
        "head": head,
        "id": f"L{layer}H{head}",
        "path": f"transformer.h.{layer}.attn head {head}",
        "d_head": hd,
        "d_model": d,
        "q_stats": _tensor_stats(q),
        "k_stats": _tensor_stats(k),
        "v_stats": _tensor_stats(v),
        "o_stats": _tensor_stats(o),
        "attention_matrix": matrix,
        "str_tokens": tokens,
        "has_pattern": matrix is not None,
        "prompt": _cache.get("prompt") if _cache else None,
        "description": (
            f"Attention head {head} in block {layer}. "
            f"Q/K/V slices of c_attn, output via c_proj rows "
            f"[{head * hd}:{(head + 1) * hd}]."
        ),
    }


def attention_head(layer: int, head: int) -> Dict[str, Any]:
    err = _ensure_loaded()
    if err:
        return err
    if not _cache:
        return {"status": "error", "error": "Run a prompt first to populate the cache."}
    detail = head_detail(layer, head)
    if detail.get("status") != "ok":
        return detail
    return {
        "status": "ok",
        "layer": detail["layer"],
        "head": detail["head"],
        "matrix": detail["attention_matrix"] or [],
        "str_tokens": detail["str_tokens"] or [],
    }


def activations(layer: int) -> Dict[str, Any]:
    err = _ensure_loaded()
    if err:
        return err
    layer = max(0, min(_n_layers() - 1, int(layer)))
    seq = len(_cache.get("str_tokens", [])) if _cache else 0
    result: Dict[str, Any] = {
        "status": "ok",
        "layer": layer,
        "resid_shape": [seq, _d_model()],
        "attn_shape": [_n_heads(), seq, seq],
        "mlp_shape": [seq, _d_mlp()],
        "has_cache": bool(_cache),
    }
    if _cache and "mlp_post" in _cache and layer in _cache["mlp_post"]:
        acts = _cache["mlp_post"][layer]
        result["mlp_last_token"] = [round(float(x), 5) for x in acts[-1].tolist()]
        result["mlp_stats"] = _tensor_stats(acts)
    if _cache and "hidden" in _cache:
        h = torch.tensor(_cache["hidden"][layer + 1])
        result["resid_stats"] = _tensor_stats(h)
        result["resid_last_token"] = [round(float(x), 5) for x in h[-1].tolist()[:64]]
    return result


def patch_head(layer: int, head: int, pos_token: str, neg_token: str) -> Dict[str, Any]:
    err = _ensure_loaded()
    if err:
        return err
    if not _cache:
        return {"status": "error", "error": "Run a prompt first to populate the cache."}
    layer = max(0, min(_n_layers() - 1, int(layer)))
    head = max(0, min(_n_heads() - 1, int(head)))
    prompt = _cache["prompt"]
    head_dim = _head_dim()

    def logit_diff() -> float:
        inputs = _tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = _model(**inputs)
        logits = out.logits[0, -1]
        return _token_logit(pos_token, logits) - _token_logit(neg_token, logits)

    def zero_head_hook(module, inp, out):
        x = inp[0]
        sl = slice(head * head_dim, (head + 1) * head_dim)
        if x.shape[-1] >= sl.stop:
            x[..., sl] = 0.0
        return out

    clean_ld = round(logit_diff(), 4)
    hook = _model.transformer.h[layer].attn.c_proj.register_forward_hook(zero_head_hook)
    try:
        patched_ld = round(logit_diff(), 4)
    finally:
        hook.remove()
    delta = round(patched_ld - clean_ld, 4)
    return {
        "status": "ok",
        "layer": layer,
        "head": head,
        "clean_ld": clean_ld,
        "patched_ld": patched_ld,
        "delta": delta,
        "direction": "hurts" if delta < 0 else "helps",
    }


def patch_neuron(
    layer: int,
    neuron_index: int,
    patch_value: float,
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Clamp an MLP neuron's post-GELU activation and measure top-token change."""
    err = _ensure_loaded()
    if err:
        return err
    layer = max(0, min(_n_layers() - 1, int(layer)))
    d_mlp = _d_mlp()
    neuron_index = max(0, min(d_mlp - 1, int(neuron_index)))
    prompt = prompt or (_cache.get("prompt") if _cache else "The capital of France is")

    def run(with_patch: bool):
        hooks = []
        if with_patch:

            def pre_hook(module, inp):
                x = inp[0]
                if x.dim() >= 2 and x.shape[-1] > neuron_index:
                    x = x.clone()
                    x[..., neuron_index] = float(patch_value)
                    return (x,) + tuple(inp[1:])
                return inp

            hooks.append(
                _model.transformer.h[layer].mlp.c_proj.register_forward_pre_hook(pre_hook)
            )
        try:
            inputs = _tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                out = _model(**inputs)
            logits = out.logits[0, -1].detach()
            top_id = int(torch.argmax(logits))
            return {
                "top_token": _decode(top_id),
                "top_logit": round(float(logits[top_id]), 4),
                "logits": logits,
            }
        finally:
            for h in hooks:
                h.remove()

    before = run(False)
    after = run(True)
    before_logits = before["logits"]
    after_logits = after["logits"]
    top_id = int(torch.argmax(before_logits))
    delta = round(float(after_logits[top_id] - before_logits[top_id]), 4)

    return {
        "status": "ok",
        "layer": layer,
        "neuron_index": neuron_index,
        "patch_value": float(patch_value),
        "prompt": prompt,
        "top_token_before": before["top_token"],
        "top_token_after": after["top_token"],
        "original_logit": before["top_logit"],
        "patched_logit": after["top_logit"],
        "delta": delta,
    }


def ioi(io_name: str, subj_name: str) -> Dict[str, Any]:
    err = _ensure_loaded()
    if err:
        return err
    clean_prompt = f"When {subj_name} and {io_name} went to the store, {subj_name} gave a bottle to"
    corrupted_prompt = f"When {subj_name} and {io_name} went to the store, {io_name} gave a bottle to"
    io_id = _tokenizer.encode(" " + io_name)[-1]
    subj_id = _tokenizer.encode(" " + subj_name)[-1]

    def run(prompt: str):
        inputs = _tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = _model(**inputs)
        logits = out.logits[0, -1]
        top1 = _decode(int(torch.argmax(torch.softmax(logits, dim=-1))))
        ld = float(logits[io_id].item()) - float(logits[subj_id].item())
        return top1, round(ld, 4)

    clean_top1, clean_ld = run(clean_prompt)
    corr_top1, corr_ld = run(corrupted_prompt)
    return {
        "status": "ok",
        "io_name": io_name,
        "subj_name": subj_name,
        "clean_prompt": clean_prompt,
        "clean_top1": clean_top1,
        "clean_ld": clean_ld,
        "ioi_pass": clean_top1.strip() == io_name,
        "corrupted_prompt": corrupted_prompt,
        "corrupted_top1": corr_top1,
        "corrupted_ld": corr_ld,
        "corrupted_pass": corr_top1.strip() == subj_name,
    }


def infer(prompt: str, model_name: str) -> Dict[str, Any]:
    err = _ensure_loaded()
    if err:
        return err
    c = _forward(prompt)
    logits = c["logits"]
    next_id = int(torch.argmax(torch.softmax(logits, dim=-1)))
    next_text = _decode(next_id)
    tokens = [{"text": t, "id": i} for t, i in zip(c["str_tokens"], c["ids"])]
    tokens.append({"text": next_text, "id": next_id})
    n = len(c["str_tokens"])
    attention_maps = [
        {
            "layer": li,
            "head": hi,
            "tokens": c["str_tokens"],
            "matrix": [
                [round(float(v), 4) for v in row]
                for row in c["attentions"][li][hi][:n, :n]
            ],
        }
        for li in range(_n_layers())
        for hi in range(_n_heads())
    ]
    # Real top-active MLP neurons per layer (not hardcoded 8 samples)
    neuron_activations = []
    for li in range(_n_layers()):
        if li in c.get("mlp_post", {}):
            last = c["mlp_post"][li][-1]
            top = torch.topk(last.abs(), k=min(16, last.numel()))
            for idx, val in zip(top.indices.tolist(), last[top.indices].tolist()):
                neuron_activations.append(
                    {
                        "layer": li,
                        "index": int(idx),
                        "activation": round(float(val), 4),
                    }
                )
    return {
        "model_name": model_name,
        "tokens": tokens,
        "generated_text": "".join(t["text"] for t in tokens),
        "attention_maps": attention_maps,
        "neuron_activations": neuron_activations,
        "n_layers": _n_layers(),
        "d_mlp": _d_mlp(),
        "d_model": _d_model(),
    }
