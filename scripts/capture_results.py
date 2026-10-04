"""Capture real, measured interpretability results from live GPT-2 weights.

Every number written by this script comes from a forward pass over real GPT-2
small weights via ``transformers``. Nothing is simulated, sampled from a
distribution, or copied from a literature baseline. The output is what the
README quotes as results.

Outputs
-------
docs/results/capture.json    structured measurements
docs/images/*.png            figures rendered from those measurements

Run:
    python scripts/capture_results.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

from backend.services import gpt2_engine as engine  # noqa: E402

OUT_DIR = os.path.join(ROOT, "docs", "results")
IMG_DIR = os.path.join(ROOT, "docs", "images")

# Standard IOI (Indirect Object Identification) template from
# Wang et al. 2022, "Interpretability in the Wild: a Circuit for
# Indirect Object Identification in GPT-2 Small" (arXiv:2211.00593).
IOI_TEMPLATES = [
    ("Then John and Mary went to the store. John gave a bottle of milk to", " Mary", " John"),
    ("Then John and Mary went to the park. John gave a bottle of milk to", " Mary", " John"),
    ("Then John and Mary went to the office. John gave a key to", " Mary", " John"),
]

CAPITAL_PROMPT = "The capital of France is"
STEER_POS = "The capital of France is"
STEER_NEG = "The capital of Japan is"


def timed(label: str, fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    print(f"  [{time.perf_counter() - start:6.2f}s] {label}")
    return result


# --------------------------------------------------------------------------
# Model level
# --------------------------------------------------------------------------
def capture_architecture() -> Dict[str, Any]:
    load = engine.load()
    arch = engine.architecture()
    return {
        "load_status": load.get("status"),
        "model_name": arch.get("model_name"),
        "n_layers": arch.get("n_layers"),
        "n_heads": arch.get("n_heads"),
        "d_model": arch.get("d_model"),
        "d_mlp": arch.get("d_mlp"),
        "d_head": arch.get("d_head"),
        "vocab_size": arch.get("vocab_size"),
        "n_positions": arch.get("n_positions"),
        "n_params": arch.get("n_params"),
        "n_params_human": arch.get("n_params_human"),
        "device": arch.get("device"),
        "dtype": arch.get("dtype"),
    }


# --------------------------------------------------------------------------
# Sanity: prove the engine reproduces real GPT-2 behaviour
# --------------------------------------------------------------------------
def capture_sanity() -> Dict[str, Any]:
    """GPT-2 small behaviours that are unambiguous, used to prove the engine is
    not producing noise. These are the checks a reviewer should reproduce."""
    def top(prompt: str, k: int = 5) -> List[Dict[str, Any]]:
        run = engine.run_prompt(prompt)
        return run.get("top5", [])[:k]

    checks = {}
    induction = top("Hello, my name is Julien. Hello, my name is")
    checks["induction"] = {
        "prompt": "Hello, my name is Julien. Hello, my name is",
        "top": induction,
        "expectation": "repeats the name stem ' Jul'",
        "passes": bool(induction and induction[0]["token"].strip() == "Jul"),
    }
    counting = top("1, 2, 3, 4, 5, 6,")
    checks["counting"] = {
        "prompt": "1, 2, 3, 4, 5, 6,",
        "top": counting,
        "expectation": "continues with ' 7'",
        "passes": bool(counting and counting[0]["token"].strip() == "7"),
    }
    return checks


# --------------------------------------------------------------------------
# IOI: clean vs corrupted
# --------------------------------------------------------------------------
def capture_ioi() -> Dict[str, Any]:
    rows = []
    for prompt, correct, incorrect in IOI_TEMPLATES:
        engine.run_prompt(prompt)  # populate cache
        res = engine.ioi(correct.strip(), incorrect.strip())
        rows.append(
            {
                "prompt": prompt,
                "correct": correct,
                "incorrect": incorrect,
                "clean_top1": res.get("clean_top1"),
                "corrupted_top1": res.get("corrupted_top1"),
                "ioi_pass": res.get("ioi_pass"),
                "corrupted_pass": res.get("corrupted_pass"),
            }
        )
    n_pass = sum(1 for r in rows if r.get("ioi_pass"))
    n_flip = sum(1 for r in rows if r.get("corrupted_pass"))
    return {
        "templates": rows,
        "n_templates": len(rows),
        "clean_correct": n_pass,
        "corrupted_flipped": n_flip,
        "clean_accuracy": round(n_pass / len(rows), 4) if rows else None,
        "corrupted_flip_rate": round(n_flip / len(rows), 4) if rows else None,
    }


# --------------------------------------------------------------------------
# Logit lens
# --------------------------------------------------------------------------
def capture_logit_lens() -> Dict[str, Any]:
    lens = engine.logit_lens_all(CAPITAL_PROMPT, top_k=5)
    rows = []
    for entry in lens.get("layers", []):
        tops = entry.get("top_k_tokens") or []
        rows.append(
            {
                "layer": entry.get("layer"),
                "top_token": entry.get("top_token"),
                "top": [
                    {"token": t.get("token"), "prob": t.get("prob")}
                    if isinstance(t, dict)
                    else t
                    for t in (tops if isinstance(tops, list) else [])
                ],
            }
        )
    paris_at = next(
        (r["layer"] for r in rows if r["top_token"] == " Paris"),
        None,
    )
    paris_ranks = {
        r["layer"]: next(
            (
                i
                for i, t in enumerate(r["top"], start=1)
                if isinstance(t, dict) and t.get("token") == " Paris"
            ),
            None,
        )
        for r in rows
    }
    return {
        "prompt": CAPITAL_PROMPT,
        "layers": rows,
        "paris_first_argmax_layer": paris_at,
        "paris_rank_by_layer": paris_ranks,
    }


# --------------------------------------------------------------------------
# Full 144-head zero-ablation sweep -> IOI circuit
# --------------------------------------------------------------------------
def capture_head_sweep() -> Dict[str, Any]:
    """Zero-ablate every one of the 12x12 heads and rank by how much the IOI
    logit difference collapses. This is the standard first pass of IOI circuit
    discovery and it is fully measured, not simulated."""
    prompt, correct, incorrect = IOI_TEMPLATES[0]
    engine.run_prompt(prompt)
    pos, neg = correct.strip(), incorrect.strip()

    heads: List[Dict[str, Any]] = []
    for layer in range(engine._n_layers()):
        for head in range(engine._n_heads()):
            res = engine.patch_head(layer, head, pos, neg)
            if res.get("status") != "ok":
                continue
            heads.append(
                {
                    "layer": layer,
                    "head": head,
                    "label": f"L{layer}H{head}",
                    "clean_ld": res.get("clean_ld"),
                    "patched_ld": res.get("patched_ld"),
                    "delta": res.get("delta"),
                    "drop_ratio": (
                        round(abs(res["delta"]) / abs(res["clean_ld"]), 4)
                        if res.get("clean_ld")
                        else None
                    ),
                }
            )

    ranked = sorted(heads, key=lambda h: abs(h["delta"] or 0), reverse=True)
    return {
        "prompt": prompt,
        "n_heads_swept": len(heads),
        "baseline_clean_ld": next((h["clean_ld"] for h in heads), None),
        "top_heads": ranked[:12],
        "all_heads": ranked,
    }


# --------------------------------------------------------------------------
# Layer ablation
# --------------------------------------------------------------------------
def capture_layer_ablation() -> Dict[str, Any]:
    prompt, correct, incorrect = IOI_TEMPLATES[0]
    rows = []
    for layer in range(engine._n_layers()):
        res = engine.ablate_layer(layer, prompt, correct.strip(), incorrect.strip())
        rows.append(
            {
                "layer": layer,
                "clean_ld": res.get("clean_ld"),
                "patched_ld": res.get("patched_ld"),
                "delta": res.get("delta"),
                "direction": res.get("direction"),
            }
        )
    return {"prompt": prompt, "layers": rows}


# --------------------------------------------------------------------------
# Activation steering
# --------------------------------------------------------------------------
def capture_steering() -> Dict[str, Any]:
    """Contrast-vector steering: the steering vector is the measured difference
    between the residual stream on a positive and a negative prompt. Sweep the
    coefficient to find where the prediction actually flips."""
    rows = []
    for layer in (6, 8, 10):
        for alpha in (5.0, 10.0, 20.0, 40.0, 80.0):
            res = engine.steer(CAPITAL_PROMPT, layer, STEER_POS, STEER_NEG, alpha)
            top5 = res.get("steered_top5") or []
            rows.append(
                {
                    "layer": layer,
                    "alpha": alpha,
                    "vector_norm": res.get("vector_norm"),
                    "clean_top": res.get("clean_top"),
                    "steered_top": res.get("steered_top"),
                    "flipped": res.get("flipped"),
                    "paris_rank": next(
                        (i for i, t in enumerate(top5, start=1) if t == " Paris"), None
                    ),
                    "steered_top5": top5,
                }
            )
    return {
        "positive_prompt": STEER_POS,
        "negative_prompt": STEER_NEG,
        "sweep": rows,
        "flips": [r for r in rows if r.get("flipped")],
    }


# --------------------------------------------------------------------------
# Inspection
# --------------------------------------------------------------------------
def capture_inspection() -> Dict[str, Any]:
    neurons = engine.list_neurons(8, "mlp", 0, 10, "in_norm", "desc")
    head = engine.head_detail(9, 9)
    return {
        "top_neurons_L8_by_in_norm": neurons.get("neurons", [])[:10],
        "head_L9H9": head,
    }


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def render_figures(data: Dict[str, Any]) -> List[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover
        print(f"  [skip figures] matplotlib unavailable: {exc}")
        return []

    os.makedirs(IMG_DIR, exist_ok=True)
    written = []

    # 1. IOI head sweep
    heads = data["head_sweep"]["all_heads"]
    if heads:
        grid = np.full((12, 12), np.nan)
        for h in heads:
            grid[h["layer"], h["head"]] = abs(h["delta"])
        fig, ax = plt.subplots(figsize=(7.2, 5.6))
        im = ax.imshow(grid, cmap="magma", aspect="auto")
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title("IOI zero-ablation: |drop in logit difference| per head")
        ax.set_xticks(range(12))
        ax.set_yticks(range(12))
        fig.colorbar(im, ax=ax, label="|Δ logit diff|")
        fig.tight_layout()
        path = os.path.join(IMG_DIR, "ioi-head-sweep.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)

    # 2. Layer ablation
    rows = data["layer_ablation"]["layers"]
    if rows:
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        ax.bar(
            [r["layer"] for r in rows],
            [r["delta"] for r in rows],
            color="#2563eb",
        )
        ax.axhline(0, color="#334155", lw=0.8)
        ax.set_xlabel("Ablated layer")
        ax.set_ylabel("Δ IOI logit difference")
        ax.set_title("Leave-one-layer-out ablation on the IOI prompt")
        ax.set_xticks(range(len(rows)))
        fig.tight_layout()
        path = os.path.join(IMG_DIR, "ioi-layer-ablation.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)

    # 3. Logit lens trajectory
    lens = data["logit_lens"]["layers"]
    if lens:
        layers = [row["layer"] for row in lens]
        probs = [
            next(
                (
                    t.get("prob")
                    for t in row["top"]
                    if isinstance(t, dict) and t.get("token") == " Paris"
                ),
                0.0,
            )
            or 0.0
            for row in lens
        ]
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        ax.plot(layers, probs, marker="o", color="#2563eb", lw=2)
        ax.fill_between(layers, probs, color="#2563eb", alpha=0.12)
        ax.set_xlabel("Layer")
        ax.set_ylabel('P(" Paris") at that layer')
        ax.set_title('Logit lens: probability of " Paris" per layer')
        ax.set_xticks(layers)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = os.path.join(IMG_DIR, "logit-lens-paris.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)

    # 4. Steering sweep: rank of " Paris" as steering strength increases
    rows = data["steering"]["sweep"]
    if rows:
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        for layer in sorted({r["layer"] for r in rows}):
            sel = sorted(
                (r for r in rows if r["layer"] == layer), key=lambda r: r["alpha"]
            )
            ax.plot(
                [r["alpha"] for r in sel],
                [r["paris_rank"] if r["paris_rank"] else 6 for r in sel],
                marker="o",
                label=f"layer {layer}",
            )
        ax.axhline(1, color="#13795f", ls="--", lw=1.2)
        ax.text(
            5.5,
            1.15,
            "rank 1 = steering wins",
            color="#13795f",
            fontsize=9,
        )
        ax.set_yticks([1, 2, 3, 4, 5, 6])
        ax.set_yticklabels(["1", "2", "3", "4", "5", "not in top 5"])
        ax.invert_yaxis()
        ax.set_xlabel("Steering coefficient α")
        ax.set_ylabel('Rank of " Paris" in steered top-5')
        ax.set_title("Contrast-vector steering: France prompt toward Paris")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = os.path.join(IMG_DIR, "steering-sweep.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)

    return written


#: The sections `main()` populates. A key missing from the captured data is a
#: section that did not run, which is what makes the provenance derivation below
#: a report rather than a claim.
EXPECTED_SECTIONS = (
    "architecture", "sanity_checks", "ioi", "logit_lens",
    "head_sweep", "layer_ablation", "steering", "inspection",
)


def derive_meta_provenance(data: Dict[str, Any]) -> Dict[str, Any]:
    """Report which sections actually produced data, and withhold `live` if not.

    This used to be a literal `"provenance": "live"` in the `meta` dict at the
    top of `main()`, stamped before a single forward pass had run and never
    revised -- `provenance` appeared exactly once in this file, in that literal.
    Every capture below it is a genuine measurement, but the flag asserted the
    outcome instead of deriving it, so any future change that let a capture fail
    quietly would keep writing `live` into the artifact the README's Results
    section is generated from.

    Presence, not truthiness: a section that legitimately found nothing -- an
    empty head sweep is a real result, not a missing measurement -- still ran and
    still measured. Only a key absent from `data` did not run.
    """
    measured = [k for k in EXPECTED_SECTIONS if k in data]
    missing = [k for k in EXPECTED_SECTIONS if k not in data]

    out: Dict[str, Any] = {
        "measured_sections": f"{len(measured)}/{len(EXPECTED_SECTIONS)}",
        "provenance": "live" if not missing else "unavailable",
    }
    if missing:
        out["reason"] = (
            "No data for: " + ", ".join(missing)
            + ". The artifact is incomplete, so it does not describe a full "
              "measurement run and must not be cited as one."
        )
        out["note"] = (
            "INCOMPLETE RUN. Some sections did not produce data, so this file "
            "does not claim that every value was measured."
        )
    return out


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading GPT-2 ...")
    data: Dict[str, Any] = {
        "meta": {
            "model": "gpt2 (124M)",
            "library": "transformers + torch",
            # provenance is set after the captures run, not asserted here -- see
            # the derivation below. It used to sit in this literal, stamped
            # "live" before a single forward pass had happened and never revised:
            # `provenance` appeared exactly once in this file. Every capture below
            # is a real measurement, but the flag claimed the outcome instead of
            # reporting it, so any future change that tolerated a failed capture
            # would keep writing "live" into the artifact this repository's
            # Results section is generated from.
            "generated_by": "scripts/capture_results.py",
            "note": (
                "Every value is measured from live GPT-2 weights by a real "
                "forward pass. No synthetic, sampled or literature-copied values."
            ),
        }
    }

    data["architecture"] = timed("architecture", capture_architecture)
    data["sanity_checks"] = timed("sanity checks", capture_sanity)
    data["ioi"] = timed("ioi (3 templates)", capture_ioi)
    data["logit_lens"] = timed("logit lens", capture_logit_lens)
    data["head_sweep"] = timed("144-head zero-ablation sweep", capture_head_sweep)
    data["layer_ablation"] = timed("layer ablation", capture_layer_ablation)
    data["steering"] = timed("steering sweep", capture_steering)
    data["inspection"] = timed("inspection", capture_inspection)

    figures = render_figures(data)

    data["figures"] = [os.path.relpath(p, ROOT).replace("\\", "/") for p in figures]
    data["meta"]["torch"] = torch.__version__

    data["meta"].update(derive_meta_provenance(data))
    if data["meta"]["provenance"] != "live":
        print(f"  [incomplete] {data['meta']['reason']}")

    path = os.path.join(OUT_DIR, "capture.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)

    print(f"\nWrote {os.path.relpath(path, ROOT)}")
    for fig in data["figures"]:
        print(f"  figure: {fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
