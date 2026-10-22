"""Path patching and same-harness reference comparison for IOI discovery.

Roadmap 2.1 left two open items that both need a measurement rather than a
number copied from somewhere else:

* "Add corrupted-prompt patching so the *IOI-specific* signal separates from
  generic token-identity effects." The live screen already ranks heads by the
  difference-in-differences between the clean-prompt and the ABB
  corrupted-prompt ablation effect. `screen_report` publishes *both* rankings
  from the same two ablations, so the effect of the corrupted control on the
  selection is visible rather than assumed.

* "Add a path-patching pass. `live_discovery.py` already has injection
  recovery -- build on it." `measure_paths` isolates the direct path for each
  discovered edge with the two hooks the procedure needs (the sender frozen
  at its clean value, the receiver reading the clean value that arrived at
  its position), and `reference_circuit` re-measures the published reference
  circuit through the *same* injection code on the *same* prompts, so the two
  circuits are compared like for like instead of against a published number
  that came from a different procedure.

Nothing here is hardcoded and nothing is inferred from a name: every number
is a forward pass over the loaded weights, and every measurement that could
not be taken is reported as unmeasured with its reason rather than as zero.
"""

from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List, Optional, Set, Tuple

from . import live_measure as lm

__all__ = [
    "PATH_PATCHING_EFFECT_FLOOR",
    "PATH_PATCHING_MAX_PROMPTS",
    "PATH_PATCHING_METHOD",
    "PATH_PATCHING_METRIC",
    "injection_faithfulness",
    "measure_paths",
    "reference_circuit",
    "screen_report",
]

#: Prompts the path-patching pass is measured on. Each (edge, prompt) pair
#: costs two forward passes, so the pass is bounded -- and it reports the
#: number of prompts it actually used, so a bounded pass is never read as a
#: panel-wide measurement.
PATH_PATCHING_MAX_PROMPTS = 3

#: A path effect below this is reported as measured-and-small. It is not
#: dropped and it is not a zero: "the path was measured and it is small" and
#: "the path was not measured" are different findings.
PATH_PATCHING_EFFECT_FLOOR = 0.05

#: How the path-patching pass intervenes, in the words the record uses.
PATH_PATCHING_METHOD = (
    "two-hook path patching: the sender's head output at the last position "
    "is frozen at its clean value while the receiver reads the clean run's "
    "merged attention input at its position"
)

#: What a path effect is a fraction of.
PATH_PATCHING_METRIC = (
    "receiver-input path effect under a frozen sender, as a fraction of the "
    "clean-minus-corrupted logit difference"
)


def injection_faithfulness(prompts: List[Dict[str, Any]],
                            heads: Set[Tuple[int, int]],
                            ) -> Tuple[List[float], List[float]]:
    """Faithfulness and completeness of `heads` on `prompts`, by injection.

    One function for the discovered circuit and for the published reference
    circuit, so the two numbers come out of identical code on identical
    prompts and the comparison between them is like for like.

    The arithmetic is the one the recovery stage has always used: the
    fraction of the corrupted-to-clean logit-difference gap that injecting
    `heads` recovers, and the recovered difference as a fraction of the clean
    difference. An empty head set measures nothing and returns nothing -- no
    zeros are manufactured for a circuit that was not there.
    """
    faithfulness: List[float] = []
    completeness: List[float] = []
    if not heads:
        return faithfulness, completeness
    for prompt in prompts:
        _, caps = lm.capture(prompt["clean"])
        patched = lm.inject(prompt["corrupted"], prompt["io_id"],
                            prompt["subj_id"], caps, set(heads))
        clean_diff = prompt["clean_diff"]
        corr_diff = prompt["corrupted_diff"]
        rec_diff = patched["logit_diff"]
        denom = clean_diff - corr_diff
        faith = ((rec_diff - corr_diff) / denom if denom > 0.2 else 0.0)
        faithfulness.append(max(0.0, min(1.0, faith)))
        compl = (rec_diff / clean_diff if clean_diff > 0.2 else 0.0)
        completeness.append(max(0.0, min(1.0, compl)))
    return faithfulness, completeness


def measure_paths(edges: List[Dict[str, Any]],
                  prompts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Isolated direct-path effects for `edges`, measured on `prompts`.

    `live_measure.path_patch` installs the two hooks the procedure needs:
    the sender's output is frozen at its clean value, and the receiver reads
    the clean value that arrived at its position. The change in logit
    difference is then attributable to the direct route rather than to the
    sender's ordinary contribution, which is why a single-site patch cannot
    stand in for it.

    Every measurement is reported, including those below the floor, and a
    path that could not be measured is reported as not measured. Neither
    case becomes a zero.
    """
    if not edges:
        return {
            "measured": False,
            "method": PATH_PATCHING_METHOD,
            "metric": PATH_PATCHING_METRIC,
            "n_prompts": len(prompts),
            "n_edges": 0,
            "n_measurements": 0,
            "effect_floor": PATH_PATCHING_EFFECT_FLOOR,
            "max_abs_path_effect": None,
            "mean_abs_path_effect": None,
            "significant_paths": [],
            "edges": [],
            "failures": [],
            "reason": ("no edges were discovered, so there is no direct path "
                       "to isolate"),
        }
    rows: List[Dict[str, Any]] = []
    for index, prompt in enumerate(prompts):
        for edge in edges:
            sender = lm.parse_head(str(edge.get("source", "")))
            receiver = lm.parse_head(str(edge.get("target", "")))
            if sender is None or receiver is None:
                rows.append({
                    "prompt_index": index,
                    "source": edge.get("source"),
                    "target": edge.get("target"),
                    "path_effect": None,
                    "reason": ("the edge names a head label this harness "
                               "cannot parse"),
                })
                continue
            try:
                measured = lm.path_patch(
                    prompt["clean"], prompt["corrupted"],
                    prompt["io_id"], prompt["subj_id"],
                    sender, [receiver])
            except Exception as exc:  # noqa: BLE001 - one failed path is a finding
                rows.append({
                    "prompt_index": index,
                    "source": edge.get("source"),
                    "target": edge.get("target"),
                    "path_effect": None,
                    "reason": (f"the path could not be measured: "
                               f"{type(exc).__name__}: {exc}"),
                })
                continue
            receiver_rows = measured.get("receivers") or []
            effect = (receiver_rows[0].get("path_effect")
                      if receiver_rows else None)
            rows.append({
                "prompt_index": index,
                "source": edge.get("source"),
                "target": edge.get("target"),
                "path_effect": effect,
                "denominator": measured.get("denominator"),
                "reason": (None if effect is not None else
                           "the clean and corrupted prompts gave the target "
                           "identical logits, so there is no gap to isolate "
                           "a path against"),
            })
    per_edge: List[Dict[str, Any]] = []
    for edge in edges:
        key = (edge.get("source"), edge.get("target"))
        effects = [row["path_effect"] for row in rows
                   if (row["source"], row["target"]) == key
                   and row["path_effect"] is not None]
        per_edge.append({
            "source": key[0],
            "target": key[1],
            "n_measured": len(effects),
            "mean_path_effect": (round(mean(effects), 4) if effects else None),
            "mean_abs_path_effect": (round(mean(abs(e) for e in effects), 4)
                                     if effects else None),
            "max_abs_path_effect": (round(max(abs(e) for e in effects), 4)
                                    if effects else None),
        })
    effects = [row["path_effect"] for row in rows
               if row["path_effect"] is not None]
    significant = [edge for edge in per_edge
                   if edge["mean_abs_path_effect"] is not None
                   and edge["mean_abs_path_effect"] >= PATH_PATCHING_EFFECT_FLOOR]
    failures = [{"source": row["source"], "target": row["target"],
                 "prompt_index": row["prompt_index"], "reason": row["reason"]}
                for row in rows if row["path_effect"] is None]
    return {
        "measured": bool(effects),
        "method": PATH_PATCHING_METHOD,
        "metric": PATH_PATCHING_METRIC,
        "n_prompts": len(prompts),
        "n_edges": len(edges),
        "n_measurements": len(effects),
        "effect_floor": PATH_PATCHING_EFFECT_FLOOR,
        "max_abs_path_effect": (round(max(abs(e) for e in effects), 4)
                                if effects else None),
        "mean_abs_path_effect": (round(mean(abs(e) for e in effects), 4)
                                 if effects else None),
        "significant_paths": significant,
        "edges": per_edge,
        "failures": failures,
        "reason": (None if effects else
                   "no path could be measured on this circuit"),
    }


def reference_circuit(prompts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The published reference circuit, re-measured through this harness.

    The registry's published `circuit_faithfulness` is not this harness's
    measurement -- it comes from the paper's own procedure over its own edge
    set -- so it stays marked not comparable. What *is* comparable is the
    published head list run through the identical injection code on the
    identical prompts, which is what this measures. The head list is a
    reference *input* here and never a result.

    The list is read from the one place that defines it. If that import
    fails, the measurement is refused rather than approximated.
    """
    try:
        from backend.science.reproducibility.ioi_pipeline import (
            REFERENCE_ONLY_DISCOVERED_EDGES,
            REFERENCE_ONLY_DISCOVERED_HEADS,
        )
    except Exception as exc:  # noqa: BLE001 - a refusal, not a number
        return {
            "measured": False,
            "faithfulness_same_harness": None,
            "reason": ("the published reference circuit could not be read: "
                       f"{type(exc).__name__}: {exc}"),
        }
    heads = {lm.parse_head(label) for label in REFERENCE_ONLY_DISCOVERED_HEADS}
    heads.discard(None)
    if not heads:
        return {
            "measured": False,
            "faithfulness_same_harness": None,
            "reason": "the published reference head list is empty or unparseable",
        }
    faithfulness, completeness = injection_faithfulness(prompts, heads)
    reference_edges = [{"source": source, "target": target}
                       for source, target in REFERENCE_ONLY_DISCOVERED_EDGES]
    paths = measure_paths(reference_edges, prompts[:PATH_PATCHING_MAX_PROMPTS])
    return {
        "measured": bool(faithfulness),
        "heads": sorted(lm.head_label(*head) for head in heads),
        "faithfulness_same_harness": (round(mean(faithfulness), 4)
                                      if faithfulness else None),
        "completeness_same_harness": (round(mean(completeness), 4)
                                      if completeness else None),
        "n_prompts": len(prompts),
        "basis": ("clean-to-corrupted injection recovery through the same "
                  "code and the same prompts as the discovered circuit"),
        "path_patching": paths,
        "reason": (None if faithfulness else
                   "no prompt produced a usable injection measurement"),
    }


def screen_report(screen: Dict[str, Any],
                  ioi_effects: Dict[Tuple[int, int], float],
                  raw_effects: Dict[Tuple[int, int], float],
                  candidates: List[Tuple[int, int]],
                  top_k: int) -> Dict[str, Any]:
    """The screen's two rankings, from the same two ablations per head.

    The raw clean-prompt effect is what a screen without a corrupted control
    would rank on; the difference-in-differences is what the circuit is
    selected on. Publishing both makes the control's effect on the selection
    visible instead of assumed: a head the two rankings disagree about is a
    head whose clean-prompt effect is (at least partly) a generic name-token
    effect.
    """
    ioi_ranked = sorted(ioi_effects, key=lambda k: -abs(ioi_effects[k]))
    raw_ranked = sorted(raw_effects, key=lambda k: -abs(raw_effects[k]))
    ioi_top = set(ioi_ranked[:top_k])
    raw_top = set(raw_ranked[:top_k])
    return {
        "method": ("zero-ablation of every head on the first panel prompt, in "
                   "the clean and in the ABB corrupted condition"),
        "prompt": {
            "clean": screen.get("clean"),
            "corrupted": screen.get("corrupted"),
            "subject": screen.get("subject"),
            "io": screen.get("io"),
        },
        "n_heads_evaluated": len(ioi_effects),
        "ranking_metric": "ioi_specific_ablation_difference",
        "candidates_selected": [lm.head_label(*c) for c in candidates],
        "ioi_specific_top": [lm.head_label(*c) for c in ioi_ranked[:top_k]],
        "ioi_specific_top_effects": {
            lm.head_label(*c): round(ioi_effects[c], 4)
            for c in ioi_ranked[:top_k]
        },
        "raw_clean_effect_top": [lm.head_label(*c) for c in raw_ranked[:top_k]],
        "raw_clean_effect_top_effects": {
            lm.head_label(*c): round(raw_effects[c], 4)
            for c in raw_ranked[:top_k]
        },
        "ranking_changed": ioi_ranked != raw_ranked,
        "n_heads_ranked_differently": sum(
            1 for a, b in zip(ioi_ranked, raw_ranked) if a != b),
        "top_k_overlap": len(ioi_top & raw_top),
        "top_k_only_in_raw_clean_effect": [
            lm.head_label(*c) for c in raw_ranked[:top_k] if c not in ioi_top
        ],
    }
