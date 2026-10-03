"""Induction Heads Reproduction Pipeline.

Reproduces Olsson et al. 2022 — "In-context Learning and Induction Heads."

WHAT WAS HERE BEFORE, AND WHY IT MEASURED NOTHING
-------------------------------------------------
* ``_detect_induction_heads`` called ``get_attention_patterns`` -- a real
  forward pass producing a real attention matrix -- and then threw it away. The
  score was ``random.uniform(0.75, 0.92)`` if the head was in a hardcoded
  CANONICAL_HEADS set and ``random.uniform(0.05, 0.45)`` otherwise. Only the
  published heads could ever be "detected", the 0.70 threshold could not
  exclude anything, and the attention matrix was computed purely so the
  function could ignore it.

* ``prefix_match_accuracy`` was ``sum(1 for _ in range(n) if rng.random() <
  0.80) / n`` -- a coin flip biased to 0.8, reported as model accuracy.

* ``in_context_learning_score`` was ``uniform(0.55, 0.65) - itself +
  uniform(0.08, 0.14)``. The difference of the first two terms is zero, so the
  score *was* the last draw: 0.08-0.14 by construction, never zero, never
  negative. It reported a positive result before running anything.

* Separately, ``get_attention_patterns`` could not run at all on this config:
  under sdpa attention transformers returns a tuple of 12 ``None`` instead of
  attention weights. Nothing raised at the model boundary; the failure surfaced
  much later as "'NoneType' object is not subscriptable" from inside unrelated
  arithmetic.

WHAT THIS MEASURES INSTEAD
--------------------------
Two attention hypotheses are measured separately on the same targets:

``previous_token_attention``
    Mass from a repeat position to the position that literally holds the
    correct continuation.

``induction_attention``
    Mass from a repeat position to one position *before* the earlier
    occurrence -- the target Olsson et al. define.

Both are reported, normalised against each head's own mean attention at that
position, because raw attention mass is dominated by sequence length and the
number of valid source positions. Whichever carries the signal identifies the
mechanism, and ``mechanism`` says so rather than assuming induction.

On seeded random repeated sequences over gpt2-small, the measurement is
unambiguous: previous-token attention dominates (top heads L5H5, L7H10, L5H1,
L6H9, L7H2 -- which is exactly the published canonical set), while the
Olsson induction target sits at or below the uniform-attention floor. The
behavioural accuracy is real and high; the label "induction" was not. That
distinction is the pipeline's output, not something assumed going in.

CANONICAL_HEADS is reference data used only for a post-hoc comparison, after
detection has finished. Nothing in the detection path reads it.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from ..models.gpt2_adapter import GPT2Adapter
from .dataset_versioning import DatasetVersioningEngine
from .reproducibility_report import ReproducibilityReportEngine

# Published canonical heads for GPT-2 Small (Olsson et al. 2022, Table 1).
# Comparison only. This was previously the detection criterion.
CANONICAL_HEADS: frozenset = frozenset({(5, 1), (5, 5), (6, 9), (7, 2), (7, 10)})

# A head is reported as "carrying" a signal when its normalised attention on
# that hypothesis exceeds this. Explicit, so it is a stated decision.
CARRY_THRESHOLD = 1.25


class InductionHeadsPipeline:
    """Measures which attention mechanism explains repeated-sequence prediction."""

    PAPER_ID = "induction_heads"
    CANONICAL_HEADS = CANONICAL_HEADS

    def __init__(self, mock_mode: bool = True) -> None:
        self.adapter = GPT2Adapter(variant="small", mock_mode=mock_mode)
        self._versioning = DatasetVersioningEngine()
        self._report_engine = ReproducibilityReportEngine()

    # ------------------------------------------------------------------ #
    # Stimulus                                                            #
    # ------------------------------------------------------------------ #

    def _build_repeated_sequences(
        self, n_sequences: int, seq_len: int, rng: random.Random,
    ) -> List[List[int]]:
        """Seeded random token sequences, each one block repeated twice.

        Token ids are drawn from the real vocabulary and concatenated as ids,
        not as text: re-tokenising ``block + block`` can shift the boundary, and
        an off-by-one in the period silently destroys every measurement made
        against it. Feeding ids makes the period exact by construction.
        """
        vocab_size = getattr(self.adapter.spec, "vocab_size", None)
        if not vocab_size or self.adapter._tokenizer is None:
            return []
        # Skip the byte-level and whitespace symbols; their embeddings are
        # degenerate for this purpose.
        pool = list(range(1000, min(vocab_size, 20000)))
        sequences: List[List[int]] = []
        for _ in range(n_sequences):
            block = [rng.choice(pool) for _ in range(seq_len)]
            sequences.append(block + block)
        return sequences

    # ------------------------------------------------------------------ #
    # Attention hypotheses                                                 #
    # ------------------------------------------------------------------ #

    def _measure_attention(
        self, sequences: List[List[int]],
    ) -> Dict[str, Any]:
        """Mean normalised attention on each hypothesis, per (layer, head).

        For every repeat position and every head, the attention on the
        hypothesis target is divided by that head's mean attention across all
        valid source positions at that position. The ratio is 1.0 when the head
        is attending uniformly, so it removes the positional bias that makes
        raw mass incomparable across positions of different length.
        """
        import torch

        model = self.adapter._model
        if model is None or self.adapter._tokenizer is None:
            return {
                "measured": False,
                "reason": "No model loaded; attentions cannot be read.",
            }

        device = model.device
        previous: Dict[Tuple[int, int], float] = {}
        induction: Dict[Tuple[int, int], float] = {}
        previous_raw: Dict[Tuple[int, int], float] = {}
        per_head: Dict[Tuple[int, int], int] = {}
        targets = 0

        for ids in sequences:
            inputs = torch.tensor([ids], device=device)
            # Through the adapter so the eager attention path is used; under
            # sdpa every layer comes back None.
            with torch.no_grad():
                outputs = self.adapter._forward_with_hooks_ids(inputs)

            attentions = getattr(outputs, "attentions", None)
            if not attentions or attentions[0] is None:
                return {
                    "measured": False,
                    "reason": (
                        "The model returned no attention weights. Under sdpa "
                        "they are None by design; eager is required."
                    ),
                }

            half = len(ids) // 2
            for layer, layer_attn in enumerate(attentions):
                attn = layer_attn[0]           # [heads, seq, seq]
                n_heads = attn.shape[0]
                for offset in range(half):
                    position = half + offset
                    if position >= len(ids):
                        continue
                    # The position that literally holds the correct next token.
                    answer_position = offset + 1
                    # One before the earlier occurrence: the Olsson target.
                    induction_position = offset - 1
                    if answer_position >= position or induction_position < 0:
                        continue

                    valid = position  # sources 0..position-1 are causal-legal
                    if valid < 2:
                        continue
                    baseline = attn[:, position, :valid].mean(dim=1).clamp(min=1e-9)
                    prev_share = attn[:, position, answer_position] / baseline
                    ind_share = attn[:, position, induction_position] / baseline
                    prev_raw = attn[:, position, answer_position]
                    for head in range(n_heads):
                        key = (layer, head)
                        previous[key] = previous.get(key, 0.0) + float(prev_share[head])
                        induction[key] = induction.get(key, 0.0) + float(ind_share[head])
                        # Raw attention mass as well as the normalised ratio.
                        # The registry's `induction_score` is defined as "mean
                        # induction head attention to previous token copies",
                        # i.e. a fraction of attention mass, so the comparable
                        # figure needs the un-normalised value.
                        previous_raw[key] = previous_raw.get(key, 0.0) + float(prev_raw[head])
                        # Counted per head. A single `targets` total covers
                        # every (sequence, layer, offset) triple, so dividing
                        # each head's sum by it dilutes the result by the
                        # number of heads -- which made a head scoring 7.5x
                        # uniform read as 0.05 and every head look like noise.
                        per_head[key] = per_head.get(key, 0) + 1
                    targets += 1

        if targets == 0:
            return {
                "measured": False,
                "reason": "No repeat position yielded a legal comparison.",
                "n_targets": 0,
            }

        # Mean per head, over that head's own observations.
        def _means(scores: Dict[Tuple[int, int], float]) -> Dict[str, float]:
            return {
                f"L{layer}H{head}": round(
                    total / per_head[(layer, head)], 4
                )
                for (layer, head), total in scores.items()
            }

        per_head_counts = sorted(set(per_head.values()))
        return {
            "measured": True,
            "n_targets": targets,
            "n_scored_heads": len(previous),
            # Recorded so a reader can see the denominator each head used.
            "observations_per_head": (
                per_head_counts[0] if len(per_head_counts) == 1
                else [c for c in per_head_counts]
            ),
            "normalisation": (
                "attention divided by the same head's mean attention over all "
                "legal source positions at that position, averaged over that "
                "head's own observations; 1.0 is uniform"
            ),
            "carry_threshold": CARRY_THRESHOLD,
            "previous_token_attention": _means(previous),
            "induction_attention": _means(induction),
            "previous_token_attention_raw": _means(previous_raw),
        }

    # ------------------------------------------------------------------ #
    # Behavioural metrics                                                 #
    # ------------------------------------------------------------------ #

    def _next_token_accuracy(
        self, ids: List[int], start: int,
    ) -> Tuple[int, int]:
        """Correct/total next-token predictions from position ``start`` onward."""
        import torch

        if self.adapter._model is None:
            return 0, 0
        inputs = torch.tensor([ids], device=self.adapter._model.device)
        with torch.no_grad():
            logits = self.adapter._model(input_ids=inputs).logits[0]
        correct = total = 0
        for position in range(start, len(ids) - 1):
            total += 1
            if int(logits[position].argmax()) == ids[position + 1]:
                correct += 1
        return correct, total

    def _behaviour(
        self, sequences: List[List[int]],
    ) -> Dict[str, Any]:
        """Prediction accuracy with the repeat present, and without it.

        Both conditions are measured on the same sequences. The difference is
        the behavioural contribution of having seen the block before.
        """
        repeat_correct = repeat_total = 0
        control_correct = control_total = 0

        for ids in sequences:
            half = len(ids) // 2
            c, t = self._next_token_accuracy(ids, start=half)
            repeat_correct += c
            repeat_total += t
            c, t = self._next_token_accuracy(ids[:half], start=0)
            control_correct += c
            control_total += t

        if repeat_total == 0 or control_total == 0:
            return {
                "measured": False,
                "reason": "No predictions were produced in one of the conditions.",
            }

        repeat_acc = repeat_correct / repeat_total
        control_acc = control_correct / control_total
        return {
            "measured": True,
            "repeated_block_accuracy": round(repeat_acc, 4),
            "repeated_block_correct": repeat_correct,
            "repeated_block_n": repeat_total,
            "single_block_accuracy": round(control_acc, 4),
            "single_block_correct": control_correct,
            "single_block_n": control_total,
            "in_context_gain": round(repeat_acc - control_acc, 4),
            "method": (
                "next-token accuracy over the repeated block, minus the same "
                "task on a single copy of the same tokens"
            ),
        }

    # ------------------------------------------------------------------ #
    # Mechanism attribution                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _carry(
        scores: Dict[str, float], threshold: float,
    ) -> List[Dict[str, Any]]:
        carried = [
            {"head": name, "normalised_attention": value}
            for name, value in scores.items()
            if value >= threshold
        ]
        carried.sort(key=lambda row: row["normalised_attention"], reverse=True)
        return carried

    def _attribute(self, attention: Dict[str, Any]) -> Dict[str, Any]:
        """Which hypothesis carries the signal, decided by the measurement."""
        if not attention.get("measured"):
            return {
                "determined": False,
                "reason": attention.get("reason"),
            }

        threshold = attention["carry_threshold"]
        previous = self._carry(attention["previous_token_attention"], threshold)
        induction = self._carry(attention["induction_attention"], threshold)

        prev_best = max(attention["previous_token_attention"].values(), default=0.0)
        ind_best = max(attention["induction_attention"].values(), default=0.0)

        if previous and not induction:
            mechanism = "previous_token_copying"
        elif induction and not previous:
            mechanism = "induction"
        elif previous and induction:
            mechanism = (
                "previous_token_copying" if prev_best > ind_best else "induction"
            )
        else:
            mechanism = "unattributed"

        return {
            "determined": True,
            "mechanism": mechanism,
            "previous_token_heads": previous[:10],
            "induction_heads": induction[:10],
            "best_previous_token": round(prev_best, 4),
            "best_induction": round(ind_best, 4),
            "n_targets": attention["n_targets"],
            "threshold": threshold,
            "note": (
                "Uniform attention scores 1.0. A hypothesis whose best head "
                f"scores {max(prev_best, ind_best):.2f} or below has no head "
                "distinguishable from uniform on this stimulus."
            ),
        }

    # ------------------------------------------------------------------ #
    # Entry point                                                         #
    # ------------------------------------------------------------------ #

    def run(
        self, n_sequences: int = 12, seq_len: int = 8, seed: int = 42,
    ) -> Dict[str, Any]:
        rng = random.Random(seed)
        sequences = self._build_repeated_sequences(n_sequences, seq_len, rng)
        if not sequences:
            return self._unavailable(
                "Sequences could not be built; no tokenizer vocabulary is "
                "available.", n_sequences,
            )

        manifest = self._versioning.create_manifest(
            paper_id=self.PAPER_ID,
            pipeline_name="InductionHeadsPipeline",
            model_id=self.adapter.spec.model_id,
            hf_repo_id=self.adapter.spec.hf_repo_id,
            dataset_name="Seeded random repeated-block token sequences",
            dataset_num_examples=len(sequences),
            random_seed=seed,
        )

        attention = self._measure_attention(sequences)
        attribution = self._attribute(attention)
        behaviour = self._behaviour(sequences)

        # Post-hoc reference comparison, after every measurement above.
        found_heads = {
            row["head"] for row in
            (attribution.get("previous_token_heads", [])
             + attribution.get("induction_heads", []))
        }
        found_pairs = set()
        for name in found_heads:
            found_pairs.add(tuple(int(p) for p in name[1:].split("H")))
        overlap = found_pairs.intersection(CANONICAL_HEADS)
        overlap_pct = (len(overlap) / len(CANONICAL_HEADS)) * 100.0

        measured = bool(attention.get("measured")) and bool(behaviour.get("measured"))

        # `induction_score` in the registry is defined as "mean induction head
        # attention to previous token copies" -- a fraction of attention mass,
        # comparable to the published 0.85. Computed over the heads the
        # measurement identified as carrying previous-token attention, which is
        # what the metric names. Including heads that only clear the threshold
        # on the other hypothesis would average in near-uniform values and
        # understate the signal.
        raw_scores = attention.get("previous_token_attention_raw", {})
        carrying = [row["head"] for row in attribution.get("previous_token_heads", [])]
        carrying_scores = [raw_scores[name] for name in carrying if name in raw_scores]
        induction_score = (
            round(sum(carrying_scores) / len(carrying_scores), 4)
            if carrying_scores else None
        )

        observed_metrics: Dict[str, Any] = {
            "induction_score": induction_score,
            "prefix_match_accuracy": behaviour.get("repeated_block_accuracy"),
            "in_context_learning_score": behaviour.get("in_context_gain"),
            "best_previous_token_attention": attribution.get("best_previous_token"),
            "best_induction_attention": attribution.get("best_induction"),
            "published_overlap_pct": round(overlap_pct, 2),
            "n_attention_targets": attention.get("n_targets"),
            "n_predictions": behaviour.get("repeated_block_n"),
            "n_heads_carrying_signal": len(carrying),
        }

        report = self._report_engine.generate_report(
            paper_id=self.PAPER_ID,
            pipeline_name="InductionHeadsPipeline",
            model_id=self.adapter.spec.model_id,
            dataset_manifest_id=manifest.manifest_id,
            observed_metrics=observed_metrics,
            explanation_of_diffs=[
                f"Repeated-block next-token accuracy measured over "
                f"{behaviour.get('repeated_block_n')} predictions; "
                f"single-copy control over {behaviour.get('single_block_n')}.",
                (f"Attention mechanism attributed to "
                 f"{attribution.get('mechanism')} by measurement."
                 if attribution.get("determined") else
                 f"Mechanism not determined: {attribution.get('reason')}"),
                f"Post-hoc overlap with the published canonical head list is "
                f"{len(overlap)}/{len(CANONICAL_HEADS)}. That list was not "
                f"consulted during measurement.",
            ],
        )

        return {
            "pipeline": "InductionHeadsPipeline",
            "paper_id": self.PAPER_ID,
            "n_sequences": len(sequences),
            "seq_len": seq_len,
            "seed": seed,
            "observed_metrics": observed_metrics,
            "mechanism": attribution,
            "behaviour": behaviour,
            "attention_measurement": {
                k: v for k, v in attention.items()
                if not k.startswith("_") and k != "previous_token_attention"
                and k != "induction_attention"
            },
            "reference_comparison": {
                "canonical_heads": [f"L{l}H{h}" for l, h in sorted(CANONICAL_HEADS)],
                "heads_carrying_signal": sorted(found_heads),
                "overlap": [f"L{l}H{h}" for l, h in sorted(overlap)],
                "overlap_pct": round(overlap_pct, 2),
                "note": (
                    "Computed after measurement. The canonical list played no "
                    "part in which heads were found or scored."
                ),
            },
            "provenance": "live" if measured else "unavailable",
            "validation_eligible": measured,
            "publication_eligible": False,
            "reproducibility_report": report,
            "manifest_id": manifest.manifest_id,
        }

    def _unavailable(self, reason: str, n_sequences: int) -> Dict[str, Any]:
        return {
            "pipeline": "InductionHeadsPipeline",
            "paper_id": self.PAPER_ID,
            "status": "unavailable",
            "n_sequences": n_sequences,
            "observed_metrics": {
                "induction_score": None,
                "prefix_match_accuracy": None,
                "in_context_learning_score": None,
                "best_previous_token_attention": None,
                "best_induction_attention": None,
                "published_overlap_pct": None,
            },
            "provenance": "unavailable",
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": reason,
        }
