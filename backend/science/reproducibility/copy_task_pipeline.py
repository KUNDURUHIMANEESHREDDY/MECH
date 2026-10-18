"""Copy Task Reproduction Pipeline.

Measures the copying behaviour Elhage et al. 2021 / Olsson et al. 2022 describe,
by comparing attention on *repeated* token blocks against a **shuffled control**
of the same blocks, and by measuring the behavioural cost of destroying the
repeat.

What this used to be
--------------------
The module returned, for every caller and with no branch::

    # Simulated metrics for the mock/test environment
    return {
        "Copy Score": 0.92,
        "Attention to previous occurrence": 0.88,
    }

There was no mock flag, so callers with weights loaded received those two numbers
and `benchmark_runner` stored them in a reproducibility report. Neither was ever
derived from attention weights, and both are implausibly high for GPT-2 small on
an induction-style measurement.

Why the machinery is imported rather than re-implemented
--------------------------------------------------------
`induction_heads_pipeline` already measures previous-token attention against a
uniform-attention baseline, normalised per head, on repeated blocks built from
real token ids so the block period is exact. A second, independent
implementation of the same quantity would be a weaker measurement with its own
undiscovered off-by-ones, and any disagreement between the two would be
unresolvable. This pipeline therefore *uses* that measurement and adds the one
thing it does not have: a control condition in which the repeat is destroyed.

The control
-----------
The second block is shuffled in place. Token multiset, sequence length and the
first block are unchanged, so the only difference is that the position which
previously held the correct continuation now holds an arbitrary token from the
same block. Attention to that position in the shuffled condition is the
like-for-like baseline; the repeated-minus-shuffled margin is the copying
signal, and it is measured rather than asserted.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from ..models.adapter_base import LiveUnavailable
from .dataset_versioning import DatasetVersioningEngine
from .induction_heads_pipeline import CANONICAL_HEADS, InductionHeadsPipeline
from .reproducibility_report import ReproducibilityReportEngine
from .pipeline_contract import LIVE, completed

#: A head is treated as carrying the copying signal when its normalised
#: previous-token attention clears this. Uniform attention scores 1.0, so this
#: is "distinguishable from uniform", the same decision the induction pipeline
#: states explicitly. Imported by value rather than shared so a change there
#: does not silently change what this pipeline reports.
CARRY_THRESHOLD = 1.25


class CopyTaskPipeline:
    """Copy/induction measurement with a shuffled-block control.

    Needs a model manager to obtain weights. `benchmark_runner` constructs this
    class as ``CopyTaskPipeline(self.model_manager)``; without one there is no
    route to weights at all, so `run` refuses rather than loading a model of its
    own and reporting the result under a manager the caller did not supply.
    """

    PAPER_ID = "copy_task"

    def __init__(self, model_manager: Any = None,
                 mock_mode: bool = False) -> None:
        self.model_manager = model_manager
        self.mock_mode = mock_mode
        self._induction: Optional[InductionHeadsPipeline] = None
        self._versioning = DatasetVersioningEngine()
        self._report_engine = ReproducibilityReportEngine()

    # ------------------------------------------------------------------ #
    # Weight availability                                                 #
    # ------------------------------------------------------------------ #

    def _measurement_rig(self) -> InductionHeadsPipeline:
        """The induction-heads rig, backed by real weights, or a refusal.

        `ModelAdapter._load_model` converts any load failure into a mock-mode
        fallback, so the adapter has to be checked for `simulated` rather than
        trusted because it was constructed with `mock_mode=False`.
        """
        if self.mock_mode:
            raise LiveUnavailable(
                "CopyTaskPipeline was constructed in mock mode. A copy-task "
                "measurement is a forward-pass result and has no fixture; no "
                "weights were read and no score is reported."
            )
        if self.model_manager is None:
            raise LiveUnavailable(
                "CopyTaskPipeline was constructed without a model manager, so it "
                "has no way to obtain weights. No copy-task measurement was "
                "performed."
            )
        if self._induction is None:
            rig = InductionHeadsPipeline(mock_mode=False)
            adapter = rig.adapter
            if (adapter.spec.mock_mode or adapter._model is None
                    or getattr(adapter, "simulated", False)):
                fallback = (getattr(adapter, "simulation_reason", "")
                            or "no reason recorded")
                raise LiveUnavailable(
                    "GPT-2 weights are not loaded, so no copy-task measurement "
                    "can be made. The adapter fell back to its mock fixtures "
                    f"({fallback})."
                )
            self._induction = rig
        return self._induction

    # ------------------------------------------------------------------ #
    # Stimulus                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _shuffle_second_block(ids: List[int], rng: random.Random) -> List[int]:
        """Destroy the repeat while preserving length and token multiset."""
        half = len(ids) // 2
        head, tail = list(ids[:half]), list(ids[half:])
        rng.shuffle(tail)
        return head + tail

    # ------------------------------------------------------------------ #
    # Measurement                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _mean(values: List[float]) -> Optional[float]:
        return round(sum(values) / len(values), 6) if values else None

    def run(self, n_sequences: int = 12, seq_len: int = 8, seed: int = 42,
            model_id: str = "gpt2-small") -> Dict[str, Any]:
        """Measure copying attention and its behavioural cost.

        `model_id` is accepted because the benchmark runner probes signatures
        and passes it; this pipeline measures whichever model the adapter holds
        and records that id, rather than pretending to switch models.
        """
        rig = self._measurement_rig()
        adapter = rig.adapter

        rng = random.Random(seed)
        repeated = rig._build_repeated_sequences(n_sequences, seq_len, rng)
        if not repeated:
            raise LiveUnavailable(
                "No repeated sequences could be built: the tokenizer vocabulary "
                "is unavailable, so there is nothing to measure."
            )
        shuffle_rng = random.Random(seed + 1)
        shuffled = [self._shuffle_second_block(ids, shuffle_rng) for ids in repeated]
        changed = sum(1 for a, b in zip(repeated, shuffled) if a != b)

        repeated_attention = rig._measure_attention(repeated)
        shuffled_attention = rig._measure_attention(shuffled)
        if not repeated_attention.get("measured") or not shuffled_attention.get(
                "measured"):
            raise LiveUnavailable(
                "Attention weights could not be read for one of the two "
                "conditions, so the repeated-versus-shuffled contrast is not "
                "measured. Repeated: "
                f"{repeated_attention.get('reason')}; shuffled: "
                f"{shuffled_attention.get('reason')}"
            )

        prev_repeat = repeated_attention["previous_token_attention"]
        prev_shuffle = shuffled_attention["previous_token_attention"]
        raw_repeat = repeated_attention["previous_token_attention_raw"]

        # Heads selected by the *repeated* condition alone, then evaluated in
        # both conditions on the same head set. Selecting on the control would
        # pick whichever heads happen to be noisy there.
        carrying = sorted(
            (name for name, value in prev_repeat.items()
             if value >= CARRY_THRESHOLD),
            key=lambda name: prev_repeat[name], reverse=True)

        # Like for like. The repeated condition's raw attention *fraction* and
        # the shuffled condition's raw attention fraction are the same quantity,
        # so their difference is a margin. An earlier version subtracted the
        # shuffled condition's normalised attention (a multiple of uniform) from
        # the repeated condition's raw mass, which produced a number that was a
        # difference of two different scales and meant nothing.
        raw_shuffle = shuffled_attention["previous_token_attention_raw"]
        copy_scores = [raw_repeat[name] for name in carrying if name in raw_repeat]
        control_scores = [raw_shuffle[name] for name in carrying
                          if name in raw_shuffle]
        repeat_norm = [prev_repeat[name] for name in carrying]
        shuffle_norm = [prev_shuffle[name] for name in carrying]

        # Behavioural measurement, from the rig's own next-token accuracy, which
        # runs a real forward pass per sequence.
        traces: List[Dict[str, Any]] = []
        for index, (rep_ids, shuf_ids) in enumerate(zip(repeated, shuffled)):
            half = len(rep_ids) // 2
            rep_correct, rep_total = rig._next_token_accuracy(rep_ids, start=half)
            shuf_correct, shuf_total = rig._next_token_accuracy(shuf_ids, start=half)
            rep_acc = (rep_correct / rep_total) if rep_total else None
            shuf_acc = (shuf_correct / shuf_total) if shuf_total else None
            traces.append({
                "sequence_index": index,
                "n_tokens": len(rep_ids),
                "block_len": half,
                "block_token_ids": list(rep_ids[:half]),
                "repeat_correct": rep_correct,
                "repeat_total": rep_total,
                "repeat_accuracy": (None if rep_acc is None else round(rep_acc, 6)),
                "shuffled_correct": shuf_correct,
                "shuffled_total": shuf_total,
                "shuffled_accuracy": (
                    None if shuf_acc is None else round(shuf_acc, 6)),
                # The per-sequence boolean. None when either condition produced
                # no legal prediction at all, which is unmeasured, not False.
                "repeat_beats_shuffle": (
                    None if (rep_acc is None or shuf_acc is None)
                    else bool(rep_acc > shuf_acc)),
                "shuffled_second_block_differs": bool(rep_ids != shuf_ids),
                "provenance": LIVE,
                "synthetic_fields": [],
            })

        usable = [t for t in traces if t["repeat_beats_shuffle"] is not None]
        if not usable:
            raise LiveUnavailable(
                "No sequence produced a legal next-token prediction in both "
                "conditions, so the behavioural contrast is unmeasured."
            )
        repeat_acc_all = [t["repeat_accuracy"] for t in usable]
        shuffled_acc_all = [t["shuffled_accuracy"] for t in usable]
        n_rep_correct = sum(t["repeat_correct"] for t in usable)
        n_rep_total = sum(t["repeat_total"] for t in usable)

        copy_score = self._mean(copy_scores)
        control_score = self._mean(control_scores)
        margin = (None if (copy_score is None or control_score is None)
                  else round(copy_score - control_score, 6))
        repeat_norm_mean = self._mean(repeat_norm)
        shuffle_norm_mean = self._mean(shuffle_norm)
        # Ratio of the two normalised means. 1.0 means the carrying heads treat
        # the copied position no differently from any other source; the repeated
        # condition should sit well above it if copying is happening.
        normalised_gain = (
            None if (repeat_norm_mean is None or not shuffle_norm_mean)
            else round(repeat_norm_mean / shuffle_norm_mean, 6))

        # Post-hoc reference comparison, after every measurement above. The
        # canonical list is not consulted while selecting heads.
        found_pairs = set()
        for name in carrying:
            try:
                layer, head = (int(p) for p in name[1:].split("H"))
            except ValueError:
                continue
            found_pairs.add((layer, head))
        overlap = found_pairs.intersection(CANONICAL_HEADS)
        overlap_pct = (len(overlap) / len(CANONICAL_HEADS)) * 100.0 if CANONICAL_HEADS else None

        measured = bool(carrying) and bool(usable)
        observed_metrics = {
            # Mean fraction of attention mass the carrying heads put on the
            # position holding the correct continuation. A fraction of mass, so
            # it is comparable to a published copy score; NOT comparable to the
            # old hardcoded 0.92, which was labelled the same way.
            "copy_score": copy_score,
            "shuffled_control_score": control_score,
            "copy_attention_margin": margin,
            "mean_normalised_repeat_attention": repeat_norm_mean,
            "mean_normalised_shuffled_attention": shuffle_norm_mean,
            "normalised_attention_gain": normalised_gain,
            "score_units": (
                "copy_score and shuffled_control_score are mean attention "
                "fractions at the copied position; the margin is their "
                "difference. The normalised figures are multiples of the same "
                "head's mean attention over legal sources, where 1.0 is uniform."),
            "n_heads_carrying_signal": len(carrying),
            "carry_threshold": CARRY_THRESHOLD,
            "prefix_match_accuracy": (
                round(n_rep_correct / n_rep_total, 4) if n_rep_total else None),
            "shuffled_match_accuracy": self._mean(
                [v for v in shuffled_acc_all if v is not None]),
            "copy_behaviour_gain": (
                None if not n_rep_total else round(
                    (n_rep_correct / n_rep_total)
                    - (sum(shuffled_acc_all) / len(shuffled_acc_all)), 4)),
            "n_predictions": n_rep_total,
            "n_sequences_compared": len(usable),
            "n_sequences_where_shuffle_changed_the_block": changed,
            "published_overlap_pct": (
                None if overlap_pct is None else round(overlap_pct, 2)),
            "carrying_heads": carrying[:10],
        }

        manifest = self._versioning.create_manifest(
            paper_id=self.PAPER_ID,
            pipeline_name="CopyTaskPipeline",
            model_id=adapter.spec.model_id,
            hf_repo_id=adapter.spec.hf_repo_id,
            dataset_name="Seeded repeated-block token sequences, plus a shuffled control",
            dataset_num_examples=len(repeated),
            random_seed=seed,
        )
        report = self._report_engine.generate_report(
            paper_id=self.PAPER_ID,
            pipeline_name="CopyTaskPipeline",
            model_id=adapter.spec.model_id,
            dataset_manifest_id=manifest.manifest_id,
            observed_metrics=observed_metrics,
            explanation_of_diffs=[
                "The control is the same repeated-block stimulus with its second "
                "block shuffled in place: identical length and token multiset, "
                "no repeat. The margin between the two conditions is the "
                "copying signal; either number alone is not.",
                f"Heads were selected in the repeated condition only "
                f"({len(carrying)} above {CARRY_THRESHOLD}) and then evaluated in "
                f"both conditions on that same head set.",
                f"Post-hoc overlap with the published canonical head list is "
                f"{len(overlap)}/{len(CANONICAL_HEADS)}. That list played no part "
                f"in which heads were measured.",
                "The old module returned a fixed copy score of 0.92 and an "
                "attention figure of 0.88 for every caller. Neither number "
                "appears here, and nothing in this pipeline can produce them "
                "without a forward pass.",
            ],
        )

        return completed(
            pipeline="CopyTaskPipeline",
            paper_id=self.PAPER_ID,
            model_id=adapter.spec.model_id,
            observed_metrics=observed_metrics,
            raw_traces=traces,
            provenance_note=(
                "Attention masses and next-token accuracies were read from "
                "forward passes on the loaded weights, for both the repeated and "
                "the shuffled condition."
            ),
            validation_eligible=measured,
            ineligible_because=(None if measured else [
                "no head cleared the carry threshold, so no copying signal was "
                "attributable",
            ]),
            publication_eligible=False,
            model_id_requested=model_id,
            repeated_attention={k: v for k, v in repeated_attention.items()
                                if k not in ("previous_token_attention",
                                             "induction_attention")},
            shuffled_attention={k: v for k, v in shuffled_attention.items()
                                if k not in ("previous_token_attention",
                                             "induction_attention")},
            reference_comparison={
                "canonical_heads": [f"L{l}H{h}" for l, h in sorted(CANONICAL_HEADS)],
                "heads_carrying_signal": carrying,
                "overlap": [f"L{l}H{h}" for l, h in sorted(overlap)],
                "overlap_pct": (None if overlap_pct is None
                                else round(overlap_pct, 2)),
                "note": (
                    "Computed after measurement. The canonical list played no "
                    "part in which heads were selected or scored."
                ),
            },
            behaviour={
                "method": (
                    "next-token accuracy over the repeated block, versus the "
                    "same positions with the second block shuffled"
                ),
                "repeat_correct": n_rep_correct,
                "repeat_total": n_rep_total,
                "n_sequences_compared": len(usable),
            },
            weights_fingerprint=adapter.model_fingerprint(),
            reproducibility_report=report,
            manifest_id=manifest.manifest_id,
        )
