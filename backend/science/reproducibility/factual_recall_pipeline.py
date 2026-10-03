"""Factual Recall Task Reproduction Pipeline.

Would reproduce the Rome/Memit-style factual-recall results (Meng et al. 2022;
Nostrand & Ochsendorf) by measuring subject-entity recall accuracy and locating
the recall circuit with causal scrubbing on real weights.

Status: NOT IMPLEMENTED
-----------------------
This module previously returned, for every caller and with no branch::

    # Simulated metrics for the mock/test environment
    return {
        "Subject Entity Recall Accuracy": 0.75,
        "Relation Attribute Attention": 0.65,
    }

The comment describes an intent the code did not implement: there was no mock
flag, so callers with weights loaded received these numbers, and
`benchmark_runner` stored them in a reproducibility report.

Neither value was ever computed. "Relation Attribute Attention" in particular
names no measurable quantity -- it does not say whose attention, at which layer,
relative to what baseline, or with what normaliser.

`run()` now raises. The scheduler and `benchmark_runner` both already treat a
raise as "did not run" and record the reason.

What implementing this requires: a named factual-recall dataset with its own
provenance, a stated prompting/formatting protocol, and a circuit localisation
that reports a per-layer effect derived from hooks on loaded weights. A dataset
choice matters especially here -- factual recall results are highly sensitive to
whether the evaluation set was already in the pretraining corpus, and an
unlabelled result cannot be checked for that.
"""

from __future__ import annotations

from typing import Any, Dict

from ..models.adapter_base import LiveUnavailable


class FactualRecallPipeline:
    """Not implemented. Raises rather than reporting a fabricated accuracy."""

    PAPER_ID = "factual_recall"

    def __init__(self, model_manager: Any = None) -> None:
        # Retained for call-site compatibility with the other pipelines.
        self.model_manager = model_manager

    def run(self, model_id: str = "gpt2-small") -> Dict[str, float]:
        raise LiveUnavailable(
            "FactualRecallPipeline is not implemented. It previously returned "
            "hardcoded values of 0.75 (Subject Entity Recall Accuracy) and 0.65 "
            "(Relation Attribute Attention) for every caller, including callers "
            "with weights loaded. Neither was computed. No factual-recall "
            "measurement is performed by this pipeline. See the module docstring "
            "for what a real implementation requires."
        )
