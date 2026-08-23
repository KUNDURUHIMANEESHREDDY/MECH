r"""Counterfactual Invariance Engine for MECH.

ALL measurements come from the live GPT-2 model via Gpt2LiveExperimentRunner.
No hardcoded numeric values exist in this module.

Tests whether I(c(M)) ≈ I(M) across three structural modifications:
    NEURON_PERMUTATION    — randomly permutes MLP neuron indices in one layer
    ROUTING_SPARSITY      — masks 20% of attention heads to zero
    DISTRACTOR_INSERTION  — inserts an orthogonal distractor token in the prompt

Invariance distance = |I(c(M)) - I(M)| measured from real GPT-2 logit diffs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class CounterfactualInvarianceResult:
    perturbation_type: str
    baseline_causal_effect: float
    counterfactual_causal_effect: float
    invariance_distance: float
    is_counterfactually_invariant: bool
    probe_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "perturbation_type": self.perturbation_type,
            "baseline_causal_effect": round(self.baseline_causal_effect, 4),
            "counterfactual_causal_effect": round(self.counterfactual_causal_effect, 4),
            "invariance_distance": round(self.invariance_distance, 4),
            "is_counterfactually_invariant": self.is_counterfactually_invariant,
            "probe_id": self.probe_id,
        }


# Backward-compatible alias
CounterfactualInvarianceRecord = CounterfactualInvarianceResult


class CounterfactualInvarianceEngine:
    """
    Tests causal mechanism invariance under structural model modifications.
    Requires a Gpt2LiveExperimentRunner to obtain real measurements.
    """

    def test_counterfactual_invariance(
        self,
        probe=None,
        runner=None,
    ) -> List[CounterfactualInvarianceResult]:
        """
        Tests counterfactual invariance across three perturbation types on GPT-2.

        Parameters
        ----------
        probe  : DynamicProbe (required)
        runner : Gpt2LiveExperimentRunner (required)

        Returns
        -------
        List[CounterfactualInvarianceResult] with real invariance distances.

        Raises
        ------
        ValueError if probe or runner is None.
        """
        if probe is None or runner is None:
            raise ValueError(
                "CounterfactualInvarianceEngine.test_counterfactual_invariance() requires "
                "both `probe` and `runner`. Hardcoded results are not permitted."
            )

        perturbation_types = [
            "NEURON_PERMUTATION",
            "ROUTING_SPARSITY",
            "DISTRACTOR_INSERTION",
        ]

        results: List[CounterfactualInvarianceResult] = []
        for ptype in perturbation_types:
            live = runner.measure_counterfactual_invariance(probe, ptype)
            results.append(CounterfactualInvarianceResult(
                perturbation_type=live.perturbation_type,
                baseline_causal_effect=live.baseline_causal_effect,
                counterfactual_causal_effect=live.counterfactual_causal_effect,
                invariance_distance=live.invariance_distance,
                is_counterfactually_invariant=live.is_counterfactually_invariant,
                probe_id=probe.probe_id,
            ))

        return results
