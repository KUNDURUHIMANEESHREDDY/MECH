r"""Feature Interpretability & Monosemanticity Analyzer for MECH SAEs.

Computes the Monosemantic Specificity Index (MSI) and measures the polysemantic gap
between raw model neurons and SAE feature dictionary directions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch

from ..sae_interface import SAEInterface


@dataclass
class MonosemanticityReport:
    """Report comparing polysemantic neuron interference vs SAE feature clarity."""
    concept_label: str
    sae_feature_idx: int
    sae_msi: float  # Monosemantic Specificity Index [0, 1]
    raw_neuron_idx: int
    raw_neuron_msi: float
    polysemantic_gap: float  # sae_msi - raw_neuron_msi
    is_monosemantic: bool
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_label": self.concept_label,
            "sae_feature_idx": self.sae_feature_idx,
            "sae_msi": round(self.sae_msi, 4),
            "raw_neuron_idx": self.raw_neuron_idx,
            "raw_neuron_msi": round(self.raw_neuron_msi, 4),
            "polysemantic_gap": round(self.polysemantic_gap, 4),
            "is_monosemantic": self.is_monosemantic,
            "summary": self.summary,
        }


class FeatureInterpretabilityAnalyzer:
    """Measures monosemantic specificity of SAE features vs raw neurons."""

    @staticmethod
    def compute_msi(target_activation: float, distractor_activations: List[float]) -> float:
        """Computes the Monosemantic Specificity Index (MSI).

        MSI = a_target / (a_target + sum(a_distractors) + eps)
        """
        denom = target_activation + sum(distractor_activations) + 1e-6
        return float(target_activation / denom)

    @classmethod
    def evaluate_disentanglement(
        cls,
        concept_label: str,
        sae_target_act: float,
        sae_distractor_acts: List[float],
        sae_feature_idx: int,
        raw_neuron_target_act: float,
        raw_neuron_distractor_acts: List[float],
        raw_neuron_idx: int,
    ) -> MonosemanticityReport:
        """Generates a comparative monosemanticity report."""
        sae_msi = cls.compute_msi(sae_target_act, sae_distractor_acts)
        raw_msi = cls.compute_msi(raw_neuron_target_act, raw_neuron_distractor_acts)
        gap = sae_msi - raw_msi
        is_mono = sae_msi >= 0.70 and gap > 0.15

        summary = (
            f"Concept '{concept_label}': SAE Feature #{sae_feature_idx} achieves "
            f"MSI={sae_msi:.3f} (+{gap:.3f} gain over raw neuron #{raw_neuron_idx} MSI={raw_msi:.3f}), "
            f"confirming {'monosemantic isolation' if is_mono else 'partial disentanglement'}."
        )

        return MonosemanticityReport(
            concept_label=concept_label,
            sae_feature_idx=sae_feature_idx,
            sae_msi=sae_msi,
            raw_neuron_idx=raw_neuron_idx,
            raw_neuron_msi=raw_msi,
            polysemantic_gap=gap,
            is_monosemantic=is_mono,
            summary=summary,
        )
