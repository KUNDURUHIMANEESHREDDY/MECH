r"""Cross-Modal Mechanism Adapter for MECH — GPT-2 Text-Domain Edition.

GPT-2 is a text-only model. Cross-modal measurements (Vision, Audio)
cannot be derived from GPT-2 weights. This module therefore measures
FUNCTIONAL CORRESPONDENCE within the text domain across different GPT-2
task categories (factual ↔ arithmetic ↔ relational ↔ ioi ↔ induction).

All scores come from real logit-lens trajectory comparisons run on the live
GPT-2 model via Gpt2LiveExperimentRunner.  No hardcoded 0.985, 0.942, 0.915.

Maps text task categories → CanonicalPrimitiveType:
    factual_recall  → MEMORY_LOOKUP
    arithmetic      → COMPARISON
    relational      → RELATIONAL_BINDING
    ioi             → ROUTING
    induction       → INHIBITION (suppressing wrong continuations)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ModalityType(str, Enum):
    TEXT       = "TEXT"
    VISION     = "VISION"
    AUDIO      = "AUDIO"
    MULTIMODAL = "MULTIMODAL"


class CanonicalPrimitiveType(str, Enum):
    RELATIONAL_BINDING = "RELATIONAL_BINDING"
    ROUTING            = "ROUTING"
    COMPARISON         = "COMPARISON"
    INHIBITION         = "INHIBITION"
    MEMORY_LOOKUP      = "MEMORY_LOOKUP"


# Canonical primitive for each text task category
_CATEGORY_TO_PRIMITIVE: Dict[str, CanonicalPrimitiveType] = {
    "factual_recall": CanonicalPrimitiveType.MEMORY_LOOKUP,
    "arithmetic":     CanonicalPrimitiveType.COMPARISON,
    "relational":     CanonicalPrimitiveType.RELATIONAL_BINDING,
    "ioi":            CanonicalPrimitiveType.ROUTING,
    "induction":      CanonicalPrimitiveType.INHIBITION,
}


@dataclass
class ModalityMappingRecord:
    modality: ModalityType
    substrate_component: str
    target_primitive: CanonicalPrimitiveType
    causal_correspondence_score: float   # real value from GPT-2 logit-lens comparison
    is_functionally_grounded: bool       # score >= 0.50

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality.value,
            "substrate_component": self.substrate_component,
            "target_primitive": self.target_primitive.value,
            "causal_correspondence_score": round(self.causal_correspondence_score, 4),
            "is_functionally_grounded": self.is_functionally_grounded,
        }


class CrossModalMechanismAdapter:
    """
    Measures real functional correspondence between GPT-2 text probe categories.

    GPT-2 is text-only. Instead of cross-modal alignment (TEXT↔VISION↔AUDIO),
    this adapter measures cross-category functional alignment within the text
    domain using live logit-lens trajectories from the real model.

    Requires a Gpt2LiveExperimentRunner to obtain real measurements.
    """

    def map_modality_component(
        self,
        modality: ModalityType,
        component_name: str,
        target_primitive: CanonicalPrimitiveType,
        probe_a=None,
        probe_b=None,
        runner=None,
    ) -> ModalityMappingRecord:
        """
        Measures real causal functional correspondence between two GPT-2 probes.

        Parameters
        ----------
        modality        : ModalityType (must be TEXT for GPT-2)
        component_name  : description of the substrate component being mapped
        target_primitive: CanonicalPrimitiveType target
        probe_a, probe_b: two DynamicProbe instances to compare (required)
        runner          : Gpt2LiveExperimentRunner (required)

        Returns
        -------
        ModalityMappingRecord with real causal_correspondence_score from GPT-2.

        Raises
        ------
        ValueError if probe_a, probe_b, or runner is None.
        ValueError if modality is not TEXT (GPT-2 is text-only).
        """
        if probe_a is None or probe_b is None or runner is None:
            raise ValueError(
                "CrossModalMechanismAdapter.map_modality_component() requires "
                "probe_a, probe_b, and runner (Gpt2LiveExperimentRunner). "
                "GPT-2 is text-only — cross-modal scores cannot be hardcoded."
            )
        if modality != ModalityType.TEXT:
            raise ValueError(
                f"GPT-2 is text-only. Cannot compute {modality.value} correspondence "
                f"from GPT-2 weights. Use ModalityType.TEXT and compare text probe categories."
            )

        fc = runner.measure_functional_correspondence(probe_a, probe_b)

        return ModalityMappingRecord(
            modality=ModalityType.TEXT,
            substrate_component=component_name,
            target_primitive=target_primitive,
            causal_correspondence_score=fc.causal_correspondence_score,
            is_functionally_grounded=fc.is_functionally_grounded,
        )

    def map_text_category_pair(
        self,
        category_a: str,
        category_b: str,
        probe_a=None,
        probe_b=None,
        runner=None,
    ) -> ModalityMappingRecord:
        """
        Convenience method: maps two text categories to their canonical primitives
        and measures real functional correspondence on GPT-2.
        """
        primitive_a = _CATEGORY_TO_PRIMITIVE.get(category_a, CanonicalPrimitiveType.RELATIONAL_BINDING)
        return self.map_modality_component(
            modality=ModalityType.TEXT,
            component_name=f"{category_a}_vs_{category_b}",
            target_primitive=primitive_a,
            probe_a=probe_a,
            probe_b=probe_b,
            runner=runner,
        )
