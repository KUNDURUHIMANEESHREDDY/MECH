"""Cross-Model Circuit Alignment Engine.

NOT IMPLEMENTED. ``compare_circuits`` returns fixed similarity scores for any
pair of model names, including pairs where the target model is not loadable at
all. It previously reported causal_similarity 0.91 and confidence 0.90 for
"GPT-2 Small" vs "Gemma-2B" without loading, running, or comparing anything.

A cross-model causal claim is one of the strongest statements this platform can
make, so a fixture must never be able to make it. The scores are kept for
catalog inspection but are labelled `reference`, ineligible, and accompanied by
the reason.
"""

from __future__ import annotations

from typing import Any, Dict
from .discovery_result import DiscoveryResultDTO


class CrossModelCircuitsEngine:
    """Reference fixtures only. No model is loaded and no comparison is run."""

    def compare_circuits(
        self,
        source_model: str = "GPT-2 Small",
        target_model: str = "Gemma-2B",
        circuit_type: str = "IOI",
    ) -> Dict[str, Any]:
        alignment = {
            "source_model": source_model,
            "target_model": target_model,
            "circuit_type": circuit_type,
            "structural_similarity": 0.84,
            "functional_similarity": 0.89,
            "causal_similarity": 0.91,
        }
        res = DiscoveryResultDTO(
            discovery_id=f"cross_{hash(source_model + target_model) & 0xffffffff:08x}",
            discovery_type="CrossModelAlignment",
            title=f"Circuit Alignment: {source_model} ↔ {target_model}",
            evidence=[],
            confidence=0.0,
        ).to_dict()
        res["alignment"] = alignment
        res["alignment_field_provenance"] = {
            key: "reference" for key in alignment
        }
        res["status"] = "unavailable"
        res["provenance"] = "reference"
        res["alignment_measured"] = False
        res["validation_eligible"] = False
        res["publication_eligible"] = False
        res["reason"] = (
            "Cross-model circuit alignment is not implemented. No models were "
            "loaded and no comparison was run; these similarity scores are "
            "fixed reference values, not measurements."
        )
        return res
