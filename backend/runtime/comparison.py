"""Model Comparison Engine.

Compares activations, predictions, attention patterns, residuals, and logits
across different models (e.g., GPT-2 vs Pythia vs Gemma) returning normalized
DTOs.

`compare()` previously returned a fixed dictionary for any prompt and any pair
of models: cosine_similarity 0.875, kl_divergence 0.142, top_token_match True
with " Paris" at prob 0.82/0.79, head_alignment_score 0.91, induction head
layers 8 and 7, layer_norm_ratio 1.04. None of it was computed from either
model. It reported that GPT-2 and Pythia agree, in detail, forever.

The comparison needs both models loaded, their activations extracted at matched
layers and positions, and the metrics actually reduced over those tensors.
There is no backend wired up here, so this reports that it cannot compare and
names what is missing. `compare_available()` exists so a caller can check
without paying for the refusal, and `TO_COMPUTE` records the metric set that a
real implementation must fill in rather than leaving it implicit.
"""

from __future__ import annotations

from typing import Any, Dict, List

# The metrics a real implementation must compute. Named here so the absence of
# a backend is a specific, checkable gap rather than a vague "not done".
TO_COMPUTE = (
    "activations.cosine_similarity",
    "activations.kl_divergence",
    "predictions.top_token_match",
    "predictions.model_a_top",
    "predictions.model_b_top",
    "attention.head_alignment_score",
    "attention.induction_head_layer_a",
    "attention.induction_head_layer_b",
    "residuals.layer_norm_ratio",
    "residuals.mean_absolute_difference",
)

_UNAVAILABLE = (
    "Cross-model comparison requires both models to be loaded and their "
    "activations, logits and attention patterns extracted at matched layers "
    "and token positions. No such backend is wired into this engine, so no "
    "comparison metric can be computed."
)


class ModelComparisonEngine:
    """Computes cross-model comparative metrics."""

    def __init__(self, backend: Any = None) -> None:
        # A backend may be injected by a caller that has one. Absent by default,
        # which is why compare() refuses.
        self.backend = backend

    def compare_available(self) -> bool:
        """True only when something can actually perform the comparison."""
        return self.backend is not None

    def compare(
        self,
        prompt: str,
        model_a: str = "GPT-2 Small",
        model_b: str = "Pythia 160M",
    ) -> Dict[str, Any]:
        """Compare two models on a prompt.

        Returns the honest refusal rather than a fixed dictionary of plausible
        numbers.
        """
        if self.backend is None:
            return {
                "status": "unavailable",
                "implemented": False,
                "prompt": prompt,
                "model_a": model_a,
                "model_b": model_b,
                "provenance": "unavailable",
                "validation_eligible": False,
                "publication_eligible": False,
                # Explicitly empty, so a consumer can assert "no comparison
                # happened" rather than reading absence as agreement.
                "activations": {},
                "predictions": {},
                "attention": {},
                "residuals": {},
                "metrics_measured": False,
                "metrics_required": list(TO_COMPUTE),
                "reason": _UNAVAILABLE,
            }

        # A caller-supplied backend is expected to return measured values.
        # This engine does not reshape or default them.
        return dict(self.backend.compare(prompt=prompt, model_a=model_a,
                                         model_b=model_b))
