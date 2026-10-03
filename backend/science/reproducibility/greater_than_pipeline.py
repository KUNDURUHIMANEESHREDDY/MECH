"""Greater-Than Circuit Reproduction Pipeline.

Reproduces Hanna et al. 2023 — "How does GPT-2 compute greater-than?" — using
real forward passes on loaded GPT-2 weights and MLP activation patching to
localise the computation layer by layer.

Measurement, not free generation
--------------------------------
The model is never asked to generate an end year. Instead each prompt supplies a
start year and the first two digits of an end year, and the score is the logit
mass the model assigns to the *valid* completions (end < start) minus the
*invalid* ones (end >= start):

    "The event lasted from 1942 to 19"
      valid   -> " 00" .. " 41"   (1900-1941 all precede 1942)
      invalid -> " 42" .. " 99"

This is deliberately world-knowledge-free. Asking the model to *produce* a
plausible end year instead measures what it knows about history, not whether it
can compare two numbers: on the earlier free-generation framing GPT-2 small
scored 0/9, predicting " 44", " April" and "," where the answer was " 18", and a
patching ratio computed against those negatives came out above 1.0 at every one
of the 12 layers, which is the signature of a broken denominator rather than a
result.

Fail-closed
-----------
The previous version of this file reported three "observed metrics" of which two
were fabricated. `_compute_patch_effect` ignored its `prompt` argument entirely
and returned a hardcoded table::

    key_layers = {7: 0.82, 8: 0.91, 9: 0.78}

so `mlp_importance_score` was a structural constant (always 0.91, layer 8) and
`patch_effect_magnitude` was a mean of the paper's own answer. The measurement
was incapable of disagreeing with the paper it claimed to reproduce. The third
metric, `circuit_accuracy`, was also wrong for an independent reason: it tested
whether the century string appeared in a single next-token prediction, which
counts (1942, 1918) as incorrect because ``range(1942, 1919)`` is empty.

If the model does not perform the comparison, this pipeline raises
`LiveUnavailable` carrying the measured evidence. It does not return a fidelity
score. A benchmark that cannot run has no number, and the scheduler already
turns a raise into NOT_RUN with the reason attached.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Tuple

from ..models.adapter_base import LiveUnavailable
from ..models.gpt2_adapter import GPT2Adapter
from .dataset_versioning import DatasetVersioningEngine
from .reproducibility_report import ReproducibilityReportEngine

# Start years for the comparison. Each expands to a prompt whose final two
# digits partition the 100 two-digit completions into valid and invalid halves.
_START_YEARS: Tuple[int, ...] = (
    1939, 1942, 1945, 1953, 1955, 1975, 1980, 1988, 1990, 1991,
)

_PROMPT_TEMPLATE = "The event lasted from {start} to {prefix}"


def _two_digit_token_ids(adapter: GPT2Adapter) -> Dict[int, int]:
    """Map 0-99 to the token id of " NN".

    Raises if any two-digit string is not a single token: the partition below
    assumes one candidate per id, and silently falling back to multi-token
    decoding would mis-assign every logit.
    """
    ids: Dict[int, int] = {}
    for n in range(100):
        encoded = adapter._tokenizer(f" {n:02d}")["input_ids"]
        if len(encoded) != 1:
            raise LiveUnavailable(
                f"two-digit completion ' {n:02d}' is not a single token "
                f"({len(encoded)} tokens); the valid/invalid partition assumes "
                f"one candidate per id"
            )
        ids[n] = encoded[0]
    return ids


class GreaterThanCircuitPipeline:
    """Greater-than circuit analysis, measured against loaded GPT-2 weights."""

    PAPER_ID = "greater_than"

    def __init__(self, mock_mode: bool = False) -> None:
        # Default is False, matching GPT2Adapter. The previous default of True
        # meant constructing this pipeline without arguments silently produced
        # mock output for a task whose entire purpose is real measurement.
        self.adapter = GPT2Adapter(variant="small", mock_mode=mock_mode)
        self._versioning = DatasetVersioningEngine()
        self._report_engine = ReproducibilityReportEngine()

    # ------------------------------------------------------------------ #
    # The comparison metric                                             #
    # ------------------------------------------------------------------ #

    def _prompt(self, start_year: int) -> str:
        return _PROMPT_TEMPLATE.format(start=str(start_year), prefix=str(start_year)[:2])

    def _group_diff(self, start_year: int, two_digit: Dict[int, int]) -> float:
        """logit mass(valid completions) - logit mass(invalid completions).

        Each group is normalised by its own size so the score does not drift with
        how many completions happen to fall on each side.
        """
        import torch

        prompt = self._prompt(start_year)
        inputs = self._tokenizer_inputs(prompt)
        last_two = int(str(start_year)[2:])
        valid = [two_digit[n] for n in range(100) if n < last_two]
        invalid = [two_digit[n] for n in range(100) if n >= last_two]

        with torch.no_grad():
            logits = self.adapter._model(**inputs).logits[0, -1, :]
        log_valid = torch.logsumexp(logits[valid], 0) - math.log(len(valid))
        log_invalid = torch.logsumexp(logits[invalid], 0) - math.log(len(invalid))
        return float(log_valid - log_invalid)

    def _tokenizer_inputs(self, prompt: str):
        return self.adapter._tokenizer(prompt, return_tensors="pt").to(
            self.adapter._model.device
        )

    # ------------------------------------------------------------------ #
    # MLP activation patching                                           #
    # ------------------------------------------------------------------ #

    def _layer_recovery(self, start_year: int, layer: int, two_digit: Dict[int, int]) -> float:
        """Fraction of the comparison recovered by patching one layer's MLP.

        Pairs each prompt with its own mirror (start and end digits swapped), so
        the donor and target prompts differ in which side of the comparison they
        imply while sharing a token length and a two-digit prefix.
        """
        start = str(start_year)
        clean = self._prompt(start_year)
        mirrored_start = int(str(start_year)[2:] + start[:2])  # 1942 -> 4219
        target = self._prompt(mirrored_start)

        donor_diff = self._group_diff(start_year, two_digit)
        target_diff = self._group_diff(mirrored_start, two_digit)
        denominator = donor_diff - target_diff
        if abs(denominator) < 1e-6:
            # A near-zero denominator makes the ratio explode; that is a prompt
            # carrying no contrast, not a large effect.
            return 0.0

        result = self.adapter.mlp_patch_logits(clean, target, layer)
        patched = self._group_diff_from_logits(result["patched_logits"], two_digit, start_year)
        return (patched - target_diff) / denominator

    def _group_diff_from_logits(self, logits, two_digit: Dict[int, int], start_year: int) -> float:
        import torch

        last_two = int(str(start_year)[2:])
        valid = [two_digit[n] for n in range(100) if n < last_two]
        invalid = [two_digit[n] for n in range(100) if n >= last_two]
        log_valid = torch.logsumexp(logits[valid], 0) - math.log(len(valid))
        log_invalid = torch.logsumexp(logits[invalid], 0) - math.log(len(invalid))
        return float(log_valid - log_invalid)

    # ------------------------------------------------------------------ #
    # Entry point                                                       #
    # ------------------------------------------------------------------ #

    def run(self, seed: int = 42, min_accuracy: float = 0.5) -> Dict[str, Any]:
        """Measure the greater-than comparison and, if performed, localise it.

        Raises `LiveUnavailable` when there are no weights, or when the model
        does not perform the comparison -- with the measured evidence attached,
        because "the task is not being done" is a finding, not an error.
        """
        if self.adapter.spec.mock_mode or self.adapter._model is None:
            raise LiveUnavailable(
                "GreaterThanCircuitPipeline requires loaded GPT-2 weights; "
                "there is no fixture for a circuit measurement."
            )

        two_digit = _two_digit_token_ids(self.adapter)

        diffs = [self._group_diff(y, two_digit) for y in _START_YEARS]
        performed = [d > 0 for d in diffs]
        accuracy = sum(performed) / len(performed)

        measurement = {
            "n_prompts": len(_START_YEARS),
            "above_chance": sum(performed),
            "accuracy": round(accuracy, 4),
            "mean_logit_diff": round(statistics.mean(diffs), 4),
            "median_logit_diff": round(statistics.median(diffs), 4),
        }

        if accuracy < min_accuracy:
            raise LiveUnavailable(
                "GPT-2 small does not perform the greater-than comparison on this "
                f"template, so there is no circuit to localise. Measured: "
                f"{sum(performed)}/{len(performed)} prompts above chance, mean "
                f"valid-minus-invalid logit difference {statistics.mean(diffs):+.4f} "
                f"(a systematically negative value means the model prefers the "
                f"invalid completions). Refusing to report a patch effect for a "
                f"task the model does not do. Hanna et al.'s claim is therefore "
                f"neither confirmed nor refuted here — this harness does not "
                f"reproduce their setup."
            )

        n_layers = self.adapter.spec.n_layers
        layer_effects: Dict[int, List[float]] = {}
        for layer in range(n_layers):
            layer_effects[layer] = [
                self._layer_recovery(y, layer, two_digit) for y in _START_YEARS
            ]
        means = {L: statistics.mean(v) for L, v in layer_effects.items()}
        dominant = max(means, key=means.get)

        observed_metrics = {
            "comparison_accuracy": measurement["accuracy"],
            "mean_logit_diff": measurement["mean_logit_diff"],
            "patch_effect_magnitude": round(statistics.mean(means.values()), 4),
            # Derived from the measurement, never asserted. If the dominant
            # layer is not where the paper places it, that disagreement is the
            # result and is reported as such.
            "mlp_importance_score": round(means[dominant], 4),
            "dominant_layer": dominant,
        }

        manifest = self._versioning.create_manifest(
            paper_id=self.PAPER_ID,
            pipeline_name="GreaterThanCircuitPipeline",
            model_id=self.adapter.spec.model_id,
            hf_repo_id=self.adapter.spec.hf_repo_id,
            dataset_name="Greater-Than start years",
            dataset_num_examples=len(_START_YEARS),
            random_seed=seed,
        )
        report = self._report_engine.generate_report(
            paper_id=self.PAPER_ID,
            pipeline_name="GreaterThanCircuitPipeline",
            model_id=self.adapter.spec.model_id,
            dataset_manifest_id=manifest.manifest_id,
            observed_metrics=observed_metrics,
            explanation_of_diffs=[
                f"Measured dominant MLP layer is {dominant}. Hanna et al. report "
                f"greater-than computation concentrated in the mid MLP layers; "
                f"agreement or disagreement here reflects this harness's template, "
                f"not a verified reproduction of their setup."
            ],
        )

        return {
            "pipeline": "GreaterThanCircuitPipeline",
            "paper_id": self.PAPER_ID,
            "provenance": "live",
            "measurement": measurement,
            "layer_patch_effects": {str(L): round(means[L], 4) for L in sorted(means)},
            "observed_metrics": observed_metrics,
            "reproducibility_report": report,
            "manifest_id": manifest.manifest_id,
        }
