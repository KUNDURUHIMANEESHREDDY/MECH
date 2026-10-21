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

Why it raises rather than returning an `unavailable` envelope
------------------------------------------------------------
`benchmark_runner` sets `status = "PASS"` for *any* pipeline that does not raise,
and never inspects the returned `status` field. Returning an unavailable
envelope from here would therefore record an unmeasured circuit as a PASS. The
measured counts are attached to the raised exception as `measured_evidence`, so
the finding ("0/10 prompts above chance, mean -0.81") survives as data instead
of only as a sentence in the reason.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Optional, Tuple

from ..models.adapter_base import LiveUnavailable
from ..models.gpt2_adapter import GPT2Adapter
from .dataset_versioning import DatasetVersioningEngine
from .reproducibility_report import ReproducibilityReportEngine
from .pipeline_contract import completed

# Start years for the comparison. Each expands to a prompt whose final two
# digits partition the 100 two-digit completions into valid and invalid halves.
_START_YEARS: Tuple[int, ...] = (
    1939, 1942, 1945, 1953, 1955, 1975, 1980, 1988, 1990, 1991,
)

_PROMPT_TEMPLATE = "The event lasted from {start} to {prefix}"

#: Below this |contrast| a layer's recovery ratio is undefined. Dividing by a
#: vanishing contrast turns numerical noise into a large effect, and the old
#: code returned 0.0 here -- the *opposite* claim ("measured, and the effect is
#: zero"), indistinguishable to a consumer from a real zero. Such a layer is now
#: reported as not measured and excluded from the mean.
_MIN_CONTRAST = 1e-6


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
        #: start_year -> last-position logits, so the comparison score and the
        #: per-prompt trace come from the same forward pass rather than from two
        #: runs that happen to agree.
        self._logits_cache: Dict[int, Any] = {}

    # ------------------------------------------------------------------ #
    # Weight availability                                                 #
    # ------------------------------------------------------------------ #

    def _require_weights(self) -> None:
        """Raise unless this adapter is backed by real, non-simulated weights.

        `ModelAdapter.__init__` swallows every load failure into a mock-mode
        fallback, so `mock_mode=False` at construction is not evidence that
        weights loaded. Checking `simulated` is what makes the refusal true: a
        fallback adapter would otherwise report mock fixtures under a live
        label, which is the failure mode `tests/pytest/_weight_guard.py`
        documents for the test suite.
        """
        if self.adapter.spec.mock_mode or self.adapter._model is None:
            raise LiveUnavailable(
                "GreaterThanCircuitPipeline requires loaded GPT-2 weights; "
                "there is no fixture for a circuit measurement."
            )
        if getattr(self.adapter, "simulated", False):
            raise LiveUnavailable(
                "GPT-2 weights are not loaded: the adapter fell back to its "
                "mock fixtures ("
                f"{self.adapter.simulation_reason or 'no reason recorded'}). "
                "No greater-than measurement was performed."
            )

    # ------------------------------------------------------------------ #
    # The comparison metric                                               #
    # ------------------------------------------------------------------ #

    def _prompt(self, start_year: int) -> str:
        return _PROMPT_TEMPLATE.format(start=str(start_year), prefix=str(start_year)[:2])

    def _tokenizer_inputs(self, prompt: str):
        return self.adapter._tokenizer(prompt, return_tensors="pt").to(
            self.adapter._model.device
        )

    def _last_position_logits(self, start_year: int):
        """Cached last-position logits for a start year's prompt."""
        if start_year not in self._logits_cache:
            import torch

            inputs = self._tokenizer_inputs(self._prompt(start_year))
            with torch.no_grad():
                self._logits_cache[start_year] = self.adapter._model(
                    **inputs).logits[0, -1, :]
        return self._logits_cache[start_year]

    @staticmethod
    def _group_diff_from_logits(logits, two_digit: Dict[int, int],
                                start_year: int) -> Dict[str, float]:
        """logit mass(valid) - logit mass(invalid), with both masses.

        Each group is normalised by its own size so the score does not drift
        with how many completions happen to fall on each side.
        """
        import torch

        last_two = int(str(start_year)[2:])
        valid = [two_digit[n] for n in range(100) if n < last_two]
        invalid = [two_digit[n] for n in range(100) if n >= last_two]
        log_valid = torch.logsumexp(logits[valid], 0) - math.log(len(valid))
        log_invalid = torch.logsumexp(logits[invalid], 0) - math.log(len(invalid))
        return {
            "log_valid_mass": float(log_valid),
            "log_invalid_mass": float(log_invalid),
            "logit_diff": float(log_valid - log_invalid),
            "n_valid": len(valid),
            "n_invalid": len(invalid),
        }

    def _group_diff(self, start_year: int, two_digit: Dict[int, int]) -> float:
        return self._group_diff_from_logits(
            self._last_position_logits(start_year), two_digit, start_year
        )["logit_diff"]

    # ------------------------------------------------------------------ #
    # MLP activation patching                                             #
    # ------------------------------------------------------------------ #

    def _layer_recovery(self, start_year: int, layer: int,
                        two_digit: Dict[int, int]) -> Optional[float]:
        """Fraction of the comparison recovered by patching one layer's MLP.

        Pairs each prompt with its own mirror (start and end digits swapped), so
        the donor and target prompts differ in which side of the comparison they
        imply while sharing a token length and a two-digit prefix.

        Returns ``None`` — not ``0.0`` — when the mirror pair carries no
        contrast: the ratio is undefined, and reporting that as a measured zero
        effect is the substitution this module exists to prevent.
        """
        start = str(start_year)
        clean = self._prompt(start_year)
        mirrored_start = int(str(start_year)[2:] + start[:2])  # 1942 -> 4219
        target = self._prompt(mirrored_start)

        donor_diff = self._group_diff(start_year, two_digit)
        target_diff = self._group_diff(mirrored_start, two_digit)
        denominator = donor_diff - target_diff
        if abs(denominator) < _MIN_CONTRAST:
            return None

        result = self.adapter.mlp_patch_logits(clean, target, layer)
        patched = self._group_diff_from_logits(
            result["patched_logits"], two_digit, start_year)["logit_diff"]
        return (patched - target_diff) / denominator

    # ------------------------------------------------------------------ #
    # Entry point                                                       #
    # ------------------------------------------------------------------ #

    def measure(self, seed: int = 42) -> Dict[str, Any]:
        """Per-prompt comparison evidence, without any localisation.

        Public so the measurement is reachable without the (much more
        expensive) layer sweep, and so a caller can obtain the numbers that
        `run` refuses to turn into a patch effect when the model does not
        perform the task.
        """
        self._require_weights()
        two_digit = _two_digit_token_ids(self.adapter)

        traces: List[Dict[str, Any]] = []
        diffs: List[float] = []
        for start_year in _START_YEARS:
            stats = self._group_diff_from_logits(
                self._last_position_logits(start_year), two_digit, start_year)
            diffs.append(stats["logit_diff"])
            traces.append({
                "start_year": start_year,
                "prompt": self._prompt(start_year),
                "completion_prefix": str(start_year)[:2],
                "n_valid_completions": stats["n_valid"],
                "n_invalid_completions": stats["n_invalid"],
                "log_valid_mass": round(stats["log_valid_mass"], 6),
                "log_invalid_mass": round(stats["log_invalid_mass"], 6),
                "logit_diff": round(stats["logit_diff"], 6),
                # The per-prompt boolean the benchmark runner uses to build its
                # interval. "Above chance" means the valid completions carry
                # more mass than the invalid ones.
                "above_chance": bool(stats["logit_diff"] > 0),
                "synthetic_fields": [],
            })

        performed = [t["above_chance"] for t in traces]
        return {
            "n_prompts": len(_START_YEARS),
            "above_chance": sum(performed),
            "accuracy": round(sum(performed) / len(performed), 4),
            "mean_logit_diff": round(statistics.mean(diffs), 4),
            "median_logit_diff": round(statistics.median(diffs), 4),
            "raw_traces": traces,
        }

    def run(self, seed: int = 42, min_accuracy: float = 0.5) -> Dict[str, Any]:
        """Measure the greater-than comparison and, if performed, localise it.

        Raises `LiveUnavailable` when there are no weights, or when the model
        does not perform the comparison -- with the measured evidence attached
        to the exception, because "the task is not being done" is a finding, not
        an error.
        """
        measurement = self.measure(seed)
        two_digit = _two_digit_token_ids(self.adapter)

        if measurement["accuracy"] < min_accuracy:
            exc = LiveUnavailable(
                "GPT-2 small does not perform the greater-than comparison on this "
                f"template, so there is no circuit to localise. Measured: "
                f"{measurement['above_chance']}/{measurement['n_prompts']} prompts "
                f"above chance, mean valid-minus-invalid logit difference "
                f"{measurement['mean_logit_diff']:+.4f} "
                f"(a systematically negative value means the model prefers the "
                f"invalid completions). Refusing to report a patch effect for a "
                f"task the model does not do. Hanna et al.'s claim is therefore "
                f"neither confirmed nor refuted here — this harness does not "
                f"reproduce their setup."
            )
            # Attached rather than flattened into the message: the counts and
            # the per-prompt booleans are the measurement, and a reader should be
            # able to check them without parsing prose. `patch_effect_magnitude`
            # and friends are present and None, because no layer sweep was run:
            # there is no comparison for a patch effect to be an effect of.
            exc.measured_evidence = {
                "observed_metrics": {
                    "comparison_accuracy": measurement["accuracy"],
                    "n_prompts": measurement["n_prompts"],
                    "n_above_chance": measurement["above_chance"],
                    "mean_logit_diff": measurement["mean_logit_diff"],
                    "median_logit_diff": measurement["median_logit_diff"],
                    "patch_effect_magnitude": None,
                    "mlp_importance_score": None,
                    "dominant_layer": None,
                },
                "raw_traces": measurement["raw_traces"],
                "reason": "comparison_accuracy below min_accuracy",
            }
            raise exc

        # Per-prompt recoveries and per-layer means are accumulated together so
        # the trace map and the layer means cannot disagree about which layers
        # were measurable.
        n_layers = self.adapter.spec.num_layers
        recovery: Dict[int, Dict[str, Optional[float]]] = {
            y: {} for y in _START_YEARS}
        layer_values: Dict[int, List[float]] = {L: [] for L in range(n_layers)}
        for layer in range(n_layers):
            for start_year in _START_YEARS:
                value = self._layer_recovery(start_year, layer, two_digit)
                recovery[start_year][str(layer)] = (
                    None if value is None else round(value, 6))
                if value is not None:
                    layer_values[layer].append(value)

        means = {L: statistics.mean(v) for L, v in layer_values.items() if v}
        unmeasured_layers = sorted(set(layer_values) - set(means))
        if not means:
            raise LiveUnavailable(
                "No layer produced a usable patch effect: every mirror prompt "
                f"pair carried a contrast below the {_MIN_CONTRAST:g} threshold, "
                "so each recovery ratio is undefined. No patch effect is "
                "reported."
            )
        dominant = max(means, key=means.get)

        for trace in measurement["raw_traces"]:
            trace["layer_recovery"] = recovery[trace["start_year"]]

        observed_metrics = {
            "comparison_accuracy": measurement["accuracy"],
            "mean_logit_diff": measurement["mean_logit_diff"],
            "median_logit_diff": measurement["median_logit_diff"],
            "patch_effect_magnitude": round(statistics.mean(means.values()), 4),
            # Derived from the measurement, never asserted. If the dominant
            # layer is not where the paper places it, that disagreement is the
            # result and is reported as such.
            "mlp_importance_score": round(means[dominant], 4),
            "dominant_layer": dominant,
            "n_prompts": measurement["n_prompts"],
            "n_above_chance": measurement["above_chance"],
            "n_layers_measured": len(means),
            "n_layers_unmeasured": len(unmeasured_layers),
            "unmeasured_layers": unmeasured_layers,
            "unmeasured_layer_reason": (
                "That layer's mirror pair carried a contrast below "
                f"{_MIN_CONTRAST:g}, so its recovery ratio is undefined. "
                "Reported as not measured rather than as an effect of 0.0."
            ),
            "min_accuracy_required": min_accuracy,
            "contrast_floor": _MIN_CONTRAST,
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

        return completed(
            pipeline="GreaterThanCircuitPipeline",
            paper_id=self.PAPER_ID,
            model_id=self.adapter.spec.model_id,
            observed_metrics=observed_metrics,
            raw_traces=measurement["raw_traces"],
            provenance_note=(
                "Every value was measured by a forward pass or an MLP patch on "
                "the loaded weights; nothing here is rescaled from a paper."
            ),
            # Ten prompts is a small stimulus for a circuit-level claim, and the
            # template is this repository's own rather than Hanna et al.'s. Real
            # measurement, but not a reproduction.
            validation_eligible=False,
            ineligible_because=[
                f"only {measurement['n_prompts']} prompts; a small stimulus for a "
                "circuit-level claim",
                "the prompt template is this harness's, not the paper's, so a "
                "match would not be a reproduction of their setup",
            ],
            layer_patch_effects={str(L): round(means[L], 4) for L in sorted(means)},
            measurement={k: v for k, v in measurement.items() if k != "raw_traces"},
            weights_fingerprint=self.adapter.model_fingerprint(),
            reproducibility_report=report,
            manifest_id=manifest.manifest_id,
        )
