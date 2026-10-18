"""Arithmetic Task Reproduction Pipeline.

Measures an addition circuit on loaded weights and localises it by causal
scrubbing: for each layer, the corrupted run's residual stream is replaced with
the clean run's at the answer position, and the recovery of the answer's logit
difference is measured. Two families are measured separately — base-10 addition
and modular addition — because a single "arithmetic accuracy" number would
conflate them.

What this used to be
--------------------
The module returned::

    # Simulated metrics for the mock/test environment
    return {
        "Modulo Addition Accuracy": 0.45,
        "Base-10 Addition Accuracy": 0.85,
    }

There was no mock flag and no branch, so these were returned to every caller
including callers with weights loaded, and `benchmark_runner` reached `run()`
through its signature fallback and stored the numbers in a reproducibility
report. Neither was computed.

Fail-closed, with the measurement kept
--------------------------------------
If GPT-2 small does not answer the prompts correctly, there is no circuit to
localise: a scrub effect measured on a task the model cannot do is a property of
the noise, not of the computation. This pipeline then raises `LiveUnavailable`
carrying the measured accuracies as `measured_evidence`, rather than returning an
`unavailable` envelope — `benchmark_runner` marks *any* pipeline that does not
raise as `PASS` and never reads the returned `status`, so returning here would
record an unmeasured circuit as a pass. The counts survive on the exception.

Model choice
------------
The platform's golden-benchmark entry for arithmetic declares Llama3-8B. That is
~16GB in fp16 and does not fit the hardware this was developed on, and comparing
a GPT-2-small measurement against an Llama-3-8B baseline is the cross-model
comparison error the validation scheduler already guards with
`baseline_comparable`. This pipeline therefore measures the model its adapter
actually holds, records that id in the result, and declares no baseline.
"""

from __future__ import annotations

import random
import statistics
from typing import Any, Dict, List, Optional, Tuple

from ..models.adapter_base import LiveUnavailable
from ..models.gpt2_adapter import GPT2Adapter
from .dataset_versioning import DatasetVersioningEngine
from .reproducibility_report import ReproducibilityReportEngine
from .pipeline_contract import LIVE, completed

#: Prompt templates. The answer follows ``=`` immediately, so the metric is read
#: at the final position and no free generation is involved.
_BASE10_TEMPLATE = "{a}+{b}="
_MODULAR_TEMPLATE = "{a}+{b} mod {modulus}="

#: A recovery ratio whose denominator is below this is undefined, not zero.
_MIN_CONTRAST = 1e-6


class ArithmeticPipeline:
    """Addition-circuit localisation by causal scrubbing on loaded weights."""

    PAPER_ID = "arithmetic"

    def __init__(self, model_manager: Any = None,
                 mock_mode: bool = False) -> None:
        self.model_manager = model_manager
        self.mock_mode = mock_mode
        self._adapter: Optional[GPT2Adapter] = None
        self._versioning = DatasetVersioningEngine()
        self._report_engine = ReproducibilityReportEngine()

    # ------------------------------------------------------------------ #
    # Weight availability                                                 #
    # ------------------------------------------------------------------ #

    def _adapter_or_raise(self) -> GPT2Adapter:
        """A weights-backed adapter, or a refusal.

        Needs a model manager: `benchmark_runner` constructs this class as
        ``ArithmeticPipeline(self.model_manager)``. Without one there is no route
        to weights, and loading a model anyway would report a measurement under a
        dependency the caller never supplied.

        `ModelAdapter._load_model` turns any load failure into a mock-mode
        fallback, so `simulated` is checked explicitly — otherwise a fallback
        adapter would report fixtures under a live label.
        """
        if self.mock_mode:
            raise LiveUnavailable(
                "ArithmeticPipeline was constructed in mock mode. An addition "
                "circuit is a forward-pass and hook measurement with no fixture; "
                "no weights were read and no accuracy is reported."
            )
        if self.model_manager is None:
            raise LiveUnavailable(
                "ArithmeticPipeline was constructed without a model manager, so "
                "it has no way to obtain weights. No arithmetic measurement was "
                "performed."
            )
        if self._adapter is None:
            adapter = GPT2Adapter(variant="small", mock_mode=False)
            if (adapter.spec.mock_mode or adapter._model is None
                    or getattr(adapter, "simulated", False)):
                fallback = (getattr(adapter, "simulation_reason", "")
                            or "no reason recorded")
                raise LiveUnavailable(
                    "GPT-2 weights are not loaded, so no arithmetic measurement "
                    "can be made. The adapter fell back to its mock fixtures "
                    f"({fallback})."
                )
            self._adapter = adapter
        return self._adapter

    # ------------------------------------------------------------------ #
    # Prompt pairs                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _flip_operand(b: int, low: int, span: int) -> int:
        """Move ``b`` half-way round its range, keeping it the same width."""
        return low + ((b - low + span // 2) % span)

    def _build_pairs(self, family: str, n_pairs: int,
                     rng: random.Random) -> List[Dict[str, Any]]:
        """Clean/corrupted prompt pairs with a flipped first answer token.

        The corrupted prompt keeps the operand widths (so the token count is
        unchanged) and changes only the second operand, which moves the answer.
        Every candidate is checked for equal token length and for a differing
        first answer token; a candidate that fails either is dropped by the
        caller, not coerced.
        """
        pairs: List[Dict[str, Any]] = []
        modulus = 11
        for _ in range(n_pairs * 3):
            if len(pairs) >= n_pairs:
                break
            if family == "modular":
                a = rng.randrange(1, 10)
                b = rng.randrange(1, 10)
                # 1..9 -> 2..9,1: always a different single-digit residue, so the
                # prompt keeps the same token count as well as flipping the answer.
                b2 = 1 + (b % 9)
                clean_prompt = _MODULAR_TEMPLATE.format(a=a, b=b, modulus=modulus)
                corrupt_prompt = _MODULAR_TEMPLATE.format(a=a, b=b2, modulus=modulus)
                clean_answer = (a + b) % modulus
                corrupt_answer = (a + b2) % modulus
            else:
                a = rng.randrange(10, 100)
                b = rng.randrange(10, 100)
                b2 = self._flip_operand(b, 10, 90)
                clean_prompt = _BASE10_TEMPLATE.format(a=a, b=b)
                corrupt_prompt = _BASE10_TEMPLATE.format(a=a, b=b2)
                clean_answer = a + b
                corrupt_answer = a + b2
            if clean_answer == corrupt_answer:
                continue
            pairs.append({
                "family": family,
                "a": a,
                "b": b,
                "b_corrupted": b2,
                "modulus": modulus if family == "modular" else None,
                "clean_prompt": clean_prompt,
                "corrupted_prompt": corrupt_prompt,
                "clean_answer": clean_answer,
                "corrupted_answer": corrupt_answer,
            })
        return pairs

    # ------------------------------------------------------------------ #
    # Hooked forward passes                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _unwrap(output: Any) -> Any:
        return output[0] if isinstance(output, tuple) else output

    @staticmethod
    def _rewrap(output: Any, new_hidden: Any) -> Any:
        return (new_hidden,) + tuple(output[1:]) if isinstance(output, tuple) else new_hidden

    def _forward(self, prompt: str):
        import torch

        inputs = self._adapter._tokenizer(prompt, return_tensors="pt").to(
            self._adapter._model.device)
        with torch.no_grad():
            return self._adapter._model(**inputs, use_cache=False)

    def _logits(self, prompt: str):
        return self._forward(prompt).logits[0, -1, :]

    def _capture_block(self, prompt: str, layer: int):
        """The block-`layer` output at the final position, from a real run."""
        captured: Dict[str, Any] = {}
        block = self._adapter._model.transformer.h[layer]

        def hook(_module, _inp, output):
            captured["hidden"] = self._unwrap(output)[0, -1, :].detach().clone()
            return output

        handle = block.register_forward_hook(hook)
        try:
            self._forward(prompt)
        finally:
            handle.remove()
        return captured.get("hidden")

    def _patched_logits(self, prompt: str, layer: int, vector):
        """Run `prompt`, writing `vector` into block `layer`'s final position.

        This is a single-site residual-stream intervention. It is expressed with
        a forward hook on the block rather than through the adapter's
        `patch_activation`, because that method's `neuron_index` is a *residual
        dimension*, not a position — using it here would write one scalar into
        the wrong axis and still return a plausible number.
        """
        block = self._adapter._model.transformer.h[layer]

        def hook(_module, _inp, output):
            hidden = self._unwrap(output).clone()
            hidden[0, -1, :] = vector
            return self._rewrap(output, hidden)

        handle = block.register_forward_hook(hook)
        try:
            return self._logits(prompt)
        finally:
            handle.remove()

    # ------------------------------------------------------------------ #
    # Measurement                                                         #
    # ------------------------------------------------------------------ #

    def _answer_token_id(self, answer: int) -> int:
        tokenizer = self._adapter._tokenizer
        ids = tokenizer(str(answer), add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise LiveUnavailable(
                f"answer {answer!r} is not a single token ({len(ids)}); the "
                f"first-token metric assumes one token per answer"
            )
        return ids[0]

    def _measure_family(self, family: str, pairs: List[Dict[str, Any]],
                        ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Per-pair correctness, plus the aggregate. No localisation here."""
        tokenizer = self._adapter._tokenizer
        traces: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []

        for pair in pairs:
            clean_len = len(tokenizer(pair["clean_prompt"],
                                     add_special_tokens=False)["input_ids"])
            corrupt_len = len(tokenizer(pair["corrupted_prompt"],
                                        add_special_tokens=False)["input_ids"])
            if clean_len != corrupt_len:
                skipped.append({
                    "clean_prompt": pair["clean_prompt"],
                    "reason": (
                        f"clean prompt is {clean_len} tokens and corrupted is "
                        f"{corrupt_len}; the pair does not preserve token length"),
                })
                continue
            try:
                clean_id = self._answer_token_id(pair["clean_answer"])
                corrupt_id = self._answer_token_id(pair["corrupted_answer"])
            except LiveUnavailable as exc:
                skipped.append({"clean_prompt": pair["clean_prompt"],
                                "reason": str(exc)})
                continue
            if clean_id == corrupt_id:
                skipped.append({
                    "clean_prompt": pair["clean_prompt"],
                    "reason": "the corrupted answer's first token is the same "
                              "token as the clean answer's",
                })
                continue

            logits = self._logits(pair["clean_prompt"])
            predicted = int(logits.argmax())
            target_logit = float(logits[clean_id])
            corrupt_logit = float(logits[corrupt_id])
            correct = predicted == clean_id
            entry = dict(pair)
            entry.update({
                "n_tokens": clean_len,
                "clean_answer_token_id": clean_id,
                "corrupted_answer_token_id": corrupt_id,
                "predicted_token_id": predicted,
                "target_logit": round(target_logit, 6),
                "corrupted_token_logit": round(corrupt_logit, 6),
                "logit_diff": round(target_logit - corrupt_logit, 6),
                "correct": bool(correct),
                "provenance": LIVE,
                "synthetic_fields": [],
            })
            if not correct:
                entry["predicted_text"] = tokenizer.decode([predicted])
                entry["clean_answer_text"] = str(pair["clean_answer"])
            traces.append(entry)

        usable = [t for t in traces]
        aggregate = {
            "family": family,
            "n_pairs_offered": len(pairs),
            "n_pairs_measured": len(usable),
            "n_pairs_skipped": len(skipped),
            "skipped": skipped,
            "n_correct": sum(1 for t in usable if t["correct"]),
            "accuracy": (round(sum(1 for t in usable if t["correct"]) / len(usable), 4)
                         if usable else None),
            "mean_logit_diff": (round(statistics.mean(
                [t["logit_diff"] for t in usable]), 6) if usable else None),
        }
        return traces, aggregate

    def _recovery(self, pair: Dict[str, Any], layer: int) -> Optional[float]:
        """Fraction of the clean logit difference recovered by patching a layer."""
        clean_logits = self._logits(pair["clean_prompt"])
        clean_diff = float(clean_logits[pair["clean_answer_token_id"]]
                           - clean_logits[pair["corrupted_answer_token_id"]])
        corrupt_logits = self._logits(pair["corrupted_prompt"])
        corrupt_diff = float(corrupt_logits[pair["clean_answer_token_id"]]
                             - corrupt_logits[pair["corrupted_answer_token_id"]])
        denominator = clean_diff - corrupt_diff
        if abs(denominator) < _MIN_CONTRAST:
            return None
        captured = self._capture_block(pair["clean_prompt"], layer)
        if captured is None:
            return None
        patched = self._patched_logits(pair["corrupted_prompt"], layer, captured)
        patched_diff = float(patched[pair["clean_answer_token_id"]]
                             - patched[pair["corrupted_answer_token_id"]])
        return (patched_diff - corrupt_diff) / denominator

    # ------------------------------------------------------------------ #
    # Entry point                                                         #
    # ------------------------------------------------------------------ #

    def run(self, n_pairs: int = 24, seed: int = 42, min_accuracy: float = 0.5,
            max_scrub_pairs: int = 8, model_id: str = "gpt2-small",
            ) -> Dict[str, Any]:
        """Measure addition correctness, then localise it by causal scrubbing."""
        adapter = self._adapter_or_raise()
        rng = random.Random(seed)

        families = {}
        for family in ("modular", "base10"):
            pairs = self._build_pairs(family, n_pairs, rng)
            traces, aggregate = self._measure_family(family, pairs)
            families[family] = {"pairs": pairs, "traces": traces,
                                "aggregate": aggregate}

        accuracies = {f: families[f]["aggregate"]["accuracy"]
                      for f in families}
        usable_accuracies = [a for a in accuracies.values() if a is not None]
        if not usable_accuracies or min(usable_accuracies) < min_accuracy:
            detail = ", ".join(
                f"{f}: {families[f]['aggregate']['n_correct']}/"
                f"{families[f]['aggregate']['n_pairs_measured']} correct"
                for f in sorted(families))
            exc = LiveUnavailable(
                "GPT-2 small does not answer these addition prompts correctly, so "
                "there is no addition circuit to localise. A scrub effect "
                "measured on a task the model cannot do is a property of the "
                f"noise rather than of the computation. Measured: {detail} "
                f"(min_accuracy={min_accuracy}). No layer effect is reported, and "
                f"Stolfo et al.'s claim is neither confirmed nor refuted here — "
                f"this harness does not reproduce their setup."
            )
            exc.measured_evidence = {
                "observed_metrics": {
                    "modular_addition_accuracy": accuracies.get("modular"),
                    "base10_addition_accuracy": accuracies.get("base10"),
                    "modular_pairs_measured": families["modular"]["aggregate"][
                        "n_pairs_measured"],
                    "base10_pairs_measured": families["base10"]["aggregate"][
                        "n_pairs_measured"],
                    # Present and None: no sweep ran, so there is nothing these
                    # could be an effect of.
                    "modular_layer_effects": None,
                    "base10_layer_effects": None,
                    "dominant_layer_base10": None,
                    "dominant_layer_modular": None,
                },
                "raw_traces": (families["modular"]["traces"]
                               + families["base10"]["traces"]),
                "reason": "addition accuracy below min_accuracy",
            }
            raise exc

        # Only reached when the model performs the task well enough for a circuit
        # claim to be about the computation.
        n_layers = adapter.spec.num_layers
        scrub_pairs = {f: families[f]["pairs"][:max_scrub_pairs] for f in families}
        layer_effects: Dict[str, Dict[int, List[float]]] = {}
        for family in families:
            effects: Dict[int, List[float]] = {L: [] for L in range(n_layers)}
            for layer in range(n_layers):
                for pair in scrub_pairs[family]:
                    value = self._recovery(pair, layer)
                    if value is not None:
                        effects[layer].append(value)
            layer_effects[family] = effects

        means = {f: {L: statistics.mean(v) for L, v in layer_effects[f].items() if v}
                 for f in families}
        if not any(means.values()):
            raise LiveUnavailable(
                "No layer produced a usable scrub effect: every clean/corrupted "
                f"pair carried a contrast below the {_MIN_CONTRAST:g} threshold, "
                "so each recovery ratio is undefined. No layer effect is reported."
            )
        dominant = {f: (max(means[f], key=means[f].get) if means[f] else None)
                    for f in families}

        observed_metrics: Dict[str, Any] = {
            "modular_addition_accuracy": accuracies.get("modular"),
            "base10_addition_accuracy": accuracies.get("base10"),
            "modular_pairs_measured": families["modular"]["aggregate"][
                "n_pairs_measured"],
            "base10_pairs_measured": families["base10"]["aggregate"][
                "n_pairs_measured"],
            "modular_pairs_skipped": families["modular"]["aggregate"]["n_pairs_skipped"],
            "base10_pairs_skipped": families["base10"]["aggregate"]["n_pairs_skipped"],
            "modular_main_effect_layer": dominant["modular"],
            "base10_main_effect_layer": dominant["base10"],
            "modular_layer_effects": (
                {str(L): round(means["modular"][L], 4)
                 for L in sorted(means["modular"])} or None),
            "base10_layer_effects": (
                {str(L): round(means["base10"][L], 4)
                 for L in sorted(means["base10"])} or None),
            "pairs_scrubbed_per_family": len(scrub_pairs["base10"]),
            "contrast_floor": _MIN_CONTRAST,
            "min_accuracy_required": min_accuracy,
        }

        for family in families:
            recovery = {
                str(L): (None if not layer_effects[family][L] else round(
                    statistics.mean(layer_effects[family][L]), 6))
                for L in sorted(layer_effects[family])}
            for trace in families[family]["traces"]:
                trace["layer_recovery"] = recovery

        raw_traces = families["modular"]["traces"] + families["base10"]["traces"]

        manifest = self._versioning.create_manifest(
            paper_id=self.PAPER_ID,
            pipeline_name="ArithmeticPipeline",
            model_id=adapter.spec.model_id,
            hf_repo_id=adapter.spec.hf_repo_id,
            dataset_name=("Seeded addition prompt pairs (base-10 and modular), "
                          "corrupted by flipping the second operand"),
            dataset_num_examples=len(raw_traces),
            random_seed=seed,
        )
        report = self._report_engine.generate_report(
            paper_id=self.PAPER_ID,
            pipeline_name="ArithmeticPipeline",
            model_id=adapter.spec.model_id,
            dataset_manifest_id=manifest.manifest_id,
            observed_metrics=observed_metrics,
            explanation_of_diffs=[
                "Layer effects are causal-scrub recoveries: the corrupted run's "
                "residual stream at the answer position is replaced with the "
                "clean run's, and the recovery of the answer logit difference is "
                "measured. They are not correlations.",
                f"Scrubbing used the first {len(scrub_pairs['base10'])} pairs per "
                f"family, not all of them; the accuracy figures use every "
                f"measured pair.",
                f"Measured on {adapter.spec.model_id}. The golden-benchmark entry "
                f"for this task declares Llama3-8B, which is not what was "
                f"measured, so no baseline comparison is claimed.",
            ],
        )

        return completed(
            pipeline="ArithmeticPipeline",
            paper_id=self.PAPER_ID,
            model_id=adapter.spec.model_id,
            observed_metrics=observed_metrics,
            raw_traces=raw_traces,
            provenance_note=(
                "Accuracies come from forward-pass argmax at the answer "
                "position; layer effects come from single-site residual-stream "
                "hooks. Nothing is rescaled from a paper."
            ),
            validation_eligible=False,
            ineligible_because=[
                "the prompt template and operand ranges are this harness's, not "
                "the paper's, so a match would not be a reproduction",
                f"only {len(scrub_pairs['base10'])} pairs per family were "
                "scrubbed, which is a small sample for a circuit-level claim",
            ],
            publication_eligible=False,
            model_id_requested=model_id,
            family_aggregates={f: families[f]["aggregate"] for f in families},
            dominant_layer=dominant,
            weights_fingerprint=adapter.model_fingerprint(),
            reproducibility_report=report,
            manifest_id=manifest.manifest_id,
        )
