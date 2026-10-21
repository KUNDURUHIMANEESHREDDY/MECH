"""Factual Recall Task Reproduction Pipeline.

Localises the subject→relation→object recall circuit on loaded weights by
noising the subject's residual stream layer by layer, following the causal-trace
protocol of Meng et al. 2022 (ROME). Reports a per-layer effect derived from
hooks and a per-fact boolean, never a single accuracy standing alone.

Dataset, and its provenance, stated exactly
-------------------------------------------
The default stimulus is ``MECH-FACT-12``: twelve subject/relation/object triples
**authored in this repository**. Its provenance is recorded in the result, and
what it is *not* matters more than what it is:

* It is **not** CounterFact, and no result from it may be reported as a
  CounterFact result. CounterFact is Meng et al.'s 21,919-item benchmark; a
  twelve-item in-repo list shares its protocol and nothing else.
* The protocol it follows is Meng et al.'s: a natural-language prompt that names
  the subject and the relation, a single-token target object, and a noising
  sweep over the subject's hidden state at the subject token positions.
* Factual-recall results are highly sensitive to whether the evaluation set was
  already in the pretraining corpus. These twelve facts are extremely well known
  and almost certainly are, which is why the result is marked
  ``publication_eligible=False`` regardless of how clean the layer curve looks.

A real published set can be measured instead by passing ``dataset_path`` to a
CounterFact-format JSON file (``requested_rewrite`` entries, or a flat list of
``{"prompt", "subject", "target"}``). When one is supplied, its own provenance
string replaces the in-repo one, and the fact that it was loaded from disk is
recorded.

What this used to be
--------------------
The module returned, for every caller and with no branch::

    # Simulated metrics for the mock/test environment
    return {
        "Subject Entity Recall Accuracy": 0.75,
        "Relation Attribute Attention": 0.65,
    }

Neither was computed, and "Relation Attribute Attention" named no measurable
quantity: it did not say whose attention, at which layer, relative to what
baseline, or with what normaliser. Neither number appears here.
"""

from __future__ import annotations

import json
import os
import statistics
from typing import Any, Dict, List, Optional, Tuple

from ..models.adapter_base import LiveUnavailable
from ..models.gpt2_adapter import GPT2Adapter
from .dataset_versioning import DatasetVersioningEngine
from .reproducibility_report import ReproducibilityReportEngine
from .pipeline_contract import completed

#: Twelve well-known facts. Authored here, not transcribed from a published
#: benchmark; see the module docstring. Each is (subject, relation, object).
MECH_FACT_12: Tuple[Tuple[str, str, str], ...] = (
    ("The Eiffel Tower", " is located in the city of", "Paris"),
    ("The Colosseum", " is located in the city of", "Rome"),
    ("The Great Wall", " is located in the country of", "China"),
    ("The Louvre", " is located in the city of", "Paris"),
    ("The Statue of Liberty", " is located in the city of", "New York"),
    ("The Pyramids of Giza", " are located in the country of", "Egypt"),
    ("The Amazon River", " flows through the country of", "Brazil"),
    ("The Sahara Desert", " is located in the continent of", "Africa"),
    ("Mount Everest", " is located in the country of", "Nepal"),
    ("The Taj Mahal", " is located in the city of", "Agra"),
    ("The Berlin Wall", " stood in the city of", "Berlin"),
    ("The Sydney Opera House", " is located in the city of", "Sydney"),
)

IN_REPO_PROVENANCE = (
    "MECH-FACT-12 — twelve subject/relation/object triples authored in this "
    "repository. NOT CounterFact and not a subset of it. The noising protocol "
    "follows Meng et al. 2022 (ROME); the stimulus does not."
)

#: A recovery ratio whose denominator is below this is undefined, not zero.
_MIN_CONTRAST = 1e-6


def load_counterfact_pairs(path: str) -> List[Dict[str, str]]:
    """Read a CounterFact-format JSON file into (prompt, subject, target) rows.

    Supports the published ``requested_rewrite`` shape and a flat list of
    ``{"prompt", "subject", "target"}``. Raises `LiveUnavailable` rather than
    guessing at an unrecognised shape: a mis-parsed dataset would silently
    measure something other than what the caller pointed at.
    """
    if not os.path.exists(path):
        raise LiveUnavailable(f"factual-recall dataset not found: {path}")
    try:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, ValueError) as exc:
        raise LiveUnavailable(
            f"factual-recall dataset could not be read: {path}: "
            f"{type(exc).__name__}: {exc}") from exc

    rows: List[Dict[str, str]] = []
    for record in raw if isinstance(raw, list) else [raw]:
        if not isinstance(record, dict):
            continue
        rewrites = record.get("requested_rewrite")
        if isinstance(rewrites, list):
            for item in rewrites:
                if not isinstance(item, dict):
                    continue
                subject = item.get("subject")
                prompt = item.get("prompt")
                target = (item.get("target_new") or {}).get("str")
                if subject and prompt and target:
                    rows.append({
                        "subject": str(subject),
                        "prompt": str(prompt).replace("{}", str(subject)),
                        "target": str(target),
                    })
        elif record.get("subject") and record.get("prompt") and record.get("target"):
            rows.append({
                "subject": str(record["subject"]),
                "prompt": str(record["prompt"]).replace("{}", str(record["subject"])),
                "target": str(record["target"]),
            })
    if not rows:
        raise LiveUnavailable(
            f"no usable factual-recall rows in {path}: expected 'requested_rewrite' "
            "entries or a list of {'prompt','subject','target'} objects"
        )
    return rows


class FactualRecallPipeline:
    """Subject-relation recall localised by noising the subject's hidden state."""

    PAPER_ID = "factual_recall"

    def __init__(self, model_manager: Any = None, mock_mode: bool = False,
                 dataset_path: Optional[str] = None) -> None:
        self.model_manager = model_manager
        self.mock_mode = mock_mode
        self.dataset_path = dataset_path
        self._adapter: Optional[GPT2Adapter] = None
        self._versioning = DatasetVersioningEngine()
        self._report_engine = ReproducibilityReportEngine()

    # ------------------------------------------------------------------ #
    # Weight availability                                                 #
    # ------------------------------------------------------------------ #

    def _adapter_or_raise(self) -> GPT2Adapter:
        """A weights-backed adapter, or a refusal.

        Needs a model manager: `benchmark_runner` constructs this class as
        ``FactualRecallPipeline(self.model_manager)``. Without one there is no
        route to weights, and loading a model anyway would report a measurement
        under a dependency the caller never supplied.

        `simulated` is checked explicitly because `ModelAdapter._load_model`
        converts any load failure into a mock-mode fallback — trusting
        `mock_mode=False` at construction would report fixtures as live.
        """
        if self.mock_mode:
            raise LiveUnavailable(
                "FactualRecallPipeline was constructed in mock mode. A recall "
                "circuit is a forward-pass and hook measurement with no fixture; "
                "no weights were read and no accuracy is reported."
            )
        if self.model_manager is None:
            raise LiveUnavailable(
                "FactualRecallPipeline was constructed without a model manager, "
                "so it has no way to obtain weights. No factual-recall "
                "measurement was performed."
            )
        if self._adapter is None:
            adapter = GPT2Adapter(variant="small", mock_mode=False)
            if (adapter.spec.mock_mode or adapter._model is None
                    or getattr(adapter, "simulated", False)):
                fallback = (getattr(adapter, "simulation_reason", "")
                            or "no reason recorded")
                raise LiveUnavailable(
                    "GPT-2 weights are not loaded, so no factual-recall "
                    "measurement can be made. The adapter fell back to its mock "
                    f"fixtures ({fallback})."
                )
            self._adapter = adapter
        return self._adapter

    # ------------------------------------------------------------------ #
    # Stimulus                                                            #
    # ------------------------------------------------------------------ #

    def _stimulus(self, n_prompts: int) -> Tuple[List[Dict[str, str]], str]:
        """The prompt rows to measure, and the provenance string for them."""
        if self.dataset_path:
            rows = load_counterfact_pairs(self.dataset_path)
            provenance = (
                f"CounterFact-format file loaded from {self.dataset_path} "
                f"({len(rows)} rows). Measured as supplied; this pipeline does "
                f"not verify which corpus the rows came from."
            )
        else:
            rows = [{"subject": s, "prompt": s + r, "target": o}
                    for s, r, o in MECH_FACT_12]
            provenance = IN_REPO_PROVENANCE
        return rows[:n_prompts], provenance

    def _subject_span(self, prompt: str, subject: str) -> Optional[Tuple[int, int]]:
        """Token span of the subject inside the prompt, verified by decoding.

        The span decides which positions are noised, so an off-by-one would
        patch a neighbouring token and still return a plausible layer curve.
        Verification is by re-decoding the slice and comparing it to the subject
        text; a row whose span cannot be confirmed is excluded and counted
        rather than patched at a guessed offset.
        """
        tokenizer = self._adapter._tokenizer
        subject_ids = tokenizer(subject, add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if not subject_ids or len(subject_ids) >= len(prompt_ids):
            return None
        end = len(subject_ids)
        if list(prompt_ids[:end]) != list(subject_ids):
            return None
        decoded = tokenizer.decode(prompt_ids[:end])
        if decoded.strip() != subject.strip():
            return None
        return 0, end

    def _target_token_id(self, target: str) -> int:
        tokenizer = self._adapter._tokenizer
        ids = tokenizer(" " + target, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise LiveUnavailable(
                f"object {target!r} is not a single token with a leading space "
                f"({len(ids)} tokens); the first-token metric assumes one token "
                f"per object"
            )
        return ids[0]

    # ------------------------------------------------------------------ #
    # Hooked forward passes                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _unwrap(output: Any) -> Any:
        return output[0] if isinstance(output, tuple) else output

    @staticmethod
    def _rewrap(output: Any, new_hidden: Any) -> Any:
        return (new_hidden,) + tuple(output[1:]) if isinstance(output, tuple) else new_hidden

    def _logits(self, prompt: str):
        import torch

        inputs = self._adapter._tokenizer(prompt, return_tensors="pt").to(
            self._adapter._model.device)
        with torch.no_grad():
            return self._adapter._model(
                **inputs, use_cache=False).logits[0, -1, :]

    def _capture_subject(self, prompt: str, layer: int, span: Tuple[int, int]):
        captured: Dict[str, Any] = {}
        block = self._adapter._model.transformer.h[layer]
        start, end = span

        def hook(_module, _inp, output):
            hidden = self._unwrap(output)
            captured["value"] = hidden[0, start:end, :].detach().clone()
            return output

        handle = block.register_forward_hook(hook)
        try:
            self._logits(prompt)
        finally:
            handle.remove()
        return captured.get("value")

    def _patched_logits(self, prompt: str, layer: int, span: Tuple[int, int],
                        vector):
        """Run `prompt` with the subject span's block output replaced."""
        block = self._adapter._model.transformer.h[layer]
        start, end = span

        def hook(_module, _inp, output):
            hidden = self._unwrap(output).clone()
            hidden[0, start:end, :] = vector
            return self._rewrap(output, hidden)

        handle = block.register_forward_hook(hook)
        try:
            return self._logits(prompt)
        finally:
            handle.remove()

    # ------------------------------------------------------------------ #
    # Measurement                                                         #
    # ------------------------------------------------------------------ #

    def run(self, n_prompts: int = 12, seed: int = 42, min_accuracy: float = 0.5,
            model_id: str = "gpt2-small") -> Dict[str, Any]:
        """Measure subject-object recall, then noise the subject layer by layer.

        `n_prompts` and `seed` exist because the benchmark runner probes the
        `run` signature and passes them; the stimulus is a fixed list, so
        `n_prompts` truncates it and `seed` is recorded but not used to sample.
        That is stated rather than hidden: a seeded pipeline that ignores its
        seed is worse than one that says it does.
        """
        adapter = self._adapter_or_raise()
        rows, dataset_provenance = self._stimulus(n_prompts)
        if not rows:
            raise LiveUnavailable(
                "No factual-recall rows were available to measure.")

        # A corrupted direction per row: the object of a different row. That
        # gives the metric a second candidate whose logit can be compared
        # against, without inventing a distractor.
        traces: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []
        prepared: List[Dict[str, Any]] = []

        for index, row in enumerate(rows):
            other = rows[(index + 1) % len(rows)]
            if other is row and len(rows) > 1:
                other = rows[(index + 2) % len(rows)]
            try:
                span = self._subject_span(row["prompt"], row["subject"])
                if span is None:
                    skipped.append({
                        "prompt": row["prompt"],
                        "reason": (
                            "the subject's token span could not be verified "
                            "inside the prompt, so the noising sites are "
                            "unknown for this row"),
                    })
                    continue
                target_id = self._target_token_id(row["target"])
                distractor_id = self._target_token_id(other["target"])
            except LiveUnavailable as exc:
                skipped.append({"prompt": row["prompt"], "reason": str(exc)})
                continue
            if target_id == distractor_id:
                skipped.append({
                    "prompt": row["prompt"],
                    "reason": "the distractor object shares the target's first token",
                })
                continue

            import torch

            inputs = adapter._tokenizer(row["prompt"], return_tensors="pt").to(
                adapter._model.device)
            with torch.no_grad():
                logits = adapter._model(
                    **inputs, use_cache=False).logits[0, -1, :]
            predicted = int(logits.argmax())
            correct = predicted == target_id
            entry = {
                "prompt": row["prompt"],
                "subject": row["subject"],
                "target": row["target"],
                "distractor": other["target"],
                "subject_span": list(span),
                "target_token_id": target_id,
                "distractor_token_id": distractor_id,
                "predicted_token_id": predicted,
                "predicted_text": adapter._tokenizer.decode([predicted]),
                "target_logit": round(float(logits[target_id]), 6),
                "distractor_logit": round(float(logits[distractor_id]), 6),
                "logit_diff": round(float(logits[target_id] - logits[distractor_id]), 6),
                "correct": bool(correct),
                "synthetic_fields": [],
            }
            traces.append(entry)
            prepared.append(dict(row, span=span, target_id=target_id,
                                 distractor_id=distractor_id))

        if not traces:
            raise LiveUnavailable(
                "No factual-recall row could be measured: every row failed the "
                "single-token or subject-span check. Reasons: "
                + "; ".join(s["reason"] for s in skipped[:5])
            )

        n_correct = sum(1 for t in traces if t["correct"])
        accuracy = round(n_correct / len(traces), 4)
        if accuracy < min_accuracy:
            exc = LiveUnavailable(
                "GPT-2 small does not recall these subject-object pairs well "
                "enough for the noising sweep to be about the recall circuit. "
                f"Measured: {n_correct}/{len(traces)} rows whose next-token "
                f"argmax is the target object (accuracy {accuracy}, "
                f"min_accuracy={min_accuracy}). A per-layer effect measured on "
                f"prompts the model does not answer is a property of the noise. "
                f"Meng et al.'s claim is neither confirmed nor refuted here."
            )
            exc.measured_evidence = {
                "observed_metrics": {
                    "subject_entity_recall_accuracy": accuracy,
                    "n_prompts_measured": len(traces),
                    "n_prompts_correct": n_correct,
                    # Present and None: no sweep ran.
                    "layer_effects": None,
                    "peak_effect_layer": None,
                    "peak_effect": None,
                },
                "raw_traces": traces,
                "reason": "recall accuracy below min_accuracy",
            }
            raise exc

        # --- subject noising sweep -------------------------------------
        n_layers = adapter.spec.num_layers
        effects: Dict[int, List[float]] = {L: [] for L in range(n_layers)}
        for layer in range(n_layers):
            for row in prepared:
                clean_logits = self._logits(row["prompt"])
                clean_diff = float(clean_logits[row["target_id"]]
                                   - clean_logits[row["distractor_id"]])
                captured = self._capture_subject(
                    row["prompt"], layer, row["span"])
                if captured is None:
                    continue
                patched = self._patched_logits(
                    row["prompt"], layer, row["span"], -captured)
                patched_diff = float(patched[row["target_id"]]
                                     - patched[row["distractor_id"]])
                # Denominator: how much of the target-vs-distractor gap the
                # subject's own hidden state carries at this prompt.
                denominator = clean_diff
                if abs(denominator) < _MIN_CONTRAST:
                    continue
                effects[layer].append((clean_diff - patched_diff) / denominator)

        means = {L: statistics.mean(v) for L, v in effects.items() if v}
        if not means:
            raise LiveUnavailable(
                "No layer produced a usable subject-noising effect: every row's "
                f"target-vs-distractor gap was below the {_MIN_CONTRAST:g} "
                "threshold, so each ratio is undefined. No layer effect is "
                "reported."
            )
        peak_layer = max(means, key=means.get)
        unmeasured_layers = sorted(set(effects) - set(means))

        for trace in traces:
            trace["layer_effect"] = {
                str(L): (None if not effects[L] else round(
                    statistics.mean(effects[L]), 6))
                for L in sorted(effects)
            }

        observed_metrics = {
            "subject_entity_recall_accuracy": accuracy,
            "n_prompts_measured": len(traces),
            "n_prompts_correct": n_correct,
            "n_prompts_skipped": len(skipped),
            "layer_effects": {str(L): round(means[L], 4) for L in sorted(means)},
            "peak_effect_layer": peak_layer,
            "peak_effect": round(means[peak_layer], 4),
            "n_layers_measured": len(means),
            "n_layers_unmeasured": len(unmeasured_layers),
            "layer_effect_definition": (
                "(clean target-vs-distractor logit gap - gap after replacing the "
                "subject span's block output with its negation) / clean gap; one "
                "value per layer, averaged over rows"
            ),
            "intervention": (
                "subject-span hidden state at block `layer` negated in place; a "
                "negation rather than a swap so no second prompt's content enters "
                "the measurement"
            ),
            "prompting_protocol": (
                "prompt = subject + relation, e.g. 'The Eiffel Tower is located in "
                "the city of'; the metric is the argmax at the final position "
                "against the object's leading-space token"
            ),
            "contrast_floor": _MIN_CONTRAST,
            "min_accuracy_required": min_accuracy,
        }

        manifest = self._versioning.create_manifest(
            paper_id=self.PAPER_ID,
            pipeline_name="FactualRecallPipeline",
            model_id=adapter.spec.model_id,
            hf_repo_id=adapter.spec.hf_repo_id,
            dataset_name=("CounterFact-format file"
                          if self.dataset_path else "MECH-FACT-12 (in-repo)"),
            dataset_num_examples=len(traces),
            random_seed=seed,
        )
        report = self._report_engine.generate_report(
            paper_id=self.PAPER_ID,
            pipeline_name="FactualRecallPipeline",
            model_id=adapter.spec.model_id,
            dataset_manifest_id=manifest.manifest_id,
            observed_metrics=observed_metrics,
            explanation_of_diffs=[
                dataset_provenance,
                "Layer effects are subject-noising causal traces: the subject "
                "span's block output is negated and the target-vs-distractor "
                "gap is re-measured. This is an intervention, not a correlation.",
                f"Measured on {adapter.spec.model_id}. The golden-benchmark entry "
                f"for this task declares a different model, so no baseline "
                f"comparison is claimed.",
                "These facts are extremely well known and were very likely in the "
                "pretraining corpus, so a clean layer curve here is not evidence "
                "of generalisation.",
            ],
        )

        return completed(
            pipeline="FactualRecallPipeline",
            paper_id=self.PAPER_ID,
            model_id=adapter.spec.model_id,
            observed_metrics=observed_metrics,
            raw_traces=traces,
            provenance_note=(
                "Recall accuracies come from forward-pass argmax; layer effects "
                "come from subject-span block hooks. Nothing is rescaled from a "
                "paper."
            ),
            validation_eligible=False,
            ineligible_because=[
                "the stimulus is not a published benchmark; see dataset_provenance",
                "these facts are almost certainly in the pretraining corpus, so "
                "recall here cannot be separated from memorisation",
            ],
            publication_eligible=False,
            model_id_requested=model_id,
            dataset_provenance=dataset_provenance,
            dataset_name=("CounterFact-format file"
                          if self.dataset_path else "MECH-FACT-12"),
            n_prompts_skipped=len(skipped),
            skipped_rows=skipped,
            weights_fingerprint=adapter.model_fingerprint(),
            reproducibility_report=report,
            manifest_id=manifest.manifest_id,
        )
