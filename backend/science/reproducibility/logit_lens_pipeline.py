"""Logit Lens Reproduction Pipeline.

Reproduces nostalgebraist 2020 — "Interpreting GPT: the logit lens" — by pushing
each intermediate residual-stream state through the model's own final layer norm
and unembedding matrix, then reading off what token the model would predict if it
stopped at that layer.

What was fabricated before
--------------------------
The previous implementation looked like a real measurement and was not. Three of
its quantities were constructed rather than computed:

* ``top_token`` per layer was ``expected if progress > 0.60 else " the"`` — the
  correct answer hardcoded in for every layer past 60% depth. The layer sweep
  therefore converged on the right answer by construction, which is the single
  thing the logit lens is used to test.
* ``_layer_entropy`` returned ``3.5 * exp(-2.0 * layer / n)``, a hand-written
  exponential with no relationship to any distribution.
* ``_convergence_layer`` returned ``int(n_layers * 0.65)`` — always 7 for GPT-2,
  for every prompt, regardless of the model.

Only ``final_layer_top1_accuracy`` and the residual norm were measured, and the
lens sweep they were embedded in was the fabricated part.

What is measured now
--------------------
``GPT2Adapter.logit_lens`` performs a real forward pass, applies ``ln_f`` and the
unembedding matrix to every intermediate state, and takes the last entry from the
model's own ``logits`` so the final row is an exact check on the rows above it.
``run`` then derives the convergence layer as the first layer whose argmax equals
the final argmax, which is a property of the measurement rather than a constant.

The result on GPT-2 small is negative and is reported as such: for
"The capital of France is" the lens moves from " destro" through a long run of
" now" to " France" at layers 10-11 and never proposes " Paris" at any layer.
Entropy is also not monotonically decreasing as the fabricated version reported —
it rises again over the last three layers. Neither is corrected to look tidier.

The dataset manifest names the prompts actually used ("Logit Lens Eval Prompts")
rather than a generic label.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List

from ..models.adapter_base import LiveUnavailable
from ..models.gpt2_adapter import GPT2Adapter
from .dataset_versioning import DatasetVersioningEngine
from .reproducibility_report import ReproducibilityReportEngine

_EVAL_PROMPTS = [
    {"prompt": "The Eiffel Tower is located in", "expected": "Paris"},
    {"prompt": "The capital of France is", "expected": "Paris"},
    {"prompt": "Water boils at 100 degrees", "expected": "Celsius"},
    {"prompt": "The speed of light is approximately", "expected": "300"},
    {"prompt": "Shakespeare wrote", "expected": "Hamlet"},
    {"prompt": "The largest planet in our solar system is", "expected": "Jupiter"},
    {"prompt": "Albert Einstein developed the theory of", "expected": "relativity"},
    {"prompt": "The human body has", "expected": "bones"},
]


class LogitLensPipeline:
    """Layer-by-layer logit lens, measured against loaded GPT-2 weights."""

    PAPER_ID = "logit_lens"

    def __init__(self, mock_mode: bool = False) -> None:
        # Default False, matching GPT2Adapter. The previous default of True
        # combined with a sweep that ignored it, so the flag controlled nothing.
        self.adapter = GPT2Adapter(variant="small", mock_mode=mock_mode)
        self._versioning = DatasetVersioningEngine()
        self._report_engine = ReproducibilityReportEngine()

    def run(self) -> Dict[str, Any]:
        if self.adapter.spec.mock_mode or self.adapter._model is None:
            raise LiveUnavailable(
                "LogitLensPipeline requires loaded GPT-2 weights. The previous "
                "implementation produced a full layer sweep without them, "
                "hardcoding the expected answer for every layer past 60% depth."
            )

        n_layers = self.adapter.spec.num_layers
        manifest = self._versioning.create_manifest(
            paper_id=self.PAPER_ID,
            pipeline_name="LogitLensPipeline",
            model_id=self.adapter.spec.model_id,
            hf_repo_id=self.adapter.spec.hf_repo_id,
            dataset_name="Logit Lens Eval Prompts",
            dataset_num_examples=len(_EVAL_PROMPTS),
            random_seed=42,
        )

        layer_results: List[Dict[str, Any]] = []
        final_correct = 0
        convergence_layers: List[int] = []
        expected_recovered_anywhere = 0

        for item in _EVAL_PROMPTS:
            prompt, expected = item["prompt"], item["expected"]
            lens = self.adapter.logit_lens(prompt)
            rows = lens["layers"]
            final_token = lens["final_top_token"]

            final_correct += 1 if expected.lower() in final_token.lower() else 0

            # Convergence: first layer already predicting what the model
            # ultimately predicts. Derived from the sweep, so a prompt whose
            # prediction only stabilises late reports a late layer.
            convergence = next(
                (row["layer"] for row in rows if row["top_token"] == final_token),
                len(rows) - 1,
            )
            convergence_layers.append(convergence)

            # Did the answer ever surface at any depth? The lens's failure mode
            # is information present but unreadable, so this distinguishes
            # "never computed" from "computed but not linearly decodable".
            ever = any(expected.lower() in row["top_token"].lower() for row in rows)
            expected_recovered_anywhere += 1 if ever else 0

            layer_results.append({
                "prompt": prompt,
                "expected": expected,
                "final_top_token": final_token,
                "final_layer_correct": expected.lower() in final_token.lower(),
                "expected_seen_at_any_layer": ever,
                "convergence_layer": convergence,
                "layer_sweep": rows,
            })

        n = len(_EVAL_PROMPTS)

        # A difference, not a ratio. The previous form,
        # ``1 - entropy_final / entropy_first``, divides by the layer-0 entropy,
        # which is near zero whenever the embedding happens to produce a peaked
        # distribution; it reported -377.49 on this model, which is not a
        # quantity of anything. The absolute change is stable and the shape is
        # carried by the curve itself.
        entropy_delta = [
            rows[-1]["entropy"] - rows[0]["entropy"]
            for rows in (r["layer_sweep"] for r in layer_results)
        ]
        mean_entropy_curve = [
            round(statistics.mean(row["layer_sweep"][i]["entropy"] for row in layer_results), 4)
            for i in range(n_layers + 1)
        ]

        observed_metrics = {
            "final_layer_top1_accuracy": round(final_correct / n, 4),
            "expected_token_seen_at_any_layer": round(expected_recovered_anywhere / n, 4),
            "convergence_layer_ratio": round(statistics.mean(convergence_layers) / n_layers, 4),
            "mean_convergence_layer": round(statistics.mean(convergence_layers), 4),
            "mean_entropy_delta_first_to_final": round(statistics.mean(entropy_delta), 4),
            "mean_entropy_by_layer": mean_entropy_curve,
        }

        report = self._report_engine.generate_report(
            paper_id=self.PAPER_ID,
            pipeline_name="LogitLensPipeline",
            model_id=self.adapter.spec.model_id,
            dataset_manifest_id=manifest.manifest_id,
            observed_metrics=observed_metrics,
            explanation_of_diffs=[
                "Every quantity is read from a real forward pass through the "
                "model's own ln_f and unembedding matrix.",
                "On GPT-2 small the lens does not recover most expected tokens "
                "at any layer, and final-layer top-1 accuracy is low. That is "
                "the measurement, not a defect in it: a 124M model is too small "
                "for linear factual recall.",
                "The previous implementation hardcoded the expected token for "
                "every layer past 60% depth, used an exponential decay for "
                "entropy, and returned a constant convergence layer.",
            ],
        )

        return {
            "pipeline": "LogitLensPipeline",
            "paper_id": self.PAPER_ID,
            "provenance": "live",
            "layer_results": layer_results,
            "observed_metrics": observed_metrics,
            "reproducibility_report": report,
            "manifest_id": manifest.manifest_id,
        }
