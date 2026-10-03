"""Causal Scrubbing Discovery Algorithm (Chan et al., 2022).

Ref: "Causal Scrubbing: a tool for rigorously testing interpretability
hypotheses" (Redwood Research).

Tests whether a computational hypothesis H explains a model behaviour by
*scrubbing*: replacing the activations at the hypothesised location with
activations resampled from a different input that satisfies an equivalence
relation E, then measuring what happens to the behaviour on the original input.

Direction of the test, which the previous version of this file had backwards in
its documentation:

    scrub breaks the behaviour  -> the hypothesised path matters -> H supported
    scrub preserves the behaviour -> the path does not matter      -> H refuted

The code was right about this and the module docstring was wrong; it described
"high behavior_preservation" as validating the hypothesis, which inverts the
paper's test. The docstring is corrected here and the outcome is reported as
`hypothesis_supported` rather than the ambiguous `validated`, which reads as if
the behaviour had been validated.

What was wrong with the implementation
--------------------------------------
Three defects, the first of which is the same bug that affected ACDC:

1. It swept `(layer, head)` pairs and called

       self.adapter.get_activations(ref_prompt, layer=layer, neuron_index=head)
       self.adapter.patch_activation(clean_prompt, layer=layer, neuron_index=head, ...)

   Both parameters are named `neuron_index` and both index something that is not
   an attention head: `get_activations` reads
   `hidden_states[layer][0, tok, :][head]`, a dimension of the 768-wide residual
   stream, and `patch_activation` writes `transformer.h[layer].mlp` output at index
   `head`, i.e. MLP neuron `head` of 3072. Both succeeded -- 768 and 3072 both
   exceed 12 -- so the sweep produced plausible numbers for a set of components it
   never touched.

2. `preservation_ratio = preserved_logit / base_logit_score` divides one logit by
   another. The ratio has no interpretation as a degree of behaviour
   preservation: it is not bounded by 0 or 1 in any meaningful sense, and it is
   not the quantity the paper defines. It was then clamped into [0, 1], so a
   meaningless quantity was laundered into a clean-looking fraction, given to the
   graph as an edge weight, *and* published as the report's confidence. The same
   division was applied again to the aggregate (`behavior_preservation`), so the
   hypothesis verdict was derived from it too.

3. `resample_count` (default 10, documented as "number of resampled reference
   runs per sample") was never used. The loop ran once per (layer, head) and
   echoed the configured value into `statistics` as though it had.

   Worse, the reference prompt was drawn with `rng.randint(0, len(prompts) - 1)`
   over the whole prompt list. With the default single-prompt dataset that is
   always index 0 -- the same prompt being scrubbed -- so the "resampled"
   activation was the original activation and the scrub was a no-op that measured
   a head against itself.

Measurement now
---------------
    clean_score       = target-token logit on the clean prompt
    corrupted_score   = target-token logit on the corrupted prompt (the floor)
    gap               = clean_score - corrupted_score

    for each candidate (layer, head) and each of `resample_count` reference
    prompts: patch that reference prompt's real head-output vector into the clean
    run and read the target-token logit

        preservation = (scrubbed - corrupted_score) / gap

so 1.0 means the scrub left the behaviour fully intact and 0.0 means it destroyed
it, which are the endpoints the paper's test needs. `hypothesis_supported` is
then `1 - mean(preservation) > tolerance`.

The value is reported unclamped. Scrubbing can overshoot (patching in a foreign
activation can push the behaviour *past* the clean run), and silently clamping
that to 1.0 would hide the most informative case.

Honest limitation
-----------------
`equivalence_class` is configurable ("token_type", "position",
"semantic_category") but membership is not verified: this checks that a reference
prompt has the same token count as the clean prompt, which is necessary for
structural equivalence and nowhere near sufficient for the semantic classes the
config names. The check that was performed is recorded in the report as
`equivalence_check`, so the gap between the configured class and the verified one
is visible rather than implied.
"""

from __future__ import annotations

import datetime as _dt
import random
import statistics
import time
from typing import Any, Dict, List, Optional, Tuple

from .base_algorithm import DiscoveryAlgorithm, DiscoveryReport
from .registry import register_algorithm, AlgorithmMetadata
from .configs import DiscoveryAlgorithmConfig, CausalScrubbingConfig


@register_algorithm(AlgorithmMetadata(
    name="causal_scrubbing",
    paper="Causal Scrubbing (Chan et al.)",
    authors="Lawrence Chan, Adrienne Tran, John S. Ward, et al.",
    year=2022,
    supported_models=["gpt2", "gemma", "llama", "mistral"],
    required_capabilities=["patch_head_output", "capture_head_outputs", "get_logits"],
    estimated_runtime="10s-1m",
    search_space="equivalence_classes",
    output_schema="DiscoveryReport"
))
class CausalScrubbingAlgorithm(DiscoveryAlgorithm):
    """Hypothesis test via activation resampling under an equivalence class."""

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _target_token_id(self, target_token: str) -> Optional[int]:
        try:
            ids = self.adapter._tokenizer(target_token)["input_ids"]
        except Exception:
            return None
        return ids[0] if len(ids) == 1 else None

    def _target_logit(self, prompt: str, target_id: int) -> float:
        """Logit of `target_id` at the final position.

        A dedicated forward pass, because `get_logits` returns only a top-k and
        the target of a scrubbed run is frequently absent from it.
        """
        import torch

        inputs = self.adapter._tokenizer(prompt, return_tensors="pt").to(
            self.adapter._model.device
        )
        with torch.no_grad():
            logits = self.adapter._model(**inputs).logits[0, -1, :]
        return float(logits[target_id])

    @staticmethod
    def _unrunnable(dataset: Dict[str, Any], model_id: str, runtime_ms: float,
                    extra: Optional[Dict[str, Any]] = None) -> DiscoveryReport:
        """A report that says the test could not be run, and why.

        Deliberately shaped like a normal report so consumers do not have to
        special-case it, but with every score None and the verdict absent.
        Reporting a verdict for a test that did not run is the failure this
        whole effort has been about.

        The reason is taken from ``extra["required"]`` rather than passed
        separately. An earlier version took both, and every call site passed a
        timestamp where the reason belonged -- so a single-prompt dataset
        reported its reason as "1791041600.7683995". Deriving it removes the
        ordering hazard instead of relying on six call sites to get it right.
        """
        details = dict(extra or {})
        reason = details.pop("required", None) or (
            "causal scrubbing could not be run; see statistics for detail"
        )
        statistics_block: Dict[str, Any] = {
            "measured": False,
            "reason": reason,
            "base_logit_score": None,
            "scrubbed_logit_score": None,
            "behavior_preservation": None,
            "hypothesis_supported": None,
            "resamples_performed": 0,
            "components_scrubbed": 0,
            "tolerance": None,
        }
        statistics_block.update(details)
        return DiscoveryReport(
            algorithm="causal_scrubbing",
            dataset_id=dataset.get("id", "unknown"),
            model_id=model_id,
            runtime_ms=round(runtime_ms, 2),
            statistics=statistics_block,
            evidence={
                "hypothesis_supported": None,
                "reason": reason,
            },
            confidence=None,
            graph={"nodes": [], "edges": [], "score": None},
            provenance={
                "measured": False,
                "reason": reason,
                # Always false, including when the reason *is* the equivalence
                # check. Omitting it here made a report that declined to run for
                # want of an equivalent resample look, on this field alone, like
                # one that had verified equivalence.
                "equivalence_verified": False,
            },
        )

    # ------------------------------------------------------------------ #
    # Entry point                                                       #
    # ------------------------------------------------------------------ #

    def run(self, dataset: Dict[str, Any], config: Optional[DiscoveryAlgorithmConfig] = None) -> DiscoveryReport:
        """Scrub hypothesised paths with resampled activations and measure."""
        t0 = time.time()
        if not isinstance(config, CausalScrubbingConfig):
            config = CausalScrubbingConfig()

        model_id = self.adapter.spec.model_id if self.adapter else "mock"

        # Fail closed rather than fall back to a fixture. The previous version
        # defaulted `base_logit_score` to 1.0 whenever `top_tokens` was absent,
        # and that 1.0 was the denominator of the preservation ratio, so a missing
        # measurement silently became the scale of every reported score.
        if self.adapter is None or getattr(self.adapter.spec, "mock_mode", False) \
                or getattr(self.adapter, "_model", None) is None:
            return self._unrunnable(
                dataset, model_id, (time.time() - t0) * 1000,
                extra={"required": "loaded GPT-2 weights; causal scrubbing has no fixture"},
            )

        prompts = dataset.get("prompts") or [{
            "id": "p0",
            "clean": dataset.get("clean", "John gave a drink to Mary"),
            "corrupted": dataset.get("corrupted", "John gave a drink to John"),
            "target": dataset.get("target_token", " Mary"),
        }]
        clean_prompt = prompts[0].get("clean", "")
        corrupted_prompt = prompts[0].get("corrupted", dataset.get("corrupted", ""))
        target_token = prompts[0].get("target", dataset.get("target_token", " Mary"))

        target_id = self._target_token_id(target_token)
        if target_id is None:
            return self._unrunnable(
                dataset, model_id, (time.time() - t0) * 1000,
                extra={"required": (
                    f"target_token {target_token!r} must be a single token for a "
                    f"logit-difference metric"
                )},
            )

        # Scrubbing needs an input to resample *from*. Resampling from the
        # prompt being scrubbed is not resampling -- it patches a head with its
        # own output and measures the head against itself.
        candidates = [
            p for p in prompts[1:]
            if p.get("clean") and p.get("clean") != clean_prompt
        ]
        if not candidates:
            return self._unrunnable(
                dataset, model_id, (time.time() - t0) * 1000,
                extra={"required": (
                    "at least one alternative prompt to resample from; with a "
                    "single prompt the scrub would patch each head with its own "
                    "activation and measure nothing. The previous implementation "
                    "drew from the whole list, which with one prompt always chose "
                    "the prompt being scrubbed."
                )},
            )

        clean_len = len(self.adapter._tokenizer(clean_prompt)["input_ids"])
        viable = [
            p for p in candidates
            if len(self.adapter._tokenizer(p["clean"])["input_ids"]) == clean_len
        ]
        equivalence_check = {
            "configured_class": config.equivalence_class,
            "verified_class": "same_token_count",
            "necessary_not_sufficient": True,
            "candidates_offered": len(candidates),
            "candidates_same_length": len(viable),
            "note": (
                f"equivalence_class={config.equivalence_class!r} is not verified. "
                f"Only equal token count is checked, which is necessary for "
                f"structural equivalence and insufficient for a semantic class."
            ),
        }
        if not viable:
            return self._unrunnable(
                dataset, model_id, (time.time() - t0) * 1000,
                extra={"required": (
                    "no alternative prompt has the same token count as the clean "
                    "prompt, so no structurally equivalent resample exists"
                ), "equivalence_check": equivalence_check},
            )

        clean_score = self._target_logit(clean_prompt, target_id)
        corrupted_score = self._target_logit(corrupted_prompt, target_id) \
            if corrupted_prompt else None
        if corrupted_score is None:
            return self._unrunnable(
                dataset, model_id, (time.time() - t0) * 1000,
                extra={"required": "a corrupted prompt, to establish the floor the "
                                   "preservation ratio is measured against"},
            )
        gap = clean_score - corrupted_score
        if abs(gap) < 1e-6:
            return self._unrunnable(
                dataset, model_id, (time.time() - t0) * 1000,
                extra={"required": (
                    "the clean and corrupted prompts give the target token "
                    "identical logits, so there is no gap to normalise by"
                ), "equivalence_check": equivalence_check},
            )

        num_layers = self.adapter.spec.num_layers
        scrub_layers = [num_layers - 2, num_layers - 1] if num_layers >= 2 else [0]
        rng = random.Random(config.seed)

        # Reference head vectors, captured once per prompt and reused. The
        # previous code called get_activations inside the head loop, i.e. once per
        # (layer, head) resample, for a value that never varied by head.
        ref_vectors: Dict[str, Dict[int, Dict[int, List[float]]]] = {}

        def vectors_for(prompt: str) -> Dict[int, Dict[int, List[float]]]:
            if prompt not in ref_vectors:
                ref_vectors[prompt] = self.adapter.capture_head_outputs(prompt)
            return ref_vectors[prompt]

        nodes: List[Dict[str, Any]] = [{
            "id": f"Hypothesis_{config.equivalence_class}",
            "type": "Hypothesis",
            "label": f"Hypothesis ({config.equivalence_class})",
        }]
        edges: List[Dict[str, Any]] = []
        root_id = nodes[0]["id"]

        all_preservations: List[float] = []
        per_component: Dict[str, List[float]] = {}
        scrubbed_scores: List[float] = []

        for layer in scrub_layers:
            for head in range(min(4, self.adapter.spec.num_heads)):
                node_id = f"Scrub_L{layer}_H{head}"
                samples: List[float] = []
                raw_scores: List[float] = []

                # `resample_count` is finally used. It was configured as 10,
                # documented as "number of resampled reference runs per sample",
                # and then never referenced outside the statistics block.
                for _ in range(config.resample_count):
                    ref = rng.choice(viable)
                    donor = vectors_for(ref["clean"]).get(layer, {}).get(head)
                    if donor is None:
                        continue
                    patch = self.adapter.patch_head_output(
                        prompt=clean_prompt,
                        layer=layer,
                        head_index=head,
                        patch_vector=donor,
                        score_token_id=target_id,
                    )
                    preservation = (patch.patched_logit - corrupted_score) / gap
                    samples.append(preservation)
                    raw_scores.append(patch.patched_logit)

                if not samples:
                    continue

                mean_preservation = statistics.mean(samples)
                all_preservations.extend(samples)
                per_component[node_id] = [round(s, 4) for s in samples]
                scrubbed_scores.extend(raw_scores)

                nodes.append({
                    "id": node_id,
                    "type": "ScrubbedPath",
                    "label": f"L{layer}H{head} (Resampled x{len(samples)})",
                    "layer": layer,
                    "head": head,
                })

                # The edge weight is the measured mean preservation. Reported
                # unclamped: a scrub can push the target logit past the clean
                # run's, giving >1.0, and clamping that to 1.0 would discard the
                # most informative outcome. `min(1.0, max(0.0, ...))` over a
                # ratio of two logits used to sit here.
                edges.append({
                    "source": root_id,
                    "target": node_id,
                    "weight": round(mean_preservation, 4),
                    "confidence": round(mean_preservation, 4),
                    "measured_from": (
                        "clean-to-corrupted target-logit preservation over "
                        f"{len(samples)} resampled reference prompts"
                    ),
                    "samples": per_component[node_id],
                    "unclamped": True,
                })

        behavior_preservation = statistics.mean(all_preservations) \
            if all_preservations else None
        hypothesis_supported = (
            (1.0 - behavior_preservation) > config.tolerance
            if behavior_preservation is not None else None
        )

        return DiscoveryReport(
            algorithm="causal_scrubbing",
            dataset_id=dataset.get("id", "unknown"),
            model_id=model_id,
            runtime_ms=round((time.time() - t0) * 1000, 2),
            statistics={
                "measured": behavior_preservation is not None,
                "base_logit_score": round(clean_score, 4),
                "corrupted_logit_score": round(corrupted_score, 4),
                "scrubbed_logit_score": round(statistics.mean(scrubbed_scores), 4)
                    if scrubbed_scores else None,
                "behavior_preservation": round(behavior_preservation, 4)
                    if behavior_preservation is not None else None,
                # Named for the paper's test rather than "validated", which read
                # as though the behaviour had been validated. The hypothesis is
                # *supported* when scrubbing breaks the behaviour.
                "hypothesis_supported": hypothesis_supported,
                "resample_count": config.resample_count,
                "resamples_performed": len(all_preservations),
                "reference_prompts_used": len(ref_vectors),
                "components_scrubbed": len(edges),
                "tolerance": config.tolerance,
                "equivalence_check": equivalence_check,
                "preservation_unclamped": True,
            },
            evidence={
                "equivalence_class": config.equivalence_class,
                "scrub_paths": config.scrub_paths,
                "hypothesis_supported": hypothesis_supported,
                "behavior_preservation": behavior_preservation,
                "clean_prompt": clean_prompt,
                "target_token": target_token,
                "per_component_preservation": per_component,
            },
            confidence=None,
            graph={
                "nodes": nodes,
                "edges": edges,
                "score": round(behavior_preservation, 4)
                    if behavior_preservation is not None else None,
            },
            provenance={
                "provenance": "live",
                "search_space": config.equivalence_class,
                "metric": (
                    "target-token logit preservation, normalised by the "
                    "clean-minus-corrupted gap; unclamped"
                ),
                "equivalence_verified": False,
            },
        )
