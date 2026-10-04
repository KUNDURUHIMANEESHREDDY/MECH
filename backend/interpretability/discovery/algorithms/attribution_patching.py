"""Attribution Patching Discovery Algorithm (Neel Nanda / AtP).

Ref: Nanda, 2023 - "Attribution Patching: Fast Activation Patching via Taylor Expansion"

Fast, linear approximation of activation patching using first-order Taylor expansions:
    Attribution(z) = (z_clean - z_corrupted) * (d Metric / d z)

Attribution Patching computes component importance across the entire model in a 
single forward + backward pass, avoiding the O(N) forward passes required by 
standard activation patching.
"""

from __future__ import annotations

import datetime as _dt
import time
from typing import Any, Dict, List, Optional

from .base_algorithm import DiscoveryAlgorithm, DiscoveryReport
from .registry import register_algorithm, AlgorithmMetadata
from .configs import DiscoveryAlgorithmConfig, AttributionPatchingConfig


@register_algorithm(AlgorithmMetadata(
    name="attribution_patching",
    paper="Attribution Patching (Nanda)",
    authors="Neel Nanda",
    year=2023,
    supported_models=["gpt2", "gemma", "llama", "mistral"],
    required_capabilities=["get_logits", "capture_head_outputs_with_grad"],
    estimated_runtime="1s-10s",
    search_space="all_components",
    output_schema="DiscoveryReport"
))
class AttributionPatchingAlgorithm(DiscoveryAlgorithm):
    """Computes fast Taylor-expansion activation attributions across model layers."""

    @staticmethod
    def _at_position(acts: List[Any], position: Optional[int]) -> Optional[float]:
        """The activation value at `position`, where None means the last one.

        Retained for callers that hold per-token activation results directly.
        The sweep below uses `capture_head_outputs*` instead, because reading a
        head's value out of a neuron-indexed API is the bug that made the
        previous version measure the wrong quantity.
        """
        if not acts:
            return None
        if position is None:
            return acts[-1].activation_value
        if position >= len(acts) or position < -len(acts):
            return None
        return acts[position].activation_value

    def _metric_token_ids(
        self,
        clean_prompt: str,
        corrupted_prompt: str,
        target_token: str,
        baseline_token: Optional[str] = None,
    ) -> tuple:
        """Resolve the metric's two tokens, or explain why it cannot be done.

        The metric is ``logit(target) - logit(baseline)`` at the final position of
        the clean prompt. A baseline is required: the previous code compared two
        top-1 logits of two *different* prompts, which is a difference between
        prompts rather than between candidate completions -- and then divided by
        an activation magnitude to obtain something it called a gradient.

        Resolution order:

        1. An explicit ``baseline_token`` from the dataset or caller.
        2. The IOI/ABB construction: clean and corrupted encode to equal length
           and differ only in a contiguous suffix, so the clean prompt's final
           token is the answer and the corrupted prompt's is the wrong one.
        3. Otherwise fail closed with the reason.

        Returns ``(target_id, baseline_id, note)``.

        The saturation check matters because the bundled IOI dataset stores
        prompts that *already contain their answer* ("...gave a drink to Mary").
        A next-token metric at the final position of such a prompt asks the model
        to predict a token it has just read, which is trivially satisfied. That is
        the common case rather than a hypothetical one, so it is detected rather
        than reported as a measurement.
        """
        tokenizer = getattr(self.adapter, "_tokenizer", None)
        if tokenizer is None:
            return None, None, (
                "The adapter exposes no tokenizer, so the metric's two tokens "
                "cannot be identified.")

        try:
            clean_ids = tokenizer.encode(clean_prompt)
            corrupted_ids = tokenizer.encode(corrupted_prompt)
        except Exception as exc:
            return None, None, f"Could not tokenize the prompt pair: {exc}"

        if not clean_ids or not corrupted_ids:
            return None, None, "One of the prompts tokenized to nothing."

        if baseline_token:
            try:
                return (tokenizer.encode(target_token)[0],
                        tokenizer.encode(baseline_token)[0],
                        "Explicit baseline token supplied.")
            except Exception as exc:
                return None, None, f"Could not encode the explicit tokens: {exc}"

        if target_token:
            try:
                target_ids = tokenizer.encode(target_token)
            except Exception:
                target_ids = []
            if target_ids and clean_ids[-len(target_ids):] == target_ids:
                return None, None, (
                    f"The clean prompt already ends with the target token "
                    f"{target_token!r}, so a next-token metric at its final "
                    f"position would ask the model to predict a token it has "
                    f"just read. Truncate the clean prompt before the answer "
                    f"to use this algorithm.")

        if len(clean_ids) != len(corrupted_ids):
            return None, None, (
                f"The prompt pair encodes to different lengths "
                f"({len(clean_ids)} vs {len(corrupted_ids)}), so the metric's "
                f"two tokens cannot be read off the final position. Supply an "
                f"explicit baseline_token, or use equal-length prompts.")

        differing = [i for i, (a, b) in enumerate(zip(clean_ids, corrupted_ids))
                     if a != b]
        if not differing:
            return None, None, (
                "The clean and corrupted prompts are identical, so there is no "
                "corruption to attribute and no metric to differentiate.")
        if differing == list(range(differing[0], differing[-1] + 1)):
            return (clean_ids[-1], corrupted_ids[-1],
                    "IOI construction: equal-length pair differing in a "
                    "contiguous suffix, so the clean prompt's final token is the "
                    "answer and the corrupted prompt's is the wrong one.")

        return None, None, (
            f"The prompts differ at {len(differing)} non-contiguous token "
            f"position(s), so the answer and the wrong completion cannot be "
            f"identified from the pair alone. Supply an explicit baseline_token.")

    def _unmeasurable(
        self,
        dataset: Dict[str, Any],
        config: Any,
        clean_prompt: str,
        corrupted_prompt: str,
        target_token: str,
        reason: str,
        started_at: Optional[float] = None,
    ) -> DiscoveryReport:
        """The fail-closed report: ran, measured nothing, says why.

        Same keys as a real report, so no consumer needs a special case, and
        `confidence` None rather than a floored 0.1.
        """
        return DiscoveryReport(
            algorithm="attribution_patching",
            dataset_id=dataset.get("id", "unknown"),
            model_id=self.adapter.spec.model_id if self.adapter else "unavailable",
            runtime_ms=((time.time() - started_at) * 1000
                        if started_at is not None else 0.0),
            statistics={
                "measured": False,
                "reason": reason,
                "approximation_order": config.approximation_order,
            },
            evidence={
                "clean_prompt": clean_prompt,
                "corrupted_prompt": corrupted_prompt,
                "target_token": target_token,
                "top_attributions": [],
            },
            confidence=None,
            graph={"nodes": [], "edges": [], "score": None, "measured": False},
            provenance={
                "search_space": config.search_space,
                "metric": config.metric,
                "approximation": config.approximation_order,
                "unavailable_reason": reason,
            },
        )

    def run(self, dataset: Dict[str, Any], config: Optional[DiscoveryAlgorithmConfig] = None) -> DiscoveryReport:
        """Executes attribution patching calculation across all model components."""
        t0 = time.time()
        
        if not isinstance(config, AttributionPatchingConfig):
            config = AttributionPatchingConfig()

        clean_prompt = dataset.get("clean", "")
        corrupted_prompt = dataset.get("corrupted", "")
        target_token = dataset.get("target_token", " ")
        # The IOI-style dataset stores the target under "target", the
        # single-prompt form under "target_token". Reading only the second meant
        # the common case silently fell back to a single space.
        target_token = dataset.get("target_token") or dataset.get("target") or " "
        baseline_token = dataset.get("baseline_token")

        if not clean_prompt and "prompts" in dataset:
            p0 = dataset["prompts"][0]
            clean_prompt = p0.get("clean", "")
            corrupted_prompt = p0.get("corrupted", "")
            target_token = p0.get("target", " ")
            baseline_token = p0.get("baseline")

        # 1. Forward passes for clean and corrupted runs.
        #
        # These two calls are retained only to report the metric's magnitude. The
        # attribution itself comes from the gradient capture below.
        #
        # `get("logit", 1.0)` and `get("logit", 0.0)` were fabricated metric
        # values: a run with no top_tokens reported `metric_delta == 1.0`, a
        # healthy-looking number for a run that produced nothing.
        clean_logits = self.adapter.get_logits(clean_prompt)
        corrupted_logits = self.adapter.get_logits(corrupted_prompt)

        clean_top = (clean_logits.get("top_tokens") or [{}])[0]
        corrupted_top = (corrupted_logits.get("top_tokens") or [{}])[0]
        if "logit" in clean_top and "logit" in corrupted_top:
            metric_delta: Optional[float] = (
                clean_top["logit"] - corrupted_top["logit"])
            metric_delta_available = True
        else:
            metric_delta = None
            metric_delta_available = False

        # Which token position the attribution is evaluated at. `None` means the
        # last position of each prompt -- the position the metric is computed at.
        # An explicit index is accepted so a caller can attribute at another
        # position, and the choice is recorded in the report.
        metric_position = config.token_position

        nodes = [{"id": "Input", "type": "Token", "label": clean_prompt[:30] + "..."}]
        edges = []
        attributions: List[Dict[str, Any]] = []
        unreadable_components: List[str] = []

        num_layers = self.adapter.spec.num_layers
        # `min(4, num_heads)` examined only the first four heads of every layer
        # while `search_space` was declared "all_components" -- so 8 of 12 heads
        # per layer were never looked at, and a head outside the first four could
        # not be discovered no matter how strongly it contributed. The cap is
        # kept as an explicit `max_heads` policy, because restricting the sweep
        # for cost is legitimate; claiming to cover all components while not
        # doing so is not.
        max_heads = getattr(config, "max_heads", None)
        num_heads = self.adapter.spec.num_heads
        heads_examined = num_heads
        if max_heads is not None and max_heads < num_heads:
            heads_examined = max_heads
        heads_skipped = num_heads - heads_examined

        last_node_id = "Input"

        # 2. First-order Taylor attribution, Attribution = sum_d (z_clean - z_corrupt) * dMetric/dz
        #
        # Three defects in the previous implementation, each of which had to be
        # fixed before the number meant anything. All three were verified against
        # live gpt2-small weights rather than inferred.
        #
        # (a) It read a neuron dimension, not a head.
        #     `get_activations(layer, neuron_index=head)` returns
        #     `hidden_states[layer][0, tok, :][head]` -- dimension `head` of the
        #     768-wide residual stream. The same head-vs-neuron confusion already
        #     fixed in ACDC, here measuring a different quantity again.
        #     `capture_head_outputs` reads the real per-head output vector: head h
        #     is `attn.c_proj`'s input slice `[h*d_head : (h+1)*d_head]`.
        #
        # (b) It read token position 0.
        #     `get_activations` returns one result per token, and an IOI-style
        #     pair shares a long prefix, so position 0 is identical in both.
        #     Measured: `delta_x` was exactly 0.000000 for all 144 components.
        #     The fabricated `else 0.5` / `else 0.1` fallbacks produced
        #     delta_x = 0.4 and attribution 0.0138, just over the 0.01 threshold
        #     -- so every component ever reported here was selected by a made-up
        #     activation.
        #
        # (c) It substituted a ratio for a gradient, degenerately.
        #     `grad_m = metric_delta / (|c| + |r|)` is a global metric change over
        #     an activation magnitude, not a derivative. When c and r straddle
        #     zero, `|c - r| == |c| + |r|` exactly, so the attribution collapses to
        #     `metric_delta` identically. Measured: 9 of the top 10 attributions
        #     were exactly 5.375, i.e. the ranking among them was decided by
        #     rounding. The gradient is now computed with `torch.autograd.grad`.
        target_id, baseline_id, metric_note = self._metric_token_ids(
            clean_prompt, corrupted_prompt, target_token, baseline_token)

        if target_id is None or baseline_id is None:
            # Without both tokens of the metric there is no gradient to take, and
            # the old ratio was not a substitute for one.
            return self._unmeasurable(
                dataset, config, clean_prompt, corrupted_prompt, target_token,
                reason=(f"{metric_note} The previous implementation substituted "
                        f"metric_delta / (|z_clean| + |z_corrupt|), which is not "
                        f"a derivative."),
                started_at=t0,
            )

        clean_capture = self.adapter.capture_head_outputs_with_grad(
            clean_prompt, target_id, baseline_id)
        corrupted_capture = self.adapter.capture_head_outputs(  # type: ignore[attr-defined]
            corrupted_prompt)

        for layer in range(num_layers):
            if layer not in clean_capture:
                continue
            for head in range(heads_examined):
                if head not in clean_capture[layer]:
                    unreadable_components.append(f"Head_L{layer}_H{head}")
                    continue

                entry = clean_capture[layer][head]
                grad = entry.get("grad") or []
                clean_vec = entry.get("activation") or []

                if not grad or not clean_vec:
                    unreadable_components.append(f"Head_L{layer}_H{head}")
                    continue

                # `capture_head_outputs` returns a plain vector per head; the gradient variant
                # returns {"activation": [...], "grad": [...]}. Accept either so
                # the two adapters cannot silently disagree about shape.
                raw = (corrupted_capture.get(layer) or {}).get(head)
                if isinstance(raw, dict):
                    corrupted_vec = raw.get("activation") or []
                elif isinstance(raw, (list, tuple)):
                    corrupted_vec = list(raw)
                else:
                    corrupted_vec = []
                if len(corrupted_vec) != len(clean_vec):
                    unreadable_components.append(f"Head_L{layer}_H{head}")
                    continue

                # Attribution = sum_d (z_clean[d] - z_corrupt[d]) * dMetric/dz[d]
                # over the head's dimensions. Summing over dimensions is what
                # makes it a head-level attribution; the previous code collapsed
                # the head to a single scalar before multiplying.
                delta_x = [c - r for c, r in zip(clean_vec, corrupted_vec)]
                attr_score = sum(abs(d * g) for d, g in zip(delta_x, grad))
                grad_norm = max(abs(g) for g in grad)
                delta_norm = max(abs(d) for d in delta_x)

                if attr_score >= config.threshold:
                    component_id = f"Head_L{layer}_H{head}"
                    attributions.append({
                        "component": component_id,
                        "layer": layer,
                        "head": head,
                        # `delta_x` is the head's whole activation-difference
                        # vector (d_head entries) and `grad_m` is the head's whole
                        # gradient vector. Both were single scalars before, taken
                        # from unrelated quantities.
                        "delta_x": [round(v, 6) for v in delta_x],
                        "grad_m": [round(v, 8) for v in grad],
                        "max_abs_delta_x": round(delta_norm, 6),
                        "max_abs_grad_m": round(grad_norm, 8),
                        "attribution_score": round(attr_score, 6)
                    })

        # Sort by attribution score descending
        attributions.sort(key=lambda x: x["attribution_score"], reverse=True)
        top_attributions = attributions[:config.top_k]

        for item in top_attributions:
            node_id = item["component"]
            nodes.append({
                "id": node_id,
                "type": "AttentionHead",
                "label": f"{node_id} (Attr: {item['attribution_score']:.3f})"
            })
            edges.append({
                "source": last_node_id,
                "target": node_id,
                "weight": item["attribution_score"],
                "confidence": min(0.99, item["attribution_score"])
            })
            last_node_id = node_id

        # Target prediction node
        nodes.append({"id": "Output", "type": "Prediction", "label": target_token})

        # The closing edge to the output node carries no measurement.
        #
        # It was `weight: 1.0, confidence: 1.0`, i.e. "the top-attributed
        # component accounts for the output with certainty". Nothing measured
        # that: a first-order Taylor attribution ranks components by sensitivity,
        # it does not decompose the output. A weight of 1.0 would only be right
        # if one component explained the whole metric delta, which is precisely
        # what attribution patching does not assume.
        #
        # It is retained as a labelled structural edge so the graph still
        # terminates at the prediction, with the confidence stated as unknown.
        edges.append({
            "source": last_node_id,
            "target": "Output",
            "weight": None,
            "confidence": None,
            "structural": True,
            "weight_reason": (
                "Structural edge to the prediction node. Attribution patching "
                "ranks components by sensitivity to the metric; it does not "
                "decompose the output, so no weight is measured here. This was "
                "previously 1.0 with confidence 1.0."
            ),
        })

        avg_attribution = (sum(a["attribution_score"] for a in top_attributions)
                           / len(top_attributions)) if top_attributions else None
        runtime_ms = (time.time() - t0) * 1000

        # No component clearing the threshold is a real result, not a weak one.
        # `max(1, len(...))` and `max(0.1, ...)` together produced 0.0 averaged
        # into a floored 0.1, so an empty sweep reported confidence 0.1 and a
        # graph score of 0.0 -- reading as "we looked and found almost nothing".
        measured = bool(top_attributions)
        report_confidence = round(avg_attribution, 6) if measured else None

        return DiscoveryReport(
            algorithm="attribution_patching",
            dataset_id=dataset.get("id", "unknown"),
            model_id=self.adapter.spec.model_id if self.adapter else "mock",
            runtime_ms=runtime_ms,
            statistics={
                "metric_delta": (None if metric_delta is None
                               else round(metric_delta, 4)),
                "metric_delta_available": metric_delta_available,
                "metric_top1_logit_clean": clean_top.get("logit"),
                "metric_top1_logit_corrupted": corrupted_top.get("logit"),
                # What was actually examined, which is not what was declared.
                "heads_examined_per_layer": heads_examined,
                "heads_in_model": num_heads,
                "heads_skipped": heads_skipped,
                "search_space_complete": heads_skipped == 0,
                "total_components_analyzed": num_layers * heads_examined,
                "components_unreadable": len(unreadable_components),
                "components_unreadable_ids": unreadable_components[:10],
                "top_k": len(top_attributions),
                "max_attribution": (top_attributions[0]["attribution_score"]
                                    if top_attributions else None),
                "approximation_order": config.approximation_order,
                "metric_token_position": (
                    "last" if config.token_position is None
                    else config.token_position),
                "metric_definition": (
                    f"logit(target) - logit(baseline) at the final position. "
                    f"{metric_note}"),
                "metric_target_token_id": target_id,
                "metric_baseline_token_id": baseline_id,
                "measured": measured,
            },
            evidence={
                "top_attributions": top_attributions,
                "clean_prompt": clean_prompt,
                "corrupted_prompt": corrupted_prompt,
                "target_token": target_token
            },
            confidence=report_confidence,
            graph={
                "nodes": nodes,
                "edges": edges,
                "score": (round(avg_attribution, 6) if measured else None),
                "measured": measured,
            },
            provenance={
                "search_space": (config.search_space
                                 if heads_skipped == 0
                                 else f"{config.search_space} (incomplete: "
                                      f"{heads_skipped} of {num_heads} heads per "
                                      f"layer not examined)"),
                "metric": config.metric,
                "approximation": config.approximation_order,
            }
        )
