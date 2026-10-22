"""Real Automated Circuit Discovery (ACDC) Search Algorithm.

Ref: Conmy et al., 2023 - Towards Automated Circuit Discovery for Arbitrary Tasks.

Implements true ACDC circuit discovery:
  1. Activation Caching (Clean & Corrupted Forward Passes)
  2. Greedy Reverse-Topological Edge Pruning
  3. Metric Evaluation (Logit Diff & Fraction of Variance Explained)
  4. Graph Reconstruction & Node Pruning Statistics
"""

from __future__ import annotations

import datetime as _dt
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .base_algorithm import DiscoveryAlgorithm, DiscoveryReport
from .registry import register_algorithm, AlgorithmMetadata
from .configs import DiscoveryAlgorithmConfig, ACDCConfig
from backend.interpretability.discovery.live_measure import MIN_DENOMINATOR


@dataclass
class ActivationCache:
    """Stores component activations across clean and corrupted runs."""
    clean_logits: Dict[str, Any]
    corrupted_logits: Dict[str, Any]
    head_activations: Dict[Tuple[int, int], float] = field(default_factory=dict)
    mlp_activations: Dict[int, float] = field(default_factory=dict)


@register_algorithm(AlgorithmMetadata(
    name="acdc",
    paper="Automatic Circuit Discovery (Conmy et al.)",
    authors="Arthur Conmy, Ian Mavor, Aengus Lynch, et al.",
    year=2023,
    supported_models=["gpt2", "gemma", "llama", "mistral"],
    required_capabilities=["patch_head_outputs", "capture_head_outputs", "get_logits"],
    estimated_runtime="10s-5m",
    search_space="attention_heads, mlps",
    output_schema="DiscoveryReport"
))
class ACDCAlgorithm(DiscoveryAlgorithm):
    """Greedy reverse-topological pruning of a measured circuit.

    The intervention is head-level throughout: the clean run's output vector for
    a candidate head is written into the corrupted run, and the recovery of the
    target token's logit is measured. The whole circuit is measured at once --
    every retained head is injected in a single forward pass -- so the number
    the search optimises is the circuit's own recovery, not a head's.

    What is real: activation capture, the pruning sweep, the resulting
    component set, the circuit's measured recovery, and each retained head's
    measured marginal contribution -- all from the adapter.

    What changed, and why. The search used to score each head independently
    against a fixed fraction of the clean-minus-corrupted logit gap and kept
    every head that cleared the bar. A head's individual contribution to a
    collective effect is small by construction, so nearly every head cleared
    it: a live run retained 138 of 144 candidates -- a "circuit" of 138 nodes
    whose own measured recovery was -307% of the gap. The search was not
    pruning, and nothing in the report's shape said so.

    Two behaviours were removed rather than fixed, because both replaced a
    measurement with an assumption:

    * When nothing survived pruning, the search result was overridden with L9H9
      and L10H0 under the comment "ensure top critical heads are retained" --
      the published answer, substituted for the answer the search gave.
    * Every retained edge carried `confidence: 0.95` and the edge completing the
      circuit carried `confidence: 1.0`.
    """


    def __init__(self, adapter: Any = None) -> None:
        super().__init__(adapter)
        # Populated by build_activation_cache; kept on the instance because the
        # sweep needs the vectors and re-capturing per candidate would cost 144
        # extra forward passes.
        self._head_vectors: Dict[Tuple[int, int], List[float]] = {}
        self._capture_error: Optional[str] = None
        # Measured circuit recoveries, keyed by the retained head set. The
        # search measures a circuit more than once (a trial set becomes the
        # next starting point), and a forward pass is the expensive part.
        self._recovery_cache: Dict[Any, Any] = {}

    def build_activation_cache(self, clean_prompt: str, corrupted_prompt: str, num_layers: int, num_heads: int) -> ActivationCache:
        """Capture the clean prompt's real per-head output vectors.

        Previously this called ``get_activations(layer=layer, neuron_index=head)``
        and stored the result as ``head_activations[(layer, head)]``. That call
        reads ``hidden_states[layer][0, tok, :][head]`` -- dimension ``head`` of
        the residual stream, which is 768-wide, so it succeeded and returned a
        plausible float. It was not an attention head. Nothing downstream could
        tell, because the value was a real activation of *something*.

        Now it uses ``capture_head_outputs``, which hooks ``attn.c_proj`` and
        slices head ``h``'s actual output vector (64 dims) out of the
        concatenated per-head tensor. ``mlp_activations`` is left empty rather
        than filled with a stand-in, because MLP activations are not what this
        algorithm sweeps and a proxy would invite the same confusion again.
        """
        clean_logits = self.adapter.get_logits(clean_prompt)
        corrupted_logits = self.adapter.get_logits(corrupted_prompt)

        head_acts: Dict[Tuple[int, int], float] = {}
        vectors: Dict[Tuple[int, int], List[float]] = {}
        try:
            captured = self.adapter.capture_head_outputs(clean_prompt)
        except Exception as exc:
            self._capture_error = f"{type(exc).__name__}: {exc}"
            captured = {}

        for layer, heads in captured.items():
            for head, vector in heads.items():
                key = (layer, head)
                vectors[key] = vector
                head_acts[key] = sum(v * v for v in vector) ** 0.5

        self._head_vectors = vectors
        return ActivationCache(
            clean_logits=clean_logits,
            corrupted_logits=corrupted_logits,
            head_activations=head_acts,
            mlp_activations={},
        )

    def _score_tokens(
        self, clean_prompt: str, corrupted_prompt: str, target_id: int
    ) -> tuple:
        """Logit of `target_id` at the final position for both prompts.

        A dedicated forward pass rather than `get_logits`, which returns only a
        top-k. The target of an IOI pair is often not in the top 5 of the
        corrupted run -- that is the whole point of the corrupted prompt -- so a
        top-k lookup could not score it.

        The attention kernel is pinned to `eager` because that is the kernel
        `capture_head_outputs` and `patch_head_outputs` run under. A baseline
        measured under one kernel and an intervention measured under the other
        carry a constant disagreement term (measured at 0.0625 logits at layer
        6), and the IOI logit gap this search normalises by is smaller than
        that -- so a mixed-kernel baseline would put a systematic error into
        every candidate's recovery.
        """
        import torch
        scores = []
        config = self.adapter._model.config
        previous_impl = getattr(config, "_attn_implementation", None)
        if previous_impl != "eager":
            config._attn_implementation = "eager"
        try:
            for prompt in (clean_prompt, corrupted_prompt):
                inputs = self.adapter._tokenizer(prompt, return_tensors="pt").to(
                    self.adapter._model.device
                )
                with torch.no_grad():
                    logits = self.adapter._model(**inputs).logits[0, -1, :]
                scores.append(float(logits[target_id]))
        finally:
            if previous_impl is not None:
                config._attn_implementation = previous_impl
        return scores[0], scores[1]

    def _target_token_id(self, target_token: str) -> Optional[int]:
        """Resolve the continuation token to a single id, or None if ambiguous.

        Returning None is a legitimate outcome: the logit difference is undefined
        for a target that is not one token, and ACDC says so rather than
        silently scoring the first piece.
        """
        try:
            ids = self.adapter._tokenizer(target_token)["input_ids"]
        except Exception:
            return None
        return ids[0] if len(ids) == 1 else None

    def _circuit_recovery(self, corrupted_prompt: str, target_id: int,
                          retained: Set[Tuple[int, int]],
                          clean_score: float, corr_score: float
                          ) -> Optional[Dict[str, Any]]:
        """Measure the circuit `retained` recovers, or ``None``.

        The retained heads' clean-run output vectors are injected into the
        corrupted run in ONE forward pass and the target token's logit is read
        from the result. That is ACDC's faithfulness metric, measured on the
        circuit rather than on one head at a time.

        ``None`` means the measurement could not be taken: a retained head with
        no captured vector, a model without a multi-head injection primitive,
        or a denominator too small to divide by. A measurement that did not
        happen is not a measurement of zero.
        """
        if not retained:
            return {"recovery": 0.0, "patched_logit": corr_score}
        denom = clean_score - corr_score
        if abs(denom) < MIN_DENOMINATOR:
            return None
        patches: Dict[Tuple[int, int], Any] = {}
        for layer, head in sorted(retained):
            vector = self._head_vectors.get((layer, head))
            if vector is None:
                return None
            patches[(layer, head)] = vector
        patcher = getattr(self.adapter, "patch_head_outputs", None)
        if patcher is None:
            return None
        key = frozenset(retained)
        cached = self._recovery_cache.get(key)
        if cached is not None:
            return cached
        try:
            result = patcher(corrupted_prompt, patches, score_token_id=target_id)
        except Exception:
            # LiveUnavailable (no weights, no multi-head primitive) and every
            # other failure land here: the circuit was not measured, so the
            # search says so instead of ranking on nothing.
            return None
        if not isinstance(result, dict):
            return None
        patched = result.get("patched_logit")
        if patched is None:
            return None
        value = (float(patched) - corr_score) / denom
        if not math.isfinite(value):
            return None
        measured = {"recovery": value, "patched_logit": float(patched)}
        self._recovery_cache[key] = measured
        return measured

    def run(self, dataset: Dict[str, Any], config: Optional[DiscoveryAlgorithmConfig] = None) -> DiscoveryReport:
        """Greedy reverse-topological pruning of a measured circuit.

        The search, in order:

          1. Capture the clean run's per-head output vectors -- a real forward
             pass over the loaded weights.
          2. Measure the baseline: the target token's logit on the clean prompt
             and on the corrupted one. Their difference is the effect the
             circuit has to explain.
          3. Start from every candidate head and remove them one at a time, in
             reverse topological order. A head is removed when the circuit that
             remains still recovers at least `threshold` of the gap; a head is
             kept when removing it would break the circuit, and the amount it
             contributes is the recovery that removing it costs.

        Step 3 measures the CIRCUIT, not one head at a time. That is the
        difference from the version it replaces, which scored each head
        independently against a fixed fraction of the gap and kept every head
        that cleared the bar. A head's individual contribution to a collective
        effect is small by construction, so nearly every head cleared it: the
        search that reported 144 candidates and 138 retained components was
        not pruning at all, and the circuit it returned recovered -307% of the
        gap. Nothing in that report was measurable, and the shape of the
        report hid it.
        """
        t0 = time.time()

        if not isinstance(config, ACDCConfig):
            config = ACDCConfig()
        threshold = config.threshold

        clean_prompt = dataset.get("clean", "John gave a drink to Mary")
        corrupted_prompt = dataset.get("corrupted", "John gave a drink to John")
        # The bundled datasets name the target as `target`; `target_token` is
        # the name this algorithm has always used. Both are read, because
        # reading only the second one silently scored " Mary" for every
        # prompt whose real target was somebody else.
        target_token = dataset.get("target_token") or dataset.get("target") or " Mary"

        num_layers = self.adapter.spec.num_layers if self.adapter else 12
        num_heads = self.adapter.spec.num_heads if self.adapter else 12

        # 1. Populate the activation cache with the clean run's head vectors.
        self._capture_error = None
        self._head_vectors = {}
        self._recovery_cache = {}
        cache = self.build_activation_cache(clean_prompt, corrupted_prompt, num_layers, num_heads)

        # 2. Candidate edges in reverse topological order.
        candidate_edges: List[Tuple[int, int]] = []
        for layer in reversed(range(num_layers)):
            for head in range(num_heads):
                candidate_edges.append((layer, head))

        # 3. The baseline: the target token's logit, clean minus corrupted.
        target_id = self._target_token_id(target_token)
        baseline_note: Optional[str] = None
        clean_score: Optional[float] = None
        corr_score: Optional[float] = None
        baseline_diff = 0.0
        if target_id is None:
            baseline_note = (
                f"target_token {target_token!r} is not a single token, so the "
                f"logit difference is undefined; no pruning was performed"
            )
        else:
            clean_score, corr_score = self._score_tokens(clean_prompt, corrupted_prompt, target_id)
            baseline_diff = clean_score - corr_score
            if abs(baseline_diff) < MIN_DENOMINATOR:
                baseline_note = (
                    f"the clean and corrupted prompts give the target token "
                    f"nearly identical logits (gap {baseline_diff:.4f}, below "
                    f"the {MIN_DENOMINATOR} floor below which a recovery "
                    f"fraction stops describing the measurement), so there is "
                    f"no gap to explain and no candidate can be ranked; no "
                    f"pruning was performed"
                )

        # 4. Greedy reverse-topological pruning on the circuit's own recovery.
        retained_components: Set[Tuple[int, int]] = set()
        pruned_components: Set[Tuple[int, int]] = set()
        contributions: Dict[Tuple[int, int], float] = {}
        total_evaluations = 0
        retained_set: Set[Tuple[int, int]] = set(candidate_edges)
        final_measurement: Optional[Dict[str, Any]] = None

        if target_id is not None and baseline_note is None:
            current = self._circuit_recovery(corrupted_prompt, target_id, retained_set,
                                             clean_score, corr_score)
            if current is None:
                baseline_note = (
                    "the whole-circuit measurement could not be taken on this "
                    "model, so no candidate could be ranked; no pruning was "
                    "performed"
                )
            else:
                for candidate in candidate_edges:
                    total_evaluations += 1
                    trial = retained_set - {candidate}
                    trial_measurement = self._circuit_recovery(
                        corrupted_prompt, target_id, trial, clean_score, corr_score)
                    if trial_measurement is None:
                        # This removal could not be measured. Keep the head
                        # rather than drop a component on a measurement that
                        # did not happen.
                        retained_components.add(candidate)
                        continue
                    trial_recovery = trial_measurement["recovery"]
                    if trial_recovery >= threshold:
                        # The circuit still explains the effect without it, so
                        # it is not part of the circuit.
                        retained_set = trial
                        pruned_components.add(candidate)
                        contributions[candidate] = 0.0
                        current = trial_measurement
                    else:
                        # Removing it would break the circuit. What it
                        # contributes is the recovery that removing it costs.
                        retained_components.add(candidate)
                        contributions[candidate] = round(
                            current["recovery"] - trial_recovery, 4)
                final_measurement = self._circuit_recovery(
                    corrupted_prompt, target_id, retained_set, clean_score, corr_score)

        # 5. The graph: what was measured, and what was not.
        #
        # The nodes are the token, the retained heads and the prediction. The
        # measured relation is each head's contribution to the output, so that
        # is what the head -> prediction edges carry. The token -> head edges
        # carry no weight, because this search measured no relation between
        # the token and any head -- the previous version chained the heads
        # together, which asserted a wiring between them that nothing had
        # measured, and made every retained set produce the same graph.
        nodes = [{"id": "T_0", "type": "Token", "label": clean_prompt}]
        edges: List[Dict[str, Any]] = []
        for layer, head in sorted(retained_components):
            node_id = f"H_L{layer}_H{head}"
            contribution = contributions.get((layer, head))
            nodes.append({
                "id": node_id,
                "type": "Head",
                "label": f"L{layer}H{head} (Retained)",
                "contribution": contribution,
            })
            edges.append({
                "source": "T_0",
                "target": node_id,
                "weight": None,
                "confidence": None,
                "reason": (
                    "The token-to-head relation was not measured. This search "
                    "measured each head's contribution to the output, not how "
                    "the heads are wired together."
                ),
            })
            edges.append({
                "source": node_id,
                "target": "P_0",
                "weight": contribution,
                "confidence": None,
                "measured_from": (
                    "target-logit recovery lost when this head is removed from "
                    "the circuit"
                ),
            })
        nodes.append({"id": "P_0", "type": "Prediction", "label": target_token})

        # 6. The headline metric: the pruned circuit's own recovery.
        logit_recovery_fidelity: Optional[float] = None
        fidelity_metric = "unmeasured"
        fidelity_detail: Dict[str, Any] = {
            "measured": False,
            "reason": baseline_note or "no circuit was measured",
        }
        if final_measurement is not None:
            logit_recovery_fidelity = round(float(final_measurement["recovery"]), 4)
            fidelity_metric = "target-logit recovery"
            fidelity_detail = {
                "measured": True,
                "metric": "target-logit recovery",
                "clean_target_logit": round(clean_score, 4),
                "corrupted_target_logit": round(corr_score, 4),
                "recovered_target_logit": round(float(final_measurement["patched_logit"]), 4),
                "denominator": round(baseline_diff, 4),
                "retained_heads": sorted(
                    f"L{layer}H{head}" for layer, head in retained_components),
            }

        runtime_ms = (time.time() - t0) * 1000
        component_list = [f"L{l}H{h}" for l, h in sorted(retained_components)]
        circuit_score = logit_recovery_fidelity

        return DiscoveryReport(
            algorithm="acdc",
            dataset_id=dataset.get("id", "ioi_0001"),
            model_id=self.adapter.spec.model_id if self.adapter else "gpt2",
            runtime_ms=round(runtime_ms, 2),
            statistics={
                "total_candidate_components": len(candidate_edges),
                "retained_components": len(retained_components),
                "pruned_components": len(pruned_components),
                # The fraction of the candidates the search REMOVED. It used to
                # be written twice with opposite meanings -- once as
                # pruned/candidates and once, later, as retained/candidates,
                # so the second silently overwrote the first and the key
                # named "pruning ratio" reported how much survived.
                "pruning_ratio": round(len(pruned_components) / max(1, len(candidate_edges)), 3),
                "retained_ratio": round(len(retained_components) / max(1, len(candidate_edges)), 3),
                "min_circuit_fidelity": threshold,
                "total_evaluations": total_evaluations,
                "logit_recovery_fidelity": logit_recovery_fidelity,
                "logit_recovery_fidelity_measured": bool(fidelity_detail.get("measured")),
                "logit_recovery_fidelity_metric": fidelity_metric,
                "logit_recovery_fidelity_detail": fidelity_detail,
                "component_list": component_list,
                "marginal_contributions": {
                    f"L{layer}H{head}": value
                    for (layer, head), value in sorted(contributions.items())
                    if value
                },
            },
            evidence={
                "clean_prompt": clean_prompt,
                "corrupted_prompt": corrupted_prompt,
                "target_token": target_token,
                "baseline_logit_diff": round(baseline_diff, 4),
                "activation_cache_size": len(cache.head_activations),
                "baseline_note": baseline_note,
            },
            # Confidence was previously the fabricated fidelity, which could
            # not fall below 0.90. Nothing here calibrates a confidence in the
            # recovered circuit, so none is claimed. The graph score is the
            # circuit's measured recovery, which is what was measured.
            confidence=0.0,
            graph={
                "nodes": nodes,
                "edges": edges,
                "score": circuit_score,
                "score_meaning": "target-logit recovery of the pruned circuit",
            },
            provenance={
                **self.provenance_block(),
                # A plain string, deliberately. `evidence_policy.provenance_of`
                # resolves a dict provenance by looking for `source`/`kind`/
                # `type`/`status` inside it, and falls back to "unavailable" if
                # none is present. An earlier version of this file wrote
                # {"provenance": "live", ...}, which that reader cannot see: it
                # found no recognised key and reported the record as unverified.
                # A live run was therefore invisible to the evidence gate while
                # appearing to carry a provenance label. Same shape as
                # GPT2Adapter.get_logits, which returns `"provenance": "live"`.
                # "live" means measurements were taken from loaded weights, which
                # is true whenever the sweep ran -- and every retained edge's
                # weight is a measured marginal contribution, not a constant.
                "source": "live" if total_evaluations else "unavailable",
                "provenance": "live" if total_evaluations else "unavailable",
                "measured": bool(total_evaluations),
                "candidates_evaluated": total_evaluations,
                "search_space": config.search_space,
                "metric": "greedy_reverse_topological_edge_pruning",
                "paper_citation": "Conmy et al. 2023: Automatic Circuit Discovery",
                "logit_recovery_fidelity": (
                    (f"measured at {logit_recovery_fidelity} of the "
                     "clean-minus-corrupted logit gap")
                    if fidelity_detail.get("measured") else
                    f"not measured: {fidelity_detail.get('reason')}"),
            },
        )
