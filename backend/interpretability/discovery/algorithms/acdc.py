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
    required_capabilities=["patch_head_output", "capture_head_outputs", "get_logits"],
    estimated_runtime="10s-5m",
    search_space="attention_heads, mlps",
    output_schema="DiscoveryReport"
))
class ACDCAlgorithm(DiscoveryAlgorithm):
    """Greedy reverse-topological edge pruning over cached activations.

    The intervention is head-level throughout: the clean run's output vector for
    a candidate head is written into the corrupted run, and the recovery of the
    target token's logit is measured. Edge weights and confidences in the
    reported graph are those measured recoveries.

    What is real: activation capture, the pruning sweep, the resulting component
    set, and the per-edge recovery fractions -- all from the adapter.

    What is not, when it does not run: the whole-circuit fidelity. Recovering the
    clean logit difference with the entire pruned circuit at once needs
    multi-site injection, which the single-head adapter primitive cannot express,
    so fidelity is reported as unmeasured rather than approximated. See `run()`.

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
        """
        import torch

        scores = []
        for prompt in (clean_prompt, corrupted_prompt):
            inputs = self.adapter._tokenizer(prompt, return_tensors="pt").to(
                self.adapter._model.device
            )
            with torch.no_grad():
                logits = self.adapter._model(**inputs).logits[0, -1, :]
            scores.append(float(logits[target_id]))
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

    def run(self, dataset: Dict[str, Any], config: Optional[DiscoveryAlgorithmConfig] = None) -> DiscoveryReport:
        """Runs greedy reverse-topological edge pruning ACDC search."""
        t0 = time.time()

        if not isinstance(config, ACDCConfig):
            config = ACDCConfig()

        threshold = config.threshold

        clean_prompt = dataset.get("clean", "John gave a drink to Mary")
        corrupted_prompt = dataset.get("corrupted", "John gave a drink to John")
        target_token = dataset.get("target_token", " Mary")

        num_layers = self.adapter.spec.num_layers if self.adapter else 12
        num_heads = self.adapter.spec.num_heads if self.adapter else 12

        # 1. Populate Activation Cache
        self._capture_error = None
        self._head_vectors = {}
        cache = self.build_activation_cache(clean_prompt, corrupted_prompt, num_layers, num_heads)

        # 2. Define Search Candidate Edges in Reverse Topological Order
        candidate_edges: List[Tuple[int, int]] = []
        for layer in reversed(range(num_layers)):
            for head in range(num_heads):
                candidate_edges.append((layer, head))

        pruned_components: Set[Tuple[int, int]] = set()
        retained_components: Set[Tuple[int, int]] = set()
        recoveries: Dict[Tuple[int, int], float] = {}

        # Baseline metric: the target token's logit on the clean prompt minus the
        # same token's logit on the corrupted prompt. That is the gap the circuit
        # is supposed to explain.
        #
        # It used to be `abs(clean_top_logit - corrupted_top_logit)`, comparing
        # whichever token each prompt happened to rank first. On the IOI pair
        # those are different tokens (" Mary" and " John"), so the "difference"
        # was between two unrelated quantities, and dividing by it scaled every
        # candidate's effect by a meaningless number. Whatever the pruning
        # selected, the divisor did not mean anything.
        target_id = self._target_token_id(target_token)
        baseline_note: Optional[str] = None
        baseline_diff = 0.0
        clean_score: Optional[float] = None
        corr_score: Optional[float] = None
        if target_id is None:
            baseline_note = (
                f"target_token {target_token!r} is not a single token, so the "
                f"logit difference is undefined; no pruning was performed"
            )
        else:
            clean_score, corr_score = self._score_tokens(
                clean_prompt, corrupted_prompt, target_id
            )
            baseline_diff = clean_score - corr_score
            if abs(baseline_diff) < 1e-6:
                baseline_note = (
                    "the clean and corrupted prompts give the target token "
                    "identical logits, so there is no gap to explain and no "
                    "candidate can be ranked; no pruning was performed"
                )

        # 3. Greedy Reverse-Topological Search Loop
        #
        # The intervention is now a genuine head-level patch: the clean run's
        # output vector for that head is written into the corrupted run, and the
        # recovery of the target logit is measured.
        #
        # It previously called `patch_activation(layer=layer, neuron_index=head,
        # patch_value=clean_act_val)`. That writes
        # `transformer.h[layer].mlp` output at index `head` -- MLP neuron `head`
        # of 3072 -- so every candidate "head" was an MLP neuron, the patch
        # succeeded (3072 > 12), and a plausible delta came back. The component
        # set this loop selected was therefore chosen on MLP noise while being
        # labelled L{layer}H{head}, and the fidelity number later computed over
        # it via `circuit_fidelity` was real -- a real measurement of a
        # noise-selected circuit. That combination is worse than either failure
        # alone, because the output looked valid.
        total_evaluations = 0
        if target_id is not None and baseline_note is None:
            for layer, head in candidate_edges:
                total_evaluations += 1
                vector = self._head_vectors.get((layer, head))
                if vector is None:
                    pruned_components.add((layer, head))
                    continue

                patch_res = self.adapter.patch_head_output(
                    prompt=corrupted_prompt,
                    layer=layer,
                    head_index=head,
                    patch_vector=vector,
                    score_token_id=target_id,
                )
                recovery = (patch_res.patched_logit - corr_score) / baseline_diff
                recoveries[(layer, head)] = round(recovery, 4)

                # If the recovered fraction is BELOW threshold, this head is not
                # carrying the effect and can be pruned.
                if abs(recovery) < threshold:
                    pruned_components.add((layer, head))
                else:
                    retained_components.add((layer, head))
        else:
            total_evaluations = 0

        # An empty retained set is reported as empty. It previously injected
        # L9H9 and L10H0 "to ensure the top critical heads are retained", with a
        # comment naming them -- i.e. when the search found nothing, the search's
        # answer was replaced by the published answer. A circuit that measures no
        # surviving head is a finding.

        # 4. Construct Reconstructed Graph
        nodes = [{"id": "T_0", "type": "Token", "label": clean_prompt}]
        edges = []
        last_node = "T_0"

        sorted_retained = sorted(list(retained_components), key=lambda x: (x[0], x[1]))
        for layer, head in sorted_retained:
            node_id = f"H_L{layer}_H{head}"
            nodes.append({
                "id": node_id,
                "type": "Head",
                "label": f"L{layer}H{head} (Retained)"
            })
            # Edge weight and confidence are the measured recovery fraction for
            # this head. They were a literal `weight: min(0.99, act_val)` (the
            # head's L2 norm, so ~0.6, carrying no meaning as an edge weight) and
            # a literal `confidence: 0.95` on every edge.
            measured = recoveries.get((layer, head))
            edges.append({
                "source": last_node,
                "target": node_id,
                "weight": measured,
                "confidence": measured,
                "measured_from": "clean-to-corrupted head-output patch recovery",
            })
            last_node = node_id

        # Prediction Output Node
        nodes.append({"id": "P_0", "type": "Prediction", "label": target_token})
        # Was `weight: 1.0, confidence: 1.0` -- an assertion of total certainty
        # on the edge completing the circuit, from no measurement. The whole-graph
        # recovery is measured below by the fidelity path; until it runs, this
        # edge is unmeasured and says so.
        edges.append({
            "source": last_node,
            "target": "P_0",
            "weight": None,
            "confidence": None,
            "reason": (
                "Whole-circuit recovery is measured by the fidelity pass, not "
                "here. The previous value of 1.0 was a literal, on the one edge "
                "that would most reward a fabricated circuit."
            ),
        })

        circuit_score = round(len(retained_components) / max(1, len(candidate_edges)), 3)
        # Real fidelity: inject the retained heads into the corrupted run and
        # compare the recovered logit difference against the clean-minus-
        # corrupted gap. The previous expression
        # (0.90 + 0.09 * (1 - circuit_score)) was a rescaling of the pruning
        # ratio into a number named "logit recovery fidelity" that could not
        # fall below 0.90 -- so it reported success no matter how little of the
        # circuit survived.
        #
        # Two metrics, never confused. The bundled datasets name a target
        # token but not the IO/subject pair a logit *difference* needs, so
        # the gap-based `circuit_fidelity` cannot run on them -- and for a
        # long time that meant fidelity was always unmeasured on the standard
        # path. When only the target token is known, fidelity is the
        # target-logit recovery from `circuit_fidelity_target`, recorded
        # under its own metric name.
        logit_recovery_fidelity: Optional[float] = None
        fidelity_metric = "unmeasured"
        fidelity_detail: Dict[str, Any] = {
            "measured": False,
            "reason": "no token ids available to score recovery against",
        }
        if dataset.get("io_id") is not None and dataset.get("subject_id") is not None:
            try:
                from backend.interpretability.discovery.live_measure import (
                    circuit_fidelity,
                )
                detail = circuit_fidelity(
                    clean_prompt, corrupted_prompt,
                    int(dataset["io_id"]), int(dataset["subject_id"]),
                    set(retained_components))
                fidelity_detail = detail
                logit_recovery_fidelity = detail.get("fidelity")
                fidelity_metric = "logit-difference recovery"
            except Exception as exc:
                fidelity_detail = {
                    "measured": False,
                    "reason": f"fidelity measurement failed: {exc}",
                }
        elif target_id is not None and retained_components:
            try:
                from backend.interpretability.discovery.live_measure import (
                    circuit_fidelity_target,
                )
                detail = circuit_fidelity_target(
                    clean_prompt, corrupted_prompt, target_id,
                    set(retained_components))
                fidelity_detail = detail
                logit_recovery_fidelity = detail.get("fidelity")
                fidelity_metric = "target-logit recovery"
            except Exception as exc:
                fidelity_detail = {
                    "measured": False,
                    "reason": f"fidelity measurement failed: {exc}",
                }
        runtime_ms = (time.time() - t0) * 1000

        # Component List for Discovery Evaluation
        component_list = [f"L{l}H{h}" for l, h in retained_components]

        return DiscoveryReport(
            algorithm="acdc",
            dataset_id=dataset.get("id", "ioi_0001"),
            model_id=self.adapter.spec.model_id if self.adapter else "gpt2",
            runtime_ms=round(runtime_ms, 2),
            statistics={
                "total_candidate_components": len(candidate_edges),
                "retained_components": len(retained_components),
                "pruned_components": len(pruned_components),
                "pruning_ratio": round(len(pruned_components) / max(1, len(candidate_edges)), 3),
                "patch_threshold": threshold,
                "total_evaluations": total_evaluations,
                "logit_recovery_fidelity": logit_recovery_fidelity,
                "logit_recovery_fidelity_measured":
                    bool(fidelity_detail.get("measured")),
                "logit_recovery_fidelity_metric": fidelity_metric,
                "logit_recovery_fidelity_detail": fidelity_detail,
                "pruning_ratio": round(len(retained_components) / max(1, len(candidate_edges)), 3),
                "component_list": component_list # Added for Phase 39.2
            },
            evidence={
                "clean_prompt": clean_prompt,
                "corrupted_prompt": corrupted_prompt,
                "target_token": target_token,
                "baseline_logit_diff": round(baseline_diff, 4),
                "activation_cache_size": len(cache.head_activations),
            },
            # Confidence was previously the fabricated fidelity, which could
            # not fall below 0.90. Nothing here calibrates a confidence in the
            # recovered circuit, so none is claimed. The graph score is the
            # pruning ratio, which is what was actually measured.
            confidence=0.0,
            graph={
                "nodes": nodes,
                "edges": edges,
                "score": circuit_score,
                "score_meaning": "pruning_ratio_retained_over_candidate"
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
                # confidence is a measured recovery fraction, not a constant. It
                # does not mean the whole-circuit fidelity was computed; that is
                # tracked separately by `logit_recovery_fidelity_measured` and
                # spelled out in `logit_recovery_fidelity` below. Conflating the
                # two made a run that measured 144 heads report `unavailable`.
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
            }
        )
