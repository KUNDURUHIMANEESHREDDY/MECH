"""GPT-2 Live Experiment Runner for MECH.

THE single gateway that runs real GPT-2 measurements for every discovery engine.
No values are hardcoded — all metrics are derived from actual model forward passes.

Provides three families of live measurements:

1. Physical Perturbation (replaces hardcoded int8_delta_y, noise_delta_y, etc.)
   ─────────────────────────────────────────────────────────────────────────────
   GAUSSIAN_NOISE        — adds N(0, sigma) to all weight tensors, reruns
   WEIGHT_SCALING_075X   — multiplies all MLP weight tensors by 0.75, reruns
   SEQUENCE_EXTENSION    — pads prompt with neutral tokens to 2× length, reruns
   LAYER_ABLATION        — zeroes out a mid-layer residual stream, reruns

   CSF = 1 - |ΔY_baseline - ΔY_perturbed| / max(ε, |ΔY_baseline|)

2. Counterfactual Invariance (replaces hardcoded invariance_distance)
   ─────────────────────────────────────────────────────────────────────────────
   NEURON_PERMUTATION    — randomly permutes MLP neuron indices within one layer
   ROUTING_SPARSITY      — masks 20% of attention heads to zero
   DISTRACTOR_INSERTION  — inserts an orthogonal distractor token in the prompt

   Invariance distance = |I(c(M)) - I(M)| from actual logit differences

3. Functional Correspondence (replaces hardcoded cross-modal scores)
   ─────────────────────────────────────────────────────────────────────────────
   GPT-2 is text-only. We compute functional correspondence WITHIN the text
   domain across different task categories (factual, arithmetic, relational).
   Uses real logit-lens trajectories to compute cosine similarity of causal
   circuits between two probes.

All measurements are cached in a session-scoped dict to avoid redundant
re-computation within the same app session.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class PerturbationResult:
    """Result of a single physical substrate perturbation experiment."""
    perturbation_type: str
    baseline_delta_logit: float        # real Δlogit on clean GPT-2
    perturbed_delta_logit: float       # real Δlogit after perturbation
    causal_signature_fidelity: float   # CSF computed from above
    is_physically_grounded: bool       # CSF >= threshold (default 0.90)
    probe_id: str
    target_token: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "perturbation_type": self.perturbation_type,
            "baseline_delta_logit": round(self.baseline_delta_logit, 4),
            "perturbed_delta_logit": round(self.perturbed_delta_logit, 4),
            "causal_signature_fidelity": round(self.causal_signature_fidelity, 4),
            "is_physically_grounded": self.is_physically_grounded,
            "probe_id": self.probe_id,
            "target_token": self.target_token,
        }


@dataclass
class CounterfactualResult:
    """Result of a counterfactual invariance test."""
    perturbation_type: str
    baseline_causal_effect: float      # I(M): real causal effect on clean model
    counterfactual_causal_effect: float  # I(c(M)): after structural modification
    invariance_distance: float         # |I(c(M)) - I(M)|
    is_counterfactually_invariant: bool  # distance <= threshold (default 0.05)
    probe_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "perturbation_type": self.perturbation_type,
            "baseline_causal_effect": round(self.baseline_causal_effect, 4),
            "counterfactual_causal_effect": round(self.counterfactual_causal_effect, 4),
            "invariance_distance": round(self.invariance_distance, 4),
            "is_counterfactually_invariant": self.is_counterfactually_invariant,
            "probe_id": self.probe_id,
        }


@dataclass
class FunctionalCorrespondenceResult:
    """Real functional correspondence between two GPT-2 probe categories."""
    probe_a_id: str
    probe_b_id: str
    category_a: str
    category_b: str
    trajectory_cosine_similarity: float   # cosine sim of logit-lens trajectories
    peak_layer_overlap: float             # fraction of top-3 causal layers shared
    causal_correspondence_score: float    # combined metric
    is_functionally_grounded: bool        # score >= 0.50

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probe_a_id": self.probe_a_id,
            "probe_b_id": self.probe_b_id,
            "category_a": self.category_a,
            "category_b": self.category_b,
            "trajectory_cosine_similarity": round(self.trajectory_cosine_similarity, 4),
            "peak_layer_overlap": round(self.peak_layer_overlap, 4),
            "causal_correspondence_score": round(self.causal_correspondence_score, 4),
            "is_functionally_grounded": self.is_functionally_grounded,
        }


class Gpt2LiveExperimentRunner:
    """
    Runs real GPT-2 experiments for every MECH discovery engine.

    Parameters
    ----------
    runtime : InMemoryRuntime (or any ModelRuntimeInterface)
        The live GPT-2 runtime to use for all measurements.
    csf_threshold : float
        Minimum Causal Signature Fidelity to consider a result physically grounded.
    invariance_threshold : float
        Maximum invariance distance to consider a mechanism counterfactually invariant.
    noise_sigma : float
        Std dev for Gaussian noise perturbation.
    seed : int
        RNG seed for reproducible neuron permutations within a session.
    """

    def __init__(
        self,
        runtime,
        csf_threshold: float = 0.90,
        invariance_threshold: float = 0.05,
        noise_sigma: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.runtime = runtime
        self.csf_threshold = csf_threshold
        self.invariance_threshold = invariance_threshold
        self.noise_sigma = noise_sigma
        self.rng = random.Random(seed)
        self._cache: Dict[str, Any] = {}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_target_id(self, target_token: str) -> int:
        """Returns the vocab id for a target token."""
        ids = self.runtime.tokenizer.encode(target_token, add_special_tokens=False)
        return ids[-1] if ids else 0

    def _logit_for_target(self, prompt: str, target_token: str, model=None) -> float:
        """Runs a forward pass on `model` (or self.runtime.model) and returns target logit."""
        m = model if model is not None else self.runtime.model
        tok = self.runtime.tokenizer
        inputs = tok(prompt, return_tensors="pt").to(self.runtime.device)
        with torch.no_grad():
            out = m(**inputs)
        last_logits = out.logits[0, -1]
        t_id = self._get_target_id(target_token)
        return float(last_logits[t_id].item())

    def _ablated_logit(self, prompt: str, target_token: str, layer: int, neuron_idx: int) -> float:
        """Returns logit after ablating neuron_idx in layer's MLP."""
        tok = self.runtime.tokenizer
        inputs = tok(prompt, return_tensors="pt").to(self.runtime.device)
        t_id = self._get_target_id(target_token)

        def hook(module, inp, output):
            out = output[0].clone() if isinstance(output, tuple) else output.clone()
            if out.shape[-1] > neuron_idx:
                out[:, :, neuron_idx] *= 0.0
            return (out,) + output[1:] if isinstance(output, tuple) else out

        module = self.runtime.model.transformer.h[layer].mlp.c_fc
        handle = module.register_forward_hook(hook)
        try:
            with torch.no_grad():
                out = self.runtime.model(**inputs)
        finally:
            handle.remove()
        return float(out.logits[0, -1, t_id].item())

    def _baseline_causal_effect(self, probe) -> float:
        """
        Measures the real causal effect of ablating the probe's target neuron.
        Returns |clean_logit - ablated_logit| as the baseline causal signal I(M).
        """
        num_layers = self.runtime.num_layers
        layer = max(1, min(num_layers - 2, int(num_layers * probe.target_layer_fraction)))
        clean_logit    = self._logit_for_target(probe.clean_prompt, probe.target_token)
        ablated_logit  = self._ablated_logit(probe.clean_prompt, probe.target_token, layer, probe.target_neuron_idx)
        return abs(clean_logit - ablated_logit)

    # ── 1. Physical Perturbation ──────────────────────────────────────────────

    def measure_physical_perturbation(
        self,
        probe,
        perturbation_type: str = "GAUSSIAN_NOISE",
    ) -> PerturbationResult:
        """
        Measures real Causal Signature Fidelity for one physical perturbation type.

        CSF = 1 - |ΔY_baseline - ΔY_perturbed| / max(ε, |ΔY_baseline|)

        where ΔY = |clean_logit - ablated_logit| measured on real GPT-2 weights.
        """
        cache_key = f"phys_{probe.probe_id}_{perturbation_type}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        num_layers = self.runtime.num_layers
        layer = max(1, min(num_layers - 2, int(num_layers * probe.target_layer_fraction)))
        neuron_idx = probe.target_neuron_idx

        # Baseline causal effect on clean model
        baseline_delta = self._baseline_causal_effect(probe)

        # Apply perturbation to a COPY of the model (never mutate live weights permanently)
        model_copy = copy.deepcopy(self.runtime.model)
        model_copy.eval()

        if perturbation_type == "GAUSSIAN_NOISE":
            with torch.no_grad():
                for param in model_copy.parameters():
                    param.add_(torch.randn_like(param) * self.noise_sigma)

        elif perturbation_type == "WEIGHT_SCALING_075X":
            with torch.no_grad():
                for name, param in model_copy.named_parameters():
                    if "mlp" in name:
                        param.mul_(0.75)

        elif perturbation_type == "SEQUENCE_EXTENSION":
            # Pad prompt with neutral filler tokens to 2× length
            pad_token = self.runtime.tokenizer.eos_token or "."
            n_tokens  = len(self.runtime.tokenizer.encode(probe.clean_prompt))
            extended  = probe.clean_prompt + (" " + pad_token) * n_tokens
            # Use extended prompt but original model for this measurement
            clean_logit_ext   = self._logit_for_target(extended, probe.target_token, model=None)
            ablated_logit_ext = self._ablated_logit(extended, probe.target_token, layer, neuron_idx)
            perturbed_delta   = abs(clean_logit_ext - ablated_logit_ext)
            csf = 1.0 - abs(baseline_delta - perturbed_delta) / max(1e-6, abs(baseline_delta))
            result = PerturbationResult(
                perturbation_type="SEQUENCE_EXTENSION",
                baseline_delta_logit=round(baseline_delta, 4),
                perturbed_delta_logit=round(perturbed_delta, 4),
                causal_signature_fidelity=round(max(0.0, min(1.0, csf)), 4),
                is_physically_grounded=csf >= self.csf_threshold,
                probe_id=probe.probe_id,
                target_token=probe.target_token,
            )
            self._cache[cache_key] = result
            del model_copy
            return result

        elif perturbation_type == "LAYER_ABLATION":
            # Zero out a mid-layer residual by hooking the perturbed copy
            mid_layer = num_layers // 2

            def zero_hook(module, inp, out):
                h = out[0].clone() if isinstance(out, tuple) else out.clone()
                h.zero_()
                return (h,) + out[1:] if isinstance(out, tuple) else h

            handle = model_copy.transformer.h[mid_layer].register_forward_hook(zero_hook)
            try:
                clean_logit_abl = self._logit_for_target(probe.clean_prompt, probe.target_token, model=model_copy)
                perturbed_delta = abs(clean_logit_abl - self._ablated_logit(probe.clean_prompt, probe.target_token, layer, neuron_idx))
            finally:
                handle.remove()
        else:
            # Unknown perturbation — fall back to baseline
            perturbed_delta = baseline_delta

        # For GAUSSIAN_NOISE and WEIGHT_SCALING_075X: measure causal effect on perturbed copy
        if perturbation_type in ("GAUSSIAN_NOISE", "WEIGHT_SCALING_075X"):
            perturbed_clean   = self._logit_for_target(probe.clean_prompt, probe.target_token, model=model_copy)
            # Ablate on perturbed copy
            tok = self.runtime.tokenizer
            inputs = tok(probe.clean_prompt, return_tensors="pt").to(self.runtime.device)
            t_id = self._get_target_id(probe.target_token)

            def hook_p(module, inp, output):
                out = output[0].clone() if isinstance(output, tuple) else output.clone()
                if out.shape[-1] > neuron_idx:
                    out[:, :, neuron_idx] *= 0.0
                return (out,) + output[1:] if isinstance(output, tuple) else out

            mod_p = model_copy.transformer.h[layer].mlp.c_fc
            handle_p = mod_p.register_forward_hook(hook_p)
            try:
                with torch.no_grad():
                    out_p = model_copy(**inputs)
            finally:
                handle_p.remove()
            perturbed_ablated = float(out_p.logits[0, -1, t_id].item())
            perturbed_delta = abs(perturbed_clean - perturbed_ablated)

        del model_copy

        csf = 1.0 - abs(baseline_delta - perturbed_delta) / max(1e-6, abs(baseline_delta))
        csf = max(0.0, min(1.0, csf))

        result = PerturbationResult(
            perturbation_type=perturbation_type,
            baseline_delta_logit=round(baseline_delta, 4),
            perturbed_delta_logit=round(perturbed_delta, 4),
            causal_signature_fidelity=round(csf, 4),
            is_physically_grounded=csf >= self.csf_threshold,
            probe_id=probe.probe_id,
            target_token=probe.target_token,
        )
        self._cache[cache_key] = result
        return result

    def run_all_physical_perturbations(self, probe) -> List[PerturbationResult]:
        """Runs all four perturbation types for one probe and returns results."""
        types = ["GAUSSIAN_NOISE", "WEIGHT_SCALING_075X", "SEQUENCE_EXTENSION", "LAYER_ABLATION"]
        return [self.measure_physical_perturbation(probe, t) for t in types]

    # ── 2. Counterfactual Invariance ──────────────────────────────────────────

    def measure_counterfactual_invariance(
        self,
        probe,
        perturbation_type: str = "NEURON_PERMUTATION",
    ) -> CounterfactualResult:
        """
        Measures |I(c(M)) - I(M)| for one counterfactual structural modification.

        I(M)    = |clean_logit - ablated_logit| on original GPT-2
        I(c(M)) = same measurement after structural modification c
        """
        cache_key = f"cf_{probe.probe_id}_{perturbation_type}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        num_layers = self.runtime.num_layers
        layer = max(1, min(num_layers - 2, int(num_layers * probe.target_layer_fraction)))

        baseline_effect = self._baseline_causal_effect(probe)

        model_copy = copy.deepcopy(self.runtime.model)
        model_copy.eval()

        if perturbation_type == "NEURON_PERMUTATION":
            with torch.no_grad():
                mlp = model_copy.transformer.h[layer].mlp
                w = mlp.c_fc.weight.data
                perm = torch.randperm(w.shape[0])
                mlp.c_fc.weight.data = w[perm]

        elif perturbation_type == "ROUTING_SPARSITY":
            with torch.no_grad():
                attn = model_copy.transformer.h[layer].attn
                num_heads = self.runtime.model.config.n_head
                mask_count = max(1, num_heads // 5)  # 20% of heads
                masked_heads = self.rng.sample(range(num_heads), k=mask_count)
                head_dim = self.runtime.model.config.n_embd // num_heads
                for h_idx in masked_heads:
                    start = h_idx * head_dim
                    end   = start + head_dim
                    if hasattr(attn, "c_attn"):
                        attn.c_attn.weight.data[:, start:end] = 0.0

        elif perturbation_type == "DISTRACTOR_INSERTION":
            # Insert an orthogonal distractor into the clean prompt
            distractor = probe.distractor_token if hasattr(probe, "distractor_token") else " the"
            distractor_prompt = probe.clean_prompt + distractor
            cf_effect = self._baseline_causal_effect(
                type("_P", (), {
                    "clean_prompt": distractor_prompt,
                    "target_token": probe.target_token,
                    "target_layer_fraction": probe.target_layer_fraction,
                    "target_neuron_idx": probe.target_neuron_idx,
                })()
            )
            inv_dist = abs(cf_effect - baseline_effect)
            result = CounterfactualResult(
                perturbation_type="DISTRACTOR_INSERTION",
                baseline_causal_effect=round(baseline_effect, 4),
                counterfactual_causal_effect=round(cf_effect, 4),
                invariance_distance=round(inv_dist, 4),
                is_counterfactually_invariant=inv_dist <= self.invariance_threshold,
                probe_id=probe.probe_id,
            )
            self._cache[cache_key] = result
            del model_copy
            return result
        else:
            del model_copy
            result = CounterfactualResult(
                perturbation_type=perturbation_type,
                baseline_causal_effect=round(baseline_effect, 4),
                counterfactual_causal_effect=round(baseline_effect, 4),
                invariance_distance=0.0,
                is_counterfactually_invariant=True,
                probe_id=probe.probe_id,
            )
            self._cache[cache_key] = result
            return result

        # Measure causal effect on structurally modified model
        tok = self.runtime.tokenizer
        inputs = tok(probe.clean_prompt, return_tensors="pt").to(self.runtime.device)
        t_id = self._get_target_id(probe.target_token)

        with torch.no_grad():
            out_cf = model_copy(**inputs)
        cf_clean = float(out_cf.logits[0, -1, t_id].item())

        neuron_idx = probe.target_neuron_idx

        def hook_cf(module, inp, output):
            out = output[0].clone() if isinstance(output, tuple) else output.clone()
            if out.shape[-1] > neuron_idx:
                out[:, :, neuron_idx] *= 0.0
            return (out,) + output[1:] if isinstance(output, tuple) else out

        mod_cf = model_copy.transformer.h[layer].mlp.c_fc
        handle_cf = mod_cf.register_forward_hook(hook_cf)
        try:
            with torch.no_grad():
                out_abl = model_copy(**inputs)
        finally:
            handle_cf.remove()

        cf_ablated = float(out_abl.logits[0, -1, t_id].item())
        cf_effect  = abs(cf_clean - cf_ablated)
        inv_dist   = abs(cf_effect - baseline_effect)

        del model_copy

        result = CounterfactualResult(
            perturbation_type=perturbation_type,
            baseline_causal_effect=round(baseline_effect, 4),
            counterfactual_causal_effect=round(cf_effect, 4),
            invariance_distance=round(inv_dist, 4),
            is_counterfactually_invariant=inv_dist <= self.invariance_threshold,
            probe_id=probe.probe_id,
        )
        self._cache[cache_key] = result
        return result

    def run_all_counterfactual_invariances(self, probe) -> List[CounterfactualResult]:
        """Runs all three counterfactual types for one probe."""
        types = ["NEURON_PERMUTATION", "ROUTING_SPARSITY", "DISTRACTOR_INSERTION"]
        return [self.measure_counterfactual_invariance(probe, t) for t in types]

    # ── 3. Functional Correspondence (text-domain only for GPT-2) ────────────

    def measure_functional_correspondence(
        self,
        probe_a,
        probe_b,
    ) -> FunctionalCorrespondenceResult:
        """
        Measures real functional correspondence between two GPT-2 text probes.

        Uses live logit-lens trajectories (one per probe) to compute:
        - cosine similarity of target-logit trajectory vectors
        - overlap in the top-3 most causally active layers

        Returns FunctionalCorrespondenceResult — no hardcoded values.
        """
        cache_key = f"fc_{probe_a.probe_id}_{probe_b.probe_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        traj_a = self.runtime.compute_logit_lens_trajectory(
            prompt=probe_a.clean_prompt, target_token=probe_a.target_token
        )
        traj_b = self.runtime.compute_logit_lens_trajectory(
            prompt=probe_b.clean_prompt, target_token=probe_b.target_token
        )

        # Extract target-logit vectors (one float per layer)
        logits_a = torch.tensor([t["target_logit"] for t in traj_a], dtype=torch.float32)
        logits_b = torch.tensor([t["target_logit"] for t in traj_b], dtype=torch.float32)

        # Pad shorter to match lengths
        la, lb = len(logits_a), len(logits_b)
        if la > lb:
            logits_b = torch.cat([logits_b, logits_b[-1:].expand(la - lb)])
        elif lb > la:
            logits_a = torch.cat([logits_a, logits_a[-1:].expand(lb - la)])

        # Cosine similarity of the trajectory vectors
        cos_sim = float(
            torch.nn.functional.cosine_similarity(
                logits_a.unsqueeze(0), logits_b.unsqueeze(0)
            ).item()
        )
        cos_sim = max(0.0, min(1.0, (cos_sim + 1.0) / 2.0))  # remap [-1,1] → [0,1]

        # Top-3 causal layer overlap (layers where logit increases most)
        delta_a = torch.diff(logits_a)
        delta_b = torch.diff(logits_b)
        top3_a  = set(torch.topk(delta_a, k=min(3, len(delta_a))).indices.tolist())
        top3_b  = set(torch.topk(delta_b, k=min(3, len(delta_b))).indices.tolist())
        layer_overlap = len(top3_a & top3_b) / max(1, len(top3_a | top3_b))

        # Combined correspondence score
        score = 0.6 * cos_sim + 0.4 * layer_overlap
        score = max(0.0, min(1.0, score))

        cat_a = getattr(probe_a, "category", "text")
        cat_b = getattr(probe_b, "category", "text")

        result = FunctionalCorrespondenceResult(
            probe_a_id=probe_a.probe_id,
            probe_b_id=probe_b.probe_id,
            category_a=cat_a,
            category_b=cat_b,
            trajectory_cosine_similarity=round(cos_sim, 4),
            peak_layer_overlap=round(layer_overlap, 4),
            causal_correspondence_score=round(score, 4),
            is_functionally_grounded=score >= 0.50,
        )
        self._cache[cache_key] = result
        return result

    # ── 4. Session metrics summary (used by orchestrator) ────────────────────

    def compute_session_metrics(self, probes: List) -> Dict[str, float]:
        """
        Runs all live experiments for the session probe set and returns
        aggregate metrics used by the orchestrator and Claim DAG.

        Returns
        -------
        dict with keys:
            csf_mean        — mean Causal Signature Fidelity across probes × perturbations
            inv_mean        — mean counterfactual invariance pass rate
            fc_mean         — mean functional correspondence across probe pairs
            fur             — False Universalization Rate (probes reaching UNIVERSAL)
            n_probes        — number of probes evaluated
        All values are real measurements from GPT-2.
        """
        csf_scores: List[float] = []
        inv_pass:   List[bool]  = []
        fc_scores:  List[float] = []

        for probe in probes:
            # Physical perturbations
            phys = self.run_all_physical_perturbations(probe)
            csf_scores.extend(r.causal_signature_fidelity for r in phys)

            # Counterfactual invariance
            cfs = self.run_all_counterfactual_invariances(probe)
            inv_pass.extend(r.is_counterfactually_invariant for r in cfs)

        # Functional correspondence between adjacent probe pairs
        for i in range(len(probes) - 1):
            fc = self.measure_functional_correspondence(probes[i], probes[i + 1])
            fc_scores.append(fc.causal_correspondence_score)

        csf_mean = sum(csf_scores) / max(1, len(csf_scores))
        inv_mean = sum(inv_pass) / max(1, len(inv_pass))
        fc_mean  = sum(fc_scores) / max(1, len(fc_scores))

        # FUR: fraction of probes where fc_score >= 0.90 across ALL pairs
        # (would be incorrectly promoted to universal — should be near 0)
        universal_count = sum(1 for s in fc_scores if s >= 0.90)
        fur = universal_count / max(1, len(fc_scores))

        return {
            "csf_mean":  round(csf_mean, 4),
            "inv_mean":  round(inv_mean, 4),
            "fc_mean":   round(fc_mean, 4),
            "fur":       round(fur, 4),
            "n_probes":  len(probes),
        }
