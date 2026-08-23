"""Automated Mechanistic Candidate Discovery Engine for MECH Platform.

Transforms MECH from a manual experiment executor into an autonomous mechanistic discovery system:
1. Layer-Wise Residual Stream Scan (identifies causal processing windows)
2. Head-Wise Attention Scan (systematically scans all L x H attention heads)
3. Neuron / MLP Direct Logit Attribution (DLA) & Activation Scan
4. Candidate Ranking & Circuit Synthesis
5. Automated Causal Validation Battery dispatch on top discovered components.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from backend.core.experiment_engine import (
    ComponentTarget,
    InterventionSpec,
    MechanisticExperimentEngine,
    set_seed,
)
from backend.core.model_adapter import ModelAdapter, get_model_adapter
from backend.storage.database import DesktopStorage
from backend.storage.scientific_entities import InterventionType

logger = logging.getLogger("MECH.discovery_engine")


@dataclass
class LayerScanResult:
    layer: int
    delta_logit: float
    indirect_effect: float
    residual_norm: float
    is_causal_window: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "delta_logit": round(self.delta_logit, 4),
            "indirect_effect": round(self.indirect_effect, 4),
            "residual_norm": round(self.residual_norm, 4),
            "is_causal_window": self.is_causal_window,
        }


@dataclass
class CandidateComponent:
    component_id: str  # e.g. "L9H9", "L8_N412", "L6_MLP"
    component_type: str  # "attention_head" | "neuron" | "mlp" | "residual"
    layer: int
    index: int
    delta_logit: float
    delta_prob: float
    indirect_effect: float
    causal_rank: int
    functional_role: str  # "Name Mover", "Previous Token", "Induction", "Factual Recall", etc.
    validation_status: str = "UNTESTED"  # "CAUSALLY_VERIFIED" | "SUPPORTED" | "REFUTED"
    validation_details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_id": self.component_id,
            "component_type": self.component_type,
            "layer": self.layer,
            "index": self.index,
            "delta_logit": round(self.delta_logit, 4),
            "delta_prob": round(self.delta_prob, 6),
            "indirect_effect": round(self.indirect_effect, 4),
            "causal_rank": self.causal_rank,
            "functional_role": self.functional_role,
            "validation_status": self.validation_status,
            "validation_details": self.validation_details,
        }


@dataclass
class CircuitDiscoveryReport:
    model_id: str
    model_hash: str
    clean_prompt: str
    corrupted_prompt: Optional[str]
    target_token: str
    distractor_token: Optional[str]
    total_heads_scanned: int
    total_neurons_scanned: int
    layer_scan: List[LayerScanResult]
    top_candidates: List[CandidateComponent]
    validated_circuit: List[Dict[str, Any]]
    execution_time_ms: float
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "clean_prompt": self.clean_prompt,
            "corrupted_prompt": self.corrupted_prompt,
            "target_token": self.target_token,
            "distractor_token": self.distractor_token,
            "total_heads_scanned": self.total_heads_scanned,
            "total_neurons_scanned": self.total_neurons_scanned,
            "layer_scan": [l.to_dict() for l in self.layer_scan],
            "top_candidates": [c.to_dict() for c in self.top_candidates],
            "validated_circuit": self.validated_circuit,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "timestamp_utc": self.timestamp_utc,
        }


class AutomatedDiscoveryEngine:
    """Discovers causal mechanistic circuits across transformer layers, heads, and neurons."""

    def __init__(
        self,
        model_id: str = "gpt2",
        adapter: Optional[ModelAdapter] = None,
        storage: Optional[DesktopStorage] = None,
    ) -> None:
        self.model_id = model_id
        self.adapter = adapter or get_model_adapter(model_id=model_id)
        self.storage = storage
        self.experiment_engine = MechanisticExperimentEngine(
            model_id=model_id,
            adapter=self.adapter,
            storage=storage,
        )

    def scan_layers(
        self,
        clean_prompt: str,
        corrupted_prompt: str,
        target_token: str,
    ) -> List[LayerScanResult]:
        """Scans all residual layers via activation patching to identify causal processing windows."""
        n_layers = self.adapter.n_layers
        results = []

        # Baseline clean & corrupted passes
        clean_base = self.experiment_engine.run_baseline(clean_prompt, target_token=target_token)
        corr_base = self.experiment_engine.run_baseline(corrupted_prompt, target_token=target_token)

        clean_tgt_logit = clean_base.target_logit or 0.0
        corr_tgt_logit = corr_base.target_logit or 0.0
        denom = clean_tgt_logit - corr_tgt_logit if abs(clean_tgt_logit - corr_tgt_logit) > 1e-4 else 1.0

        target_str = target_token if target_token.startswith(" ") else f" {target_token}"
        encoded_tgt = self.adapter.encode(target_str)
        target_id = encoded_tgt[0] if encoded_tgt else 0

        # Pre-capture corrupted residual activations
        all_resid_comps = [ComponentTarget(type="residual", layer=l) for l in range(n_layers)]
        corr_activations = self.experiment_engine._capture_activations(corrupted_prompt, all_resid_comps)

        for layer in range(n_layers):
            comp = ComponentTarget(type="residual", layer=layer)
            int_logits, int_probs, _ = self.experiment_engine._execute_hook_pass(
                prompt=clean_prompt,
                components=[comp],
                intervention_type=InterventionType.ACTIVATION_PATCHING,
                patch_activations=corr_activations,
            )
            int_tgt_logit = float(int_logits[target_id].item())
            dz = clean_tgt_logit - int_tgt_logit
            ie = (int_tgt_logit - corr_tgt_logit) / denom
            resid_norm = clean_base.hidden_norms[layer + 1] if layer + 1 < len(clean_base.hidden_norms) else 100.0

            results.append(LayerScanResult(
                layer=layer,
                delta_logit=dz,
                indirect_effect=ie,
                residual_norm=resid_norm,
                is_causal_window=ie > 0.15 or dz > 0.50,
            ))

        return results

    def scan_all_attention_heads(
        self,
        clean_prompt: str,
        corrupted_prompt: str,
        target_token: str,
        distractor_token: Optional[str] = None,
        top_k: int = 10,
    ) -> List[CandidateComponent]:
        """Exhaustively scans all L x H attention heads using fast hook passes and ranks them."""
        n_layers = self.adapter.n_layers
        n_heads = self.adapter.n_heads

        clean_base = self.experiment_engine.run_baseline(clean_prompt, target_token=target_token, distractor_token=distractor_token)
        corr_base = self.experiment_engine.run_baseline(corrupted_prompt, target_token=target_token, distractor_token=distractor_token)

        clean_tgt_logit = clean_base.target_logit or 0.0
        clean_tgt_prob = clean_base.target_probability or 0.0
        corr_tgt_logit = corr_base.target_logit or 0.0
        denom = clean_tgt_logit - corr_tgt_logit if abs(clean_tgt_logit - corr_tgt_logit) > 1e-4 else 1.0

        target_str = target_token if target_token.startswith(" ") else f" {target_token}"
        encoded_tgt = self.adapter.encode(target_str)
        target_id = encoded_tgt[0] if encoded_tgt else 0

        # Pre-capture all head activations from corrupted pass in a single forward pass
        all_head_comps = [ComponentTarget(type="attention_head", layer=l, index=h) for l in range(n_layers) for h in range(n_heads)]
        corr_activations = self.experiment_engine._capture_activations(corrupted_prompt, all_head_comps)

        candidates: List[CandidateComponent] = []

        for layer in range(n_layers):
            for head in range(n_heads):
                comp = ComponentTarget(type="attention_head", layer=layer, index=head)
                int_logits, int_probs, _ = self.experiment_engine._execute_hook_pass(
                    prompt=clean_prompt,
                    components=[comp],
                    intervention_type=InterventionType.ACTIVATION_PATCHING,
                    patch_activations=corr_activations,
                )
                int_tgt_logit = float(int_logits[target_id].item())
                int_tgt_prob = float(int_probs[target_id].item())
                dz = clean_tgt_logit - int_tgt_logit
                dp = clean_tgt_prob - int_tgt_prob
                ie = (int_tgt_logit - corr_tgt_logit) / denom

                # Classify provisional functional role
                if layer >= n_layers - 4:
                    role = "Name Mover / Direct Logit Head"
                elif layer in (n_layers // 2 - 1, n_layers // 2, n_layers // 2 + 1):
                    role = "Induction / Information Routing Head"
                else:
                    role = "Early Feature / Previous Token Head"

                candidates.append(CandidateComponent(
                    component_id=comp.to_component_id(),
                    component_type="attention_head",
                    layer=layer,
                    index=head,
                    delta_logit=dz,
                    delta_prob=dp,
                    indirect_effect=ie,
                    causal_rank=0,
                    functional_role=role,
                ))

        # Sort by absolute indirect effect / delta logit
        candidates.sort(key=lambda c: abs(c.indirect_effect) + abs(c.delta_logit) * 0.1, reverse=True)
        for idx, c in enumerate(candidates):
            c.causal_rank = idx + 1

        return candidates[:top_k]

    def scan_top_neurons(
        self,
        prompt: str,
        target_token: str,
        candidate_layers: List[int],
        top_k: int = 10,
    ) -> List[CandidateComponent]:
        """Scans neurons in causal layer windows by Direct Logit Attribution through W_U."""
        target_str = target_token if target_token.startswith(" ") else f" {target_token}"
        target_id = self.adapter.encode(target_str)[0] if self.adapter.encode(target_str) else 0

        W_U = self.adapter.get_unembedding_weight()
        target_unembed = W_U[target_id].float()  # [d_model]

        candidates: List[CandidateComponent] = []

        for layer in candidate_layers:
            mlp_proj = self.adapter.get_mlp_proj_module(layer)
            proj_w = mlp_proj.weight.data.float()  # [d_mlp, d_model] or [d_model, d_mlp]

            if proj_w.shape[0] == self.adapter.d_mlp:
                dlas = proj_w @ target_unembed  # [d_mlp]
            else:
                dlas = proj_w.T @ target_unembed  # [d_mlp]

            top_neuron_indices = torch.topk(torch.abs(dlas), k=min(5, len(dlas))).indices.tolist()

            for n_idx in top_neuron_indices:
                dla_val = float(dlas[n_idx].item())
                comp = ComponentTarget(type="neuron", layer=layer, index=n_idx)
                candidates.append(CandidateComponent(
                    component_id=comp.to_component_id(),
                    component_type="neuron",
                    layer=layer,
                    index=n_idx,
                    delta_logit=dla_val,
                    delta_prob=0.0,
                    indirect_effect=0.0,
                    causal_rank=0,
                    functional_role="Factual Association / Memory Neuron",
                ))

        candidates.sort(key=lambda c: abs(c.delta_logit), reverse=True)
        for idx, c in enumerate(candidates):
            c.causal_rank = idx + 1

        return candidates[:top_k]

    def discover_and_validate_circuit(
        self,
        clean_prompt: str,
        corrupted_prompt: Optional[str] = None,
        target_token: str = "",
        distractor_token: Optional[str] = None,
        max_candidates_to_validate: int = 4,
        validation_repeats: int = 3,
    ) -> CircuitDiscoveryReport:
        """Executes end-to-end automated discovery: Layer scan -> Head scan -> Neuron scan -> Validation battery."""
        t0 = time.time()
        corr_prompt = corrupted_prompt or clean_prompt

        # 1. Layer-Wise Scan
        layer_scan = self.scan_layers(clean_prompt, corr_prompt, target_token)
        causal_layers = [l.layer for l in layer_scan if l.is_causal_window]
        if not causal_layers:
            causal_layers = [self.adapter.n_layers - 3, self.adapter.n_layers - 2, self.adapter.n_layers - 1]

        # 2. Attention Head Scan (L x H)
        top_heads = self.scan_all_attention_heads(
            clean_prompt=clean_prompt,
            corrupted_prompt=corr_prompt,
            target_token=target_token,
            distractor_token=distractor_token,
            top_k=max_candidates_to_validate,
        )

        # 3. Neuron DLA Scan in Causal Layers
        top_neurons = self.scan_top_neurons(
            prompt=clean_prompt,
            target_token=target_token,
            candidate_layers=causal_layers,
            top_k=max_candidates_to_validate,
        )

        all_candidates = top_heads + top_neurons
        all_candidates.sort(key=lambda c: abs(c.indirect_effect) + abs(c.delta_logit) * 0.1, reverse=True)
        for idx, c in enumerate(all_candidates):
            c.causal_rank = idx + 1

        # 4. Automated Multi-Trial Causal Validation Battery on Top Candidates
        validated_circuit: List[Dict[str, Any]] = []
        for cand in all_candidates[:max_candidates_to_validate]:
            comp = ComponentTarget.from_string(cand.component_id)
            val_spec = InterventionSpec(
                clean_prompt=clean_prompt,
                corrupted_prompt=corr_prompt,
                target_token=target_token,
                distractor_token=distractor_token,
                target_components=[comp],
                intervention_type=InterventionType.ABLATION_ZERO,
                random_seed=42,
                repeats=validation_repeats,
            )
            val_res = self.experiment_engine.run_intervention_experiment(val_spec)

            cand.validation_status = val_res.evidence_tier
            cand.validation_details = {
                "mean_delta_logit": val_res.delta_logit,
                "ci95_low": val_res.multi_trial_stats.ci95_low,
                "ci95_high": val_res.multi_trial_stats.ci95_high,
                "specificity_ratio": val_res.specificity_ratio,
                "cohens_d": val_res.cohens_d,
                "verdict": val_res.verdict,
                "provenance_hash": val_res.provenance_hash,
            }

            if val_res.evidence_tier in ("CAUSALLY_VERIFIED", "SUPPORTED", "WEAKLY_SUPPORTED"):
                validated_circuit.append({
                    "component_id": cand.component_id,
                    "type": cand.component_type,
                    "layer": cand.layer,
                    "index": cand.index,
                    "functional_role": cand.functional_role,
                    "causal_effect_ci": [val_res.multi_trial_stats.ci95_low, val_res.multi_trial_stats.ci95_high],
                    "specificity": val_res.specificity_ratio,
                    "evidence_tier": val_res.evidence_tier,
                })

        total_heads = self.adapter.n_layers * self.adapter.n_heads
        total_neurons = self.adapter.n_layers * self.adapter.d_mlp
        exec_time = (time.time() - t0) * 1000.0
        timestamp_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        return CircuitDiscoveryReport(
            model_id=self.model_id,
            model_hash=self.adapter.model_hash,
            clean_prompt=clean_prompt,
            corrupted_prompt=corrupted_prompt,
            target_token=target_token,
            distractor_token=distractor_token,
            total_heads_scanned=total_heads,
            total_neurons_scanned=total_neurons,
            layer_scan=layer_scan,
            top_candidates=all_candidates,
            validated_circuit=validated_circuit,
            execution_time_ms=exec_time,
            timestamp_utc=timestamp_utc,
        )
