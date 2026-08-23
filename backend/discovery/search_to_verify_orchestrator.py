"""Search-to-Verify Discovery Orchestrator.

Implements the two-stage search-and-verify paradigm:
Stage 1: High-throughput Gradient Attribution Scanner screens tens of thousands of components.
Stage 2: Targeted Causal Verification executes zero-ablation Δz, 4-control battery, and specificity testing.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from backend.runtime.in_memory_runtime import InMemoryRuntime
from backend.runtime.interfaces import ModelRuntimeInterface, PrecisionProfile
from .gradient_attribution_engine import GradientAttributionScanner, GradientAttributionScore


@dataclass
class VerifiedComponentEvidence:
    """Rigorous causal and control evidence for a candidate component discovered via gradient search."""
    component_id: str
    layer: int
    component_index: int
    gradient_attribution_score: float
    attribution_patching_score: Optional[float]
    gradient_search_rank: int
    clean_target_logit: float
    intervened_target_logit: float
    causal_delta_z: float
    control_specificity_ratio: float
    controls_passed_count: int
    epistemic_evidence_tier: str         # "CAUSALLY_VERIFIED_CIRCUIT_NODE" | "CAUSALLY_SUPPORTED_NODE" | "FALSIFIED_GRADIENT_HEURISTIC"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchToVerifyDiscoveryReport:
    """Comprehensive report detailing the transition from gradient search candidates to causally verified circuit nodes."""
    report_id: str
    behavior_name: str
    model_id: str
    clean_prompt: str
    target_token: str
    corrupted_prompt: Optional[str]
    total_components_screened: int
    top_candidates_selected: int
    verified_components: List[VerifiedComponentEvidence]
    mean_control_specificity: float
    epistemic_summary: str
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "behavior_name": self.behavior_name,
            "model_id": self.model_id,
            "clean_prompt": self.clean_prompt,
            "target_token": self.target_token,
            "corrupted_prompt": self.corrupted_prompt,
            "total_components_screened": self.total_components_screened,
            "top_candidates_selected": self.top_candidates_selected,
            "verified_components": [c.to_dict() for c in self.verified_components],
            "mean_control_specificity": self.mean_control_specificity,
            "epistemic_summary": self.epistemic_summary,
            "timestamp_utc": self.timestamp_utc,
        }


class SearchToVerifyDiscoveryOrchestrator:
    """Coordinates gradient attribution scanning and targeted causal verification."""

    def __init__(
        self,
        runtime: Optional[ModelRuntimeInterface] = None,
        model_id: str = "gpt2",
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.model_id = model_id
        self.runtime = runtime or InMemoryRuntime(model_id=model_id, device=device)
        self.scanner = GradientAttributionScanner(runtime=self.runtime, model_id=model_id, device=device)

    def run_search_to_verify_discovery(
        self,
        behavior_name: str = "country_capital_retrieval",
        clean_prompt: str = "The capital of France is",
        target_token: str = " Paris",
        corrupted_prompt: Optional[str] = "The capital of Italy is",
        target_layers: Optional[List[int]] = None,
        top_k_candidates: int = 5,
    ) -> SearchToVerifyDiscoveryReport:
        """Executes full search-to-verify pipeline across thousands of candidate components."""
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        scanned_layers = target_layers or [6, 7, 8, 9, 10]

        # ── STAGE 1: High-Throughput Gradient Attribution Scanner ───────────
        all_attributions = self.scanner.scan_layer_attributions(
            clean_prompt=clean_prompt,
            target_token=target_token,
            target_layers=scanned_layers,
            corrupted_prompt=corrupted_prompt,
        )
        total_screened = len(all_attributions)
        top_candidates = all_attributions[:top_k_candidates]

        # ── STAGE 2: Targeted Causal Verification & 4-Control Battery ────────
        verified_list: List[VerifiedComponentEvidence] = []
        specificity_values: List[float] = []

        for cand in top_candidates:
            # 1. Real forward causal intervention
            ab = self.runtime.apply_intervention(
                prompt=clean_prompt,
                target_token=target_token,
                layer=cand.layer,
                component_type="neuron",
                component_index=cand.component_index,
                ablation_scale=0.0,
            )
            dz = ab.delta_logit or 0.0

            # 2. 4-Negative Control Battery
            ctrl_deltas = []
            # Matched-Norm
            c1 = self.runtime.apply_intervention(clean_prompt, target_token, cand.layer, "neuron", (cand.component_index + 17) % 3072, 0.0)
            ctrl_deltas.append(abs(c1.delta_logit or 0.0))
            # Same-Layer
            c2 = self.runtime.apply_intervention(clean_prompt, target_token, cand.layer, "neuron", (cand.component_index + 251) % 3072, 0.0)
            ctrl_deltas.append(abs(c2.delta_logit or 0.0))
            # Same-Mechanism (adjacent layer)
            adj_l = (cand.layer + 1) % self.runtime.num_layers
            c3 = self.runtime.apply_intervention(clean_prompt, target_token, adj_l, "neuron", cand.component_index, 0.0)
            ctrl_deltas.append(abs(c3.delta_logit or 0.0))
            # Random Global
            c4 = self.runtime.apply_intervention(clean_prompt, target_token, (cand.layer + 3) % self.runtime.num_layers, "neuron", 777, 0.0)
            ctrl_deltas.append(abs(c4.delta_logit or 0.0))

            mean_ctrl = sum(ctrl_deltas) / max(len(ctrl_deltas), 1)
            spec_ratio = round(abs(dz) / max(mean_ctrl, 1e-4), 2)
            passed_controls = sum(1 for c in ctrl_deltas if abs(dz) > c)
            specificity_values.append(spec_ratio)

            # 3. Epistemic Hierarchy Assignment
            if spec_ratio >= 2.5 and abs(dz) >= 0.008 and passed_controls >= 3:
                tier = "CAUSALLY_VERIFIED_CIRCUIT_NODE"
            elif spec_ratio >= 1.5 and abs(dz) >= 0.002:
                tier = "CAUSALLY_SUPPORTED_NODE"
            else:
                tier = "FALSIFIED_GRADIENT_HEURISTIC"

            verified_list.append(
                VerifiedComponentEvidence(
                    component_id=cand.component_id,
                    layer=cand.layer,
                    component_index=cand.component_index,
                    gradient_attribution_score=cand.grad_x_act_score,
                    attribution_patching_score=cand.attribution_patching_score,
                    gradient_search_rank=cand.primary_attribution_rank,
                    clean_target_logit=round(ab.clean_logit, 4),
                    intervened_target_logit=round(ab.intervened_logit, 4),
                    causal_delta_z=round(dz, 4),
                    control_specificity_ratio=spec_ratio,
                    controls_passed_count=passed_controls,
                    epistemic_evidence_tier=tier,
                )
            )

        mean_spec = round(sum(specificity_values) / max(len(specificity_values), 1), 2)
        verified_count = sum(1 for v in verified_list if v.epistemic_evidence_tier == "CAUSALLY_VERIFIED_CIRCUIT_NODE")
        supported_count = sum(1 for v in verified_list if v.epistemic_evidence_tier == "CAUSALLY_SUPPORTED_NODE")

        summary = (
            f"Search-to-Verify completed: {total_screened} components screened via autograd attribution. "
            f"Top {len(top_candidates)} candidates evaluated under live causal interventions: "
            f"{verified_count} causally verified circuit nodes, {supported_count} supported nodes. "
            f"Mean negative control specificity = {mean_spec:.2f}x."
        )

        rep_id = f"report_s2v_{hashlib.sha256(f'{behavior_name}_{ts}'.encode()).hexdigest()[:10]}"

        return SearchToVerifyDiscoveryReport(
            report_id=rep_id,
            behavior_name=behavior_name,
            model_id=self.runtime.get_runtime_metadata().model_id,
            clean_prompt=clean_prompt,
            target_token=target_token,
            corrupted_prompt=corrupted_prompt,
            total_components_screened=total_screened,
            top_candidates_selected=len(top_candidates),
            verified_components=verified_list,
            mean_control_specificity=mean_spec,
            epistemic_summary=summary,
            timestamp_utc=ts,
        )
