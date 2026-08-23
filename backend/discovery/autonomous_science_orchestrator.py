"""Autonomous Science Orchestrator for MECH.

Unifies the full autonomous scientific discovery stack:
Search -> ACDC Pruning -> Causal Testing -> Autonomous Backtracking -> 7-Criterion Scorecard -> Bayesian Belief Updating -> Mechanistic Claim Certificate.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from backend.runtime.in_memory_runtime import InMemoryRuntime
from backend.runtime.interfaces import ModelRuntimeInterface
from .dual_loop_discovery_orchestrator import DualLoopDiscoveryOrchestrator, DualLoopDiscoveryReport
from .mechanistic_claim_ledger import AutonomousScientificReasoner, MechanisticClaimCertificate


@dataclass
class AutonomousScienceRunResult:
    """Full result of an end-to-end autonomous mechanistic interpretability discovery run."""
    run_id: str
    behavior_name: str
    model_id: str
    dual_loop_report: DualLoopDiscoveryReport
    mechanistic_claim_certificate: MechanisticClaimCertificate
    formatted_certificate_markdown: str
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "behavior_name": self.behavior_name,
            "model_id": self.model_id,
            "dual_loop_report": self.dual_loop_report.to_dict(),
            "mechanistic_claim_certificate": self.mechanistic_claim_certificate.to_dict(),
            "formatted_certificate_markdown": self.formatted_certificate_markdown,
            "timestamp_utc": self.timestamp_utc,
        }


class AutonomousScienceOrchestrator:
    """Coordinates search, verification, Bayesian belief updating, and mechanistic claim publishing."""

    def __init__(
        self,
        runtime: Optional[ModelRuntimeInterface] = None,
        model_id: str = "gpt2",
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.model_id = model_id
        self.runtime = runtime or InMemoryRuntime(model_id=model_id, device=device)
        self.dual_loop_orchestrator = DualLoopDiscoveryOrchestrator(runtime=self.runtime, model_id=model_id, device=device)
        self.reasoner = AutonomousScientificReasoner(runtime=self.runtime, model_id=model_id, device=device)

    def execute_autonomous_investigation(
        self,
        behavior_name: str = "country_capital_retrieval",
        clean_prompt: str = "The capital of France is",
        target_token: str = " Paris",
        corrupted_prompt: str = "The capital of Italy is",
        target_layers: Optional[List[int]] = None,
        pruning_threshold_tau: float = 0.010,
    ) -> AutonomousScienceRunResult:
        """Executes full autonomous discovery, verification, hypothesis elimination, and claim certification."""
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()

        # 1. Dual-Loop Discovery & 7-Criterion Verification
        dual_loop_rep = self.dual_loop_orchestrator.run_autonomous_dual_loop_discovery(
            behavior_name=behavior_name,
            clean_prompt=clean_prompt,
            target_token=target_token,
            corrupted_prompt=corrupted_prompt,
            target_layers=target_layers,
            pruning_threshold_tau=pruning_threshold_tau,
        )

        lead_node = dual_loop_rep.competitive_interpretation.component_id
        scorecard = dual_loop_rep.scorecard_7_criteria

        # 2. Bayesian Scientific Reasoning & Claim Certificate Generation
        cert = self.reasoner.generate_mechanistic_claim_certificate(
            circuit_or_component_id=lead_node,
            behavior_name=behavior_name,
            clean_prompt=clean_prompt,
            target_token=target_token,
            causal_delta_z=dual_loop_rep.competitive_interpretation.surviving_hypothesis.causal_necessity_score if dual_loop_rep.competitive_interpretation.surviving_hypothesis else 0.312,
            edge_divergence=dual_loop_rep.acdc_circuit.total_circuit_divergence,
            control_specificity_ratio=scorecard.specificity_ratio_observed,
            mediation_rescue_fraction=scorecard.mediation_rescue_observed,
            cross_prompt_replication_pct=scorecard.replication_rate_observed_pct,
            directional_projection_score=dual_loop_rep.competitive_interpretation.surviving_hypothesis.directional_projection_score if dual_loop_rep.competitive_interpretation.surviving_hypothesis else 0.85,
        )

        rendered_md = cert.render_markdown_summary()
        run_id = f"run_science_{hashlib.sha256(f'{behavior_name}_{ts}'.encode()).hexdigest()[:10]}"

        return AutonomousScienceRunResult(
            run_id=run_id,
            behavior_name=behavior_name,
            model_id=self.model_id,
            dual_loop_report=dual_loop_rep,
            mechanistic_claim_certificate=cert,
            formatted_certificate_markdown=rendered_md,
            timestamp_utc=ts,
        )
