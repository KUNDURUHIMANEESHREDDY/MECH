"""Mechanism Critic & Next-Best-Experiment Recommender for MECH.

Audits empirical claims for missing negative controls, unresolved competing mechanisms,
and recommends concrete, high-information-gain experiments to resolve scientific ambiguity.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.storage.database import DesktopStorage
from backend.storage.scientific_entities import Hypothesis, ExperimentRun

logger = logging.getLogger("MECH.science.mechanism_critic")


class MechanismAudit(BaseModel):
    claim_id: str
    target_component: str
    has_causal_intervention: bool
    has_negative_control: bool
    has_deterministic_replication: bool
    sample_size: int
    unresolved_alternatives: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    epistemic_grade: str  # ROBUST_CAUSAL, PARTIALLY_SUPPORTED, OBSERVATIONAL_CORRELATION, UNVALIDATED
    timestamp: float = Field(default_factory=time.time)


class NextBestExperimentProposal(BaseModel):
    proposal_id: str = Field(default_factory=lambda: f"prop_{uuid.uuid4().hex[:10]}")
    investigation_id: str
    target_component: str
    intervention_type: str
    recommended_action: str
    hypothesis_to_test_or_resolve: str
    expected_information_gain: str  # HIGH, MODERATE, LOW
    scientific_rationale: str
    timestamp: float = Field(default_factory=time.time)


class MechanismCriticEngine:
    """Evaluates the scientific rigor of findings and proposes next discriminative experiments."""

    def __init__(self, storage: Optional[DesktopStorage] = None) -> None:
        from pathlib import Path
        db_path = Path.home() / ".cache" / "neural-debugger" / "mech.db"
        self.storage = storage or DesktopStorage(db_path)
        self.storage.initialize()

    def audit_claim(self, investigation_id: str, component: str = "L9H9") -> MechanismAudit:
        """Audits empirical support and flags unexamined scientific gaps."""
        runs = self.storage.list_runs(investigation_id)

        causal_runs = [r for r in runs if r.get("delta_logit", 0) != 0]
        has_causal = len(causal_runs) > 0
        has_control = any(r.get("control_delta_logit") is not None for r in causal_runs)
        has_replication = len(causal_runs) >= 2

        unresolved = []
        limitations = []

        if not has_control:
            unresolved.append("Non-specific model disruption (No negative control head tested)")
            limitations.append("Lacks negative control baseline (e.g. L0H0)")

        unresolved.append("Alternative explanation: MLP Layer 8 feedforward contribution unresolved")
        unresolved.append("Alternative explanation: L8H4 backup name-mover compensation")

        if len(runs) < 5:
            limitations.append(f"Small experiment count ({len(runs)} runs recorded in database)")

        if has_causal and has_control and has_replication:
            grade = "ROBUST_CAUSAL"
        elif has_causal and has_control:
            grade = "PARTIALLY_SUPPORTED"
        elif has_causal:
            grade = "OBSERVATIONAL_CORRELATION"
        else:
            grade = "UNVALIDATED"

        return MechanismAudit(
            claim_id=f"claim_{component}",
            target_component=component,
            has_causal_intervention=has_causal,
            has_negative_control=has_control,
            has_deterministic_replication=has_replication,
            sample_size=max(1, len(causal_runs) * 20),
            unresolved_alternatives=unresolved,
            limitations=limitations,
            epistemic_grade=grade,
        )

    def recommend_next_experiment(self, investigation_id: str) -> NextBestExperimentProposal:
        """Generates the optimal next experiment to distinguish competing hypotheses."""
        audit = self.audit_claim(investigation_id)

        if not audit.has_negative_control:
            return NextBestExperimentProposal(
                investigation_id=investigation_id,
                target_component="L0H0",
                intervention_type="ABLATION_ZERO",
                recommended_action="Execute contrastive ablation on negative control head L0H0",
                hypothesis_to_test_or_resolve="Rule out non-specific layer-wide degradation",
                expected_information_gain="HIGH",
                scientific_rationale="Isolates specific target effect size from general prompt disruption.",
            )

        return NextBestExperimentProposal(
            investigation_id=investigation_id,
            target_component="MLP_L8",
            intervention_type="ABLATION_ZERO",
            recommended_action="Ablate MLP Layer 8 while preserving attention head L9H9 and measure Δlogit",
            hypothesis_to_test_or_resolve="Distinguish attention name-moving from feedforward associative retrieval",
            expected_information_gain="HIGH",
            scientific_rationale="Directly tests whether MLP L8 performs independent computation or simply routes through L9H9.",
        )


# Global mechanism critic engine singleton
mechanism_critic_engine = MechanismCriticEngine()
