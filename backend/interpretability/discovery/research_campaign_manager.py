"""Research Campaign Manager - Scientist's Workspace Engine.

Manages end-to-end scientific research campaigns, tracking:
  - Goal & Research Objectives
  - Planned vs. Completed vs. Failed Experiments
  - Accumulated Evidence & Remaining Uncertainty
  - Compute Consumed (FLOPs / GPU seconds) & Remaining Budget ($)
  - Papers Reproduced
  - Emitted Mechanism Claims
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .discovery_planner import ResearchGoal


@dataclass
class ExperimentRecord:
    """Record of an executed experiment within a campaign."""
    experiment_id: str
    algorithm_name: str
    target_model: str
    status: str  # Completed, Failed
    compute_consumed_flops: float
    runtime_ms: float
    evidence: Dict[str, Any]
    uncertainty_before: float
    uncertainty_after: float
    timestamp: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")


@dataclass
class ResearchCampaign:
    """Complete scientist workspace object representing an ongoing or completed research campaign."""
    campaign_id: str
    title: str
    goal_description: str
    status: str = "Running"  # Running, Completed, Failed, Paused
    target_model: str = "GPT2-S"
    planned_experiments: List[str] = field(default_factory=list)
    completed_experiments: List[ExperimentRecord] = field(default_factory=list)
    failed_experiments: List[ExperimentRecord] = field(default_factory=list)
    evidence_accumulated: List[Dict[str, Any]] = field(default_factory=list)
    remaining_uncertainty: float = 0.50
    budget_remaining_usd: float = 10.00
    compute_consumed_flops: float = 0.0
    papers_reproduced: List[str] = field(default_factory=list)
    final_mechanism_claims: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")
    updated_at: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "title": self.title,
            "goal_description": self.goal_description,
            "status": self.status,
            "target_model": self.target_model,
            "planned_experiments": self.planned_experiments,
            "completed_experiments": [e.__dict__ for e in self.completed_experiments],
            "failed_experiments": [e.__dict__ for e in self.failed_experiments],
            "evidence_accumulated": self.evidence_accumulated,
            "remaining_uncertainty": self.remaining_uncertainty,
            "budget_remaining_usd": round(self.budget_remaining_usd, 2),
            "compute_consumed_flops": round(self.compute_consumed_flops, 2),
            "papers_reproduced": self.papers_reproduced,
            "final_mechanism_claims": self.final_mechanism_claims,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ResearchCampaign:
        comp_exps = [ExperimentRecord(**e) for e in data.get("completed_experiments", [])]
        fail_exps = [ExperimentRecord(**e) for e in data.get("failed_experiments", [])]
        return cls(
            campaign_id=data["campaign_id"],
            title=data["title"],
            goal_description=data.get("goal_description", ""),
            status=data.get("status", "Running"),
            target_model=data.get("target_model", "GPT2-S"),
            planned_experiments=data.get("planned_experiments", []),
            completed_experiments=comp_exps,
            failed_experiments=fail_exps,
            evidence_accumulated=data.get("evidence_accumulated", []),
            remaining_uncertainty=data.get("remaining_uncertainty", 0.50),
            budget_remaining_usd=data.get("budget_remaining_usd", 10.00),
            compute_consumed_flops=data.get("compute_consumed_flops", 0.0),
            papers_reproduced=data.get("papers_reproduced", []),
            final_mechanism_claims=data.get("final_mechanism_claims", []),
            created_at=data.get("created_at", _dt.datetime.utcnow().isoformat() + "Z"),
            updated_at=data.get("updated_at", _dt.datetime.utcnow().isoformat() + "Z"),
        )


class ResearchCampaignManager:
    """Persistent Manager & Scientist Workspace Service managing research campaigns."""

    def __init__(self, storage_dir: str = "backend/research_datasets/campaigns") -> None:
        self.storage_dir = storage_dir
        self.storage_file = os.path.join(storage_dir, "campaigns_index.json")
        self._campaigns: Dict[str, ResearchCampaign] = {}
        self.load()

    def load(self) -> None:
        """Loads campaigns from disk storage."""
        if not os.path.exists(self.storage_file):
            self._seed_defaults()
            return

        try:
            with open(self.storage_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data.get("campaigns", []):
                    c = ResearchCampaign.from_dict(item)
                    self._campaigns[c.campaign_id] = c
        except Exception:
            self._seed_defaults()

    def save(self) -> None:
        """Persists all campaigns to disk."""
        os.makedirs(self.storage_dir, exist_ok=True)
        data = {
            "version": "1.0",
            "updated_at": _dt.datetime.utcnow().isoformat() + "Z",
            "campaigns": [c.to_dict() for c in self._campaigns.values()]
        }
        with open(self.storage_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _seed_defaults(self) -> None:
        """Seeds benchmark research campaigns into workspace."""
        c1 = ResearchCampaign(
            campaign_id="campaign_ioi_01",
            title="IOI Circuit Discovery & Cross-Model Validation",
            goal_description="Discover, prune, and validate indirect object identification circuit across GPT-2, Gemma, and Llama.",
            status="Completed",
            target_model="GPT2-S",
            planned_experiments=["attribution_patching", "acdc", "path_patching", "causal_scrubbing", "feature_universality"],
            completed_experiments=[
                ExperimentRecord(
                    experiment_id="exp_01_attr", algorithm_name="attribution_patching", target_model="GPT2-S",
                    status="Completed", compute_consumed_flops=1.2e12, runtime_ms=1200.0,
                    evidence={"isolated_heads": ["L9H9", "L10H0"]}, uncertainty_before=0.50, uncertainty_after=0.35
                ),
                ExperimentRecord(
                    experiment_id="exp_02_acdc", algorithm_name="acdc", target_model="GPT2-S",
                    status="Completed", compute_consumed_flops=8.5e12, runtime_ms=7800.0,
                    evidence={"retained_nodes": 4, "pruned_ratio": 0.94}, uncertainty_before=0.35, uncertainty_after=0.18
                ),
                ExperimentRecord(
                    experiment_id="exp_03_scrub", algorithm_name="causal_scrubbing", target_model="GPT2-S",
                    status="Completed", compute_consumed_flops=1.1e13, runtime_ms=9500.0,
                    evidence={"falsification_verdict": "PASS", "behavior_preserved": 0.96}, uncertainty_before=0.18, uncertainty_after=0.038
                )
            ],
            failed_experiments=[],
            evidence_accumulated=[
                {"evidence_id": "ev_01", "type": "ACDC Pruning", "value": "94% pruned; 3-head minimal subgraph"},
                {"evidence_id": "ev_02", "type": "Causal Scrubbing", "value": "Preserved 96% accuracy under name-type equivalence class"}
            ],
            remaining_uncertainty=0.038,
            budget_remaining_usd=8.45,
            compute_consumed_flops=2.07e13,
            papers_reproduced=["paper_wang_2022", "paper_conmy_2023"],
            final_mechanism_claims=["claim_ioi_name_mover"]
        )
        self._campaigns[c1.campaign_id] = c1
        self.save()

    def create_campaign(self, title: str, goal: ResearchGoal, initial_budget_usd: float = 10.0) -> ResearchCampaign:
        """Creates and registers a new research campaign."""
        cid = f"campaign_{hash(title + str(time.time())) & 0xffffffff:08x}"
        campaign = ResearchCampaign(
            campaign_id=cid,
            title=title,
            goal_description=goal.description,
            status="Running",
            target_model=goal.model_id,
            planned_experiments=["attribution_patching", "acdc", "causal_scrubbing"],
            remaining_uncertainty=0.50,
            budget_remaining_usd=initial_budget_usd
        )
        self._campaigns[cid] = campaign
        self.save()
        return campaign

    def record_experiment_run(
        self,
        campaign_id: str,
        algorithm_name: str,
        target_model: str,
        success: bool,
        compute_flops: float,
        runtime_ms: float,
        evidence: Dict[str, Any],
        uncertainty_after: float
    ) -> ResearchCampaign:
        """Logs an experiment execution run into a campaign."""
        if campaign_id not in self._campaigns:
            raise ValueError(f"Campaign '{campaign_id}' not found.")

        c = self._campaigns[campaign_id]
        exp_id = f"exp_{len(c.completed_experiments) + len(c.failed_experiments) + 1:02d}_{algorithm_name}"
        unc_before = c.remaining_uncertainty

        record = ExperimentRecord(
            experiment_id=exp_id,
            algorithm_name=algorithm_name,
            target_model=target_model,
            status="Completed" if success else "Failed",
            compute_consumed_flops=compute_flops,
            runtime_ms=runtime_ms,
            evidence=evidence,
            uncertainty_before=unc_before,
            uncertainty_after=uncertainty_after
        )

        c.compute_consumed_flops += compute_flops
        # Estimated cost: $0.00001 per 1e12 FLOPs
        cost_usd = (compute_flops / 1e12) * 0.05
        c.budget_remaining_usd = max(0.0, c.budget_remaining_usd - cost_usd)

        if success:
            c.completed_experiments.append(record)
            c.remaining_uncertainty = uncertainty_after
            c.evidence_accumulated.append({"exp_id": exp_id, "algorithm": algorithm_name, "evidence": evidence})
        else:
            c.failed_experiments.append(record)

        if c.remaining_uncertainty <= 0.05:
            c.status = "Completed"

        c.updated_at = _dt.datetime.utcnow().isoformat() + "Z"
        self.save()
        return c

    def get(self, campaign_id: str) -> Optional[ResearchCampaign]:
        """Retrieves a campaign by ID."""
        return self._campaigns.get(campaign_id)

    def list_all(self) -> List[ResearchCampaign]:
        """Lists all registered research campaigns."""
        return list(self._campaigns.values())
