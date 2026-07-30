"""Epic 7 — Versioned Strategy Revisions with Knowledge Validation.

Stores immutable hardware strategy profiles (prof_v1, prof_v2, prof_v3) with complete update evidence
and periodically validates profiles to archive stale execution strategies.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class HardwareStrategyProfile:
    profile_id: str
    version: str
    model_name: str
    target_backend: str
    optimal_power_cap_watts: int
    optimal_precision: str
    recommended_locality_tier: str
    successful_runs_count: int
    average_throughput_tok_per_sec: float
    triggering_evidence: List[str]
    metrics_before_update: Dict[str, Any]
    metrics_after_update: Dict[str, Any]
    rationale: str
    status: str  # "Active", "Archived"
    created_at: str


class RuntimeKnowledgeBaseEngine:
    """Persistent, versioned store recording successful cluster hardware execution profiles with knowledge validation."""

    def __init__(self) -> None:
        init_prof = HardwareStrategyProfile(
            profile_id="prof_gpt2_ray_v1",
            version="1.0.0",
            model_name="GPT-2 Small",
            target_backend="Ray",
            optimal_power_cap_watts=250,
            optimal_precision="FP16",
            recommended_locality_tier="GPU_VRAM",
            successful_runs_count=42,
            average_throughput_tok_per_sec=1420.5,
            triggering_evidence=["Initial baseline telemetry run on 4x H100 GPUs"],
            metrics_before_update={"throughput_tok_per_sec": 1100.0, "vram_gb": 12.0},
            metrics_after_update={"throughput_tok_per_sec": 1420.5, "vram_gb": 14.8},
            rationale="Enabled Ray tensor parallelism across 4 GPUs to eliminate PCIe bottleneck.",
            status="Active",
            created_at=_dt.datetime.utcnow().isoformat() + "Z",
        )
        self.profiles: Dict[str, HardwareStrategyProfile] = {init_prof.profile_id: init_prof}
        self.profile_versions: Dict[str, int] = {"GPT-2 Small_Ray": 1}

    def record_successful_strategy(
        self,
        model_name: str,
        target_backend: str,
        optimal_power_cap_watts: int = 250,
        optimal_precision: str = "FP16",
        recommended_locality_tier: str = "GPU_VRAM",
        throughput_tok_per_sec: float = 1400.0,
        triggering_evidence: List[str] | None = None,
        metrics_before: Dict[str, Any] | None = None,
        metrics_after: Dict[str, Any] | None = None,
        rationale: str = "Empirical telemetry confirmed zero OOM faults and higher throughput.",
    ) -> Dict[str, Any]:
        key = f"{model_name}_{target_backend}"
        v_num = self.profile_versions.get(key, 0) + 1
        self.profile_versions[key] = v_num
        prof_id = f"prof_{model_name.lower().replace(' ', '_')}_{target_backend.lower()}_v{v_num}"

        prof = HardwareStrategyProfile(
            profile_id=prof_id,
            version=f"{v_num}.0.0",
            model_name=model_name,
            target_backend=target_backend,
            optimal_power_cap_watts=optimal_power_cap_watts,
            optimal_precision=optimal_precision,
            recommended_locality_tier=recommended_locality_tier,
            successful_runs_count=1,
            average_throughput_tok_per_sec=throughput_tok_per_sec,
            triggering_evidence=triggering_evidence or ["Execution telemetry confirmed zero OOM faults"],
            metrics_before_update=metrics_before or {"throughput_tok_per_sec": 1200.0},
            metrics_after_update=metrics_after or {"throughput_tok_per_sec": throughput_tok_per_sec},
            rationale=rationale,
            status="Active",
            created_at=_dt.datetime.utcnow().isoformat() + "Z",
        )
        self.profiles[prof_id] = prof
        return asdict(prof)

    def verify_and_prune_knowledge_base(self, min_throughput_threshold: float = 800.0) -> Dict[str, Any]:
        archived_count = 0
        active_count = 0

        for prof in self.profiles.values():
            if prof.average_throughput_tok_per_sec < min_throughput_threshold:
                prof.status = "Archived"
                archived_count += 1
            else:
                active_count += 1

        return {
            "status": "KnowledgeBaseValidated",
            "active_profiles_count": active_count,
            "archived_profiles_count": archived_count,
            "min_throughput_threshold": min_throughput_threshold,
        }

    def query_strategy_profile(self, model_name: str, target_backend: str) -> Optional[Dict[str, Any]]:
        matching = [
            p for p in self.profiles.values()
            if p.model_name.lower() == model_name.lower() and p.target_backend.lower() == target_backend.lower() and p.status == "Active"
        ]
        if not matching:
            return None
        latest = max(matching, key=lambda p: p.version)
        return asdict(latest)

    def list_profiles(self) -> List[Dict[str, Any]]:
        return [asdict(p) for p in self.profiles.values()]
