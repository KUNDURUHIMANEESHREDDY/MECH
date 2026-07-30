"""Research Policy Repository.

Stores immutable versioned research strategy policies (policy_v1, policy_v2, policy_v3)
to enable comparative evaluation and prevent policy overwrite.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ResearchPolicy:
    """Dataclass storing an immutable, versioned research execution strategy policy."""

    policy_id: str
    version: str
    domain: str
    workflow: List[str]
    success_rate: float
    average_cost_usd: float
    average_runtime_min: float
    created_at: str
    created_by: str = "Research Strategy Optimizer"


class PolicyRepository:
    """Persistent, versioned store for research strategy policies."""

    def __init__(self) -> None:
        init_policy = ResearchPolicy(
            policy_id="pol_cd_v1",
            version="1.0.0",
            domain="circuit_discovery",
            workflow=["Run SAE Inspection", "Causal Path Patching", "Cross-Model Alignment"],
            success_rate=0.91,
            average_cost_usd=4.12,
            average_runtime_min=9.0,
            created_at=_dt.datetime.utcnow().isoformat() + "Z",
        )
        self.policies: Dict[str, ResearchPolicy] = {init_policy.policy_id: init_policy}
        self.domain_versions: Dict[str, int] = {"circuit_discovery": 1}

    def save_policy(
        self,
        domain: str,
        workflow: List[str],
        success_rate: float,
        average_cost_usd: float,
        average_runtime_min: float,
    ) -> Dict[str, Any]:
        v_num = self.domain_versions.get(domain, 0) + 1
        self.domain_versions[domain] = v_num
        policy_id = f"pol_{domain}_v{v_num}"

        policy = ResearchPolicy(
            policy_id=policy_id,
            version=f"{v_num}.0.0",
            domain=domain,
            workflow=workflow,
            success_rate=round(success_rate, 4),
            average_cost_usd=round(average_cost_usd, 2),
            average_runtime_min=round(average_runtime_min, 2),
            created_at=_dt.datetime.utcnow().isoformat() + "Z",
        )
        self.policies[policy_id] = policy
        return asdict(policy)

    def get_latest_policy(self, domain: str = "circuit_discovery") -> Optional[Dict[str, Any]]:
        domain_policies = [p for p in self.policies.values() if p.domain == domain]
        if not domain_policies:
            return None
        latest = max(domain_policies, key=lambda p: p.version)
        return asdict(latest)

    def compare_policies(self, policy_id_a: str, policy_id_b: str) -> Dict[str, Any]:
        pol_a = self.policies.get(policy_id_a)
        pol_b = self.policies.get(policy_id_b)

        if not pol_a or not pol_b:
            return {"status": "PolicyNotFound", "policy_a": policy_id_a, "policy_b": policy_id_b}

        return {
            "policy_a": asdict(pol_a),
            "policy_b": asdict(pol_b),
            "success_rate_delta": round(pol_b.success_rate - pol_a.success_rate, 4),
            "cost_delta_usd": round(pol_b.average_cost_usd - pol_a.average_cost_usd, 2),
            "runtime_delta_min": round(pol_b.average_runtime_min - pol_a.average_runtime_min, 2),
            "improvement_detected": pol_b.success_rate > pol_a.success_rate,
        }

    def list_policies(self) -> List[Dict[str, Any]]:
        return [asdict(p) for p in self.policies.values()]
