"""Policy Engine Layer with Separated Hardware Constraints and Utility Weights.

Defines top-level policy utility weights (runtime, cost, reproducibility, energy)
and separates them from hard hardware constraints (budget, VRAM, deadline, carbon limit).
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class HardwareConstraints:
    """Hard physical hardware limits."""

    max_vram_gb: float = 16.0
    max_budget_usd_per_hour: float = 2.50
    deadline_sec: float = 3600.0
    carbon_limit_kg: float = 5.0
    enable_nvme_offload: bool = True


@dataclass
class RuntimeOptimizationPolicy:
    """Soft policy utility objectives and weights."""

    policy_id: str
    objective_name: str
    runtime_weight: float
    cost_weight: float
    reproducibility_weight: float
    energy_weight: float
    target_backend_priority: List[str]
    precision_constraint: str
    checkpoint_frequency_steps: int
    constraints: HardwareConstraints
    created_at: str


class PolicyEngine:
    """Manages runtime optimization policies and enforces hard hardware constraints."""

    SUPPORTED_OBJECTIVES = [
        "Fastest Completion",
        "Lowest Cost",
        "Lowest Energy",
        "Highest Reproducibility",
        "Balanced",
        "Interactive Debugging",
        "Large-Scale Benchmark",
    ]

    def __init__(self) -> None:
        init_constraints = HardwareConstraints(max_vram_gb=16.0, max_budget_usd_per_hour=2.50)
        init_pol = RuntimeOptimizationPolicy(
            policy_id="pol_balanced",
            objective_name="Balanced",
            runtime_weight=0.4,
            cost_weight=0.4,
            reproducibility_weight=0.1,
            energy_weight=0.1,
            target_backend_priority=["Ray", "Kubernetes", "Local"],
            precision_constraint="FP16",
            checkpoint_frequency_steps=100,
            constraints=init_constraints,
            created_at=_dt.datetime.utcnow().isoformat() + "Z",
        )
        self.active_policies: Dict[str, RuntimeOptimizationPolicy] = {init_pol.objective_name: init_pol}

    def configure_policy(
        self,
        objective_name: str = "Lowest Cost",
        custom_constraints: HardwareConstraints | None = None,
    ) -> Dict[str, Any]:
        if objective_name not in self.SUPPORTED_OBJECTIVES:
            objective_name = "Balanced"

        constraints = custom_constraints or HardwareConstraints()

        if objective_name == "Lowest Cost":
            r_wt, c_wt, rep_wt, e_wt = 0.1, 0.7, 0.1, 0.1
            backend_prio = ["Kubernetes", "Spot_Ray", "Local"]
            precision = "INT8"
            if custom_constraints is None:
                constraints.max_vram_gb = 12.0
                constraints.max_budget_usd_per_hour = 1.20
            ckpt_freq = 200
        elif objective_name == "Fastest Completion":
            r_wt, c_wt, rep_wt, e_wt = 0.8, 0.1, 0.05, 0.05
            backend_prio = ["Ray_H100", "Ray_A100", "Local"]
            precision = "FP16"
            if custom_constraints is None:
                constraints.max_vram_gb = 80.0
                constraints.max_budget_usd_per_hour = 10.0
            ckpt_freq = 500
        elif objective_name == "Highest Reproducibility":
            r_wt, c_wt, rep_wt, e_wt = 0.1, 0.1, 0.7, 0.1
            backend_prio = ["Deterministic_Local", "Ray_SingleNode"]
            precision = "FP32"
            if custom_constraints is None:
                constraints.max_vram_gb = 32.0
                constraints.max_budget_usd_per_hour = 5.0
            ckpt_freq = 50
        elif objective_name == "Interactive Debugging":
            r_wt, c_wt, rep_wt, e_wt = 0.6, 0.2, 0.1, 0.1
            backend_prio = ["Local_GPU", "Local_CPU"]
            precision = "FP16"
            if custom_constraints is None:
                constraints.max_vram_gb = 16.0
                constraints.max_budget_usd_per_hour = 2.0
            ckpt_freq = 1
        else:  # Balanced
            r_wt, c_wt, rep_wt, e_wt = 0.4, 0.4, 0.1, 0.1
            backend_prio = ["Ray", "Kubernetes", "Local"]
            precision = "FP16"
            if custom_constraints is None:
                constraints.max_vram_gb = 16.0
                constraints.max_budget_usd_per_hour = 2.50
            ckpt_freq = 100

        policy_id = f"pol_{objective_name.lower().replace(' ', '_')}"
        policy = RuntimeOptimizationPolicy(
            policy_id=policy_id,
            objective_name=objective_name,
            runtime_weight=r_wt,
            cost_weight=c_wt,
            reproducibility_weight=rep_wt,
            energy_weight=e_wt,
            target_backend_priority=backend_prio,
            precision_constraint=precision,
            checkpoint_frequency_steps=ckpt_freq,
            constraints=constraints,
            created_at=_dt.datetime.utcnow().isoformat() + "Z",
        )
        self.active_policies[objective_name] = policy
        return asdict(policy)

    def get_active_policy(self, objective_name: str = "Balanced") -> Dict[str, Any]:
        pol = self.active_policies.get(objective_name) or self.configure_policy(objective_name)
        return asdict(pol) if isinstance(pol, RuntimeOptimizationPolicy) else pol
