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
    """Dataclass storing an immutable, versioned research execution strategy policy.

    The three effectiveness fields are ``Optional`` on purpose. ``None`` means
    *not measured*, which is a legitimate and common state: a strategy template
    has been applied but no run has scored it yet. Inventing a float there is
    how this repository used to report ``success_rate=0.91`` for a policy that
    had never been executed.
    """

    policy_id: str
    version: str
    domain: str
    workflow: List[str]
    success_rate: Optional[float]
    average_cost_usd: Optional[float]
    average_runtime_min: Optional[float]
    created_at: str
    created_by: str = "Research Strategy Optimizer"
    metrics_measured: bool = True
    metrics_source: Optional[str] = None

    @property
    def provenance(self) -> str:
        """Live only when every effectiveness figure came from a real run."""
        return "live" if self.metrics_measured else "unavailable"


def _round_or_none(value: Optional[float], places: int) -> Optional[float]:
    """Round a measurement, or pass ``None`` through untouched.

    ``round(None, 2)`` raises ``TypeError``. An unmeasured policy has to be
    storable, or the honest answer becomes an exception.
    """
    if value is None:
        return None
    return round(float(value), places)


class PolicyRepository:
    """Persistent, versioned store for research strategy policies."""

    def __init__(self) -> None:
        # The baseline policy is a *workflow template*, not a record of a run.
        # It previously shipped success_rate=0.91 / cost=4.12 / runtime=9.0,
        # which read as measured results and were then used as the comparison
        # baseline in `compare_policies`.
        init_policy = ResearchPolicy(
            policy_id="pol_cd_v1",
            version="1.0.0",
            domain="circuit_discovery",
            workflow=["Run SAE Inspection", "Causal Path Patching", "Cross-Model Alignment"],
            success_rate=None,
            average_cost_usd=None,
            average_runtime_min=None,
            created_at=_dt.datetime.utcnow().isoformat() + "Z",
            metrics_measured=False,
            metrics_source=None,
        )
        self.policies: Dict[str, ResearchPolicy] = {init_policy.policy_id: init_policy}
        self.domain_versions: Dict[str, int] = {"circuit_discovery": 1}

    def save_policy(
        self,
        domain: str,
        workflow: List[str],
        success_rate: Optional[float] = None,
        average_cost_usd: Optional[float] = None,
        average_runtime_min: Optional[float] = None,
        metrics_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        v_num = self.domain_versions.get(domain, 0) + 1
        self.domain_versions[domain] = v_num
        policy_id = f"pol_{domain}_v{v_num}"

        # Measured only when a caller supplied real figures AND said where they
        # came from. A bare number with no source is an unsourced claim.
        measured = metrics_source is not None and all(
            v is not None for v in (success_rate, average_cost_usd, average_runtime_min)
        )

        policy = ResearchPolicy(
            policy_id=policy_id,
            version=f"{v_num}.0.0",
            domain=domain,
            workflow=workflow,
            success_rate=_round_or_none(success_rate, 4),
            average_cost_usd=_round_or_none(average_cost_usd, 2),
            average_runtime_min=_round_or_none(average_runtime_min, 2),
            created_at=_dt.datetime.utcnow().isoformat() + "Z",
            metrics_measured=measured,
            metrics_source=metrics_source,
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

        # A delta between two unmeasured figures is not a small number, it is
        # an absence of one. Report None and say why, rather than inventing a
        # comparison that would read as "B is no better than A".
        def _delta(field: str, places: int) -> Optional[float]:
            a_val, b_val = getattr(pol_a, field), getattr(pol_b, field)
            if a_val is None or b_val is None:
                return None
            return round(float(b_val) - float(a_val), places)

        success_delta = _delta("success_rate", 4)
        unmeasured = sorted(
            field for field in ("success_rate", "average_cost_usd", "average_runtime_min")
            if getattr(pol_a, field) is None or getattr(pol_b, field) is None
        )

        return {
            "policy_a": asdict(pol_a),
            "policy_b": asdict(pol_b),
            "success_rate_delta": success_delta,
            "cost_delta_usd": _delta("average_cost_usd", 2),
            "runtime_delta_min": _delta("average_runtime_min", 2),
            # None when unmeasured -- previously this was
            # `pol_b.success_rate > pol_a.success_rate`, which raised TypeError
            # on the honest path and would have been arbitrary otherwise.
            "improvement_detected": (
                None if success_delta is None else success_delta > 0
            ),
            "metrics_measured": not unmeasured,
            "provenance": (
                "live" if not unmeasured else "unavailable"
            ),
            "unmeasured_fields": unmeasured,
            "reason": (
                None if not unmeasured else
                f"no comparison possible: {', '.join(unmeasured)} "
                "not measured for at least one policy"
            ),
        }

    def list_policies(self) -> List[Dict[str, Any]]:
        return [asdict(p) for p in self.policies.values()]
