r"""Theory Structural Diff Engine for MECH.

Computes formal, machine-verifiable diffs between theory generations (T_t -> T_{t+1}):
    ΔT = TheoryDiff(added_assumptions, removed_assumptions, parameter_shifts, domain_deltas, causal_reason)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

from .theory_version_registry import TheoryVersion


@dataclass
class TheoryDiffRecord:
    source_theory_id: str
    target_theory_id: str
    added_assumptions: List[str]
    removed_assumptions: List[str]
    parameter_shifts: Dict[str, float]
    expanded_domains: List[str]
    contracted_domains: List[str]
    complexity_delta: float
    causal_reason_for_revision: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_theory_id": self.source_theory_id,
            "target_theory_id": self.target_theory_id,
            "added_assumptions": self.added_assumptions,
            "removed_assumptions": self.removed_assumptions,
            "parameter_shifts": {k: round(v, 4) for k, v in self.parameter_shifts.items()},
            "expanded_domains": self.expanded_domains,
            "contracted_domains": self.contracted_domains,
            "complexity_delta": round(self.complexity_delta, 4),
            "causal_reason_for_revision": self.causal_reason_for_revision,
        }


class TheoryDiffEngine:
    """Computes and validates structural diffs between theory versions."""

    def compute_diff(
        self,
        t_source: TheoryVersion,
        t_target: TheoryVersion,
        causal_reason: str,
    ) -> TheoryDiffRecord:
        """Calculates exact delta between source and revised theory."""
        src_assump = set(t_source.assumptions)
        tgt_assump = set(t_target.assumptions)

        added_assumptions = sorted(list(tgt_assump - src_assump))
        removed_assumptions = sorted(list(src_assump - tgt_assump))

        src_dom = set(t_source.validity_domain)
        tgt_dom = set(t_target.validity_domain)

        expanded_domains = sorted(list(tgt_dom - src_dom))
        contracted_domains = sorted(list(src_dom - tgt_dom))

        param_shifts = {
            "delta_w_role": t_target.equation.w_role - t_source.equation.w_role,
            "delta_w_linear": t_target.equation.w_linear - t_source.equation.w_linear,
            "delta_w_poly": t_target.equation.w_poly - t_source.equation.w_poly,
            "delta_w_dim": t_target.equation.w_dim - t_source.equation.w_dim,
            "delta_w_routing": t_target.equation.w_routing - t_source.equation.w_routing,
            "delta_bias": t_target.equation.bias - t_source.equation.bias,
        }

        # Complexity penalty: non-zero weights added + assumptions
        complexity_delta = len(added_assumptions) * 0.10 + abs(param_shifts["delta_w_routing"]) * 0.20

        return TheoryDiffRecord(
            source_theory_id=t_source.version_id,
            target_theory_id=t_target.version_id,
            added_assumptions=added_assumptions,
            removed_assumptions=removed_assumptions,
            parameter_shifts=param_shifts,
            expanded_domains=expanded_domains,
            contracted_domains=contracted_domains,
            complexity_delta=complexity_delta,
            causal_reason_for_revision=causal_reason,
        )
