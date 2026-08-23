r"""Generation-Aware Theory Complexity Engine for MECH.

Quantifies the structural and epistemic complexity of candidate theories:
    C(T) = C_nodes + C_edges + C_params + C_primitives + C_exceptions

Penalizes accumulation of ad-hoc special cases and exemptions:
    A theory should not survive merely by accumulating exceptions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class TheoryComplexityBreakdown:
    theory_id: str
    c_nodes: float
    c_edges: float
    c_params: float
    c_primitives: float
    c_exceptions: float
    total_complexity: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "theory_id": self.theory_id,
            "c_nodes": round(self.c_nodes, 4),
            "c_edges": round(self.c_edges, 4),
            "c_params": round(self.c_params, 4),
            "c_primitives": round(self.c_primitives, 4),
            "c_exceptions": round(self.c_exceptions, 4),
            "total_complexity": round(self.total_complexity, 4),
        }


class TheoryComplexityEngine:
    """Calculates generation-aware complexity penalties across candidate theory populations."""

    def calculate_complexity(
        self,
        theory_id: str,
        num_primitives: int,
        num_assumptions: int,
        num_parameters: int,
        num_special_case_exceptions: int,
    ) -> TheoryComplexityBreakdown:
        """Computes structural Occam penalty."""
        c_nodes = num_primitives * 0.10
        c_edges = (num_primitives * (num_primitives - 1) / 2.0) * 0.02 if num_primitives > 1 else 0.0
        c_params = num_parameters * 0.05
        c_primitives = num_primitives * 0.15
        # Heavy penalty on ad-hoc exceptions (0.25 per exception)
        c_exceptions = num_special_case_exceptions * 0.25

        total = c_nodes + c_edges + c_params + c_primitives + c_exceptions

        return TheoryComplexityBreakdown(
            theory_id=theory_id,
            c_nodes=c_nodes,
            c_edges=c_edges,
            c_params=c_params,
            c_primitives=c_primitives,
            c_exceptions=c_exceptions,
            total_complexity=total,
        )
