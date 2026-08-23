r"""Theory Branching Engine for Multi-Generation Population Evolution in MECH.

Spawns structured, non-cosmetic theory branches from empirical failure points:
    T_parent -> {Branch_A (Routing), Branch_B (Scale), Branch_C (Polysemantic), Branch_D (SSM Primitive)}
"""

from __future__ import annotations

import copy
import datetime as _dt
import hashlib
from typing import Any, Dict, List, Optional

from .theory_version_registry import TheoryVersion, TransportabilityLawEquation


class TheoryBranchingEngine:
    """Spawns diverse evolutionary branches to explain residual discrepancies."""

    def branch_theory(
        self,
        parent_theory: TheoryVersion,
        generation_index: int,
    ) -> List[TheoryVersion]:
        """Generates 4 distinct offspring theory versions."""
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        eq = parent_theory.equation

        # Branch A: MoE Routing Specialized
        eq_a = copy.deepcopy(eq)
        eq_a.w_routing = -1.45
        t_a = TheoryVersion(
            version_id=f"T_G{generation_index}_BRANCH_A_MOE_ROUTED",
            parent_version_id=parent_theory.version_id,
            equation=eq_a,
            assumptions=parent_theory.assumptions + ["MoE Dynamic Routing Penalty"],
            validity_domain=["Sparse MoE Transformers (e.g. Mixtral)"],
            known_boundaries=["SSM Recurrent Architectures"],
            evidence_experiment_ids=parent_theory.evidence_experiment_ids + [f"EXP_BRANCH_A_G{generation_index}"],
            timestamp_utc=ts,
        )

        # Branch B: Scale Asymmetry Specialized
        eq_b = copy.deepcopy(eq)
        eq_b.w_dim = eq.w_dim - 0.65
        t_b = TheoryVersion(
            version_id=f"T_G{generation_index}_BRANCH_B_SCALE_ASYMMETRY",
            parent_version_id=parent_theory.version_id,
            equation=eq_b,
            assumptions=parent_theory.assumptions + ["Asymmetric Capacity Manifold Alignment"],
            validity_domain=["Cross-Scale Dense Transformers (e.g. 2B -> 70B)"],
            known_boundaries=["SSM Recurrent Architectures"],
            evidence_experiment_ids=parent_theory.evidence_experiment_ids + [f"EXP_BRANCH_B_G{generation_index}"],
            timestamp_utc=ts,
        )

        # Branch C: High Polysemantic Damping
        eq_c = copy.deepcopy(eq)
        eq_c.w_poly = eq.w_poly - 0.75
        t_c = TheoryVersion(
            version_id=f"T_G{generation_index}_BRANCH_C_POLY_DAMPING",
            parent_version_id=parent_theory.version_id,
            equation=eq_c,
            assumptions=parent_theory.assumptions + ["Polysemantic Interference SAE Latent Damping"],
            validity_domain=["Deep Superposition Dense Models"],
            known_boundaries=["SSM Recurrent Architectures"],
            evidence_experiment_ids=parent_theory.evidence_experiment_ids + [f"EXP_BRANCH_C_G{generation_index}"],
            timestamp_utc=ts,
        )

        # Branch D: Substrate State Space Primitive
        eq_d = copy.deepcopy(eq)
        eq_d.bias = eq.bias - 1.20
        t_d = TheoryVersion(
            version_id=f"T_G{generation_index}_BRANCH_D_SSM_PRIMITIVE",
            parent_version_id=parent_theory.version_id,
            equation=eq_d,
            assumptions=parent_theory.assumptions + ["Recurrent State Space Selective Scan Alignment"],
            validity_domain=["SSM Recurrent Architectures (e.g. Mamba-2)"],
            known_boundaries=[],
            evidence_experiment_ids=parent_theory.evidence_experiment_ids + [f"EXP_BRANCH_D_G{generation_index}"],
            timestamp_utc=ts,
        )

        return [t_a, t_b, t_c, t_d]
