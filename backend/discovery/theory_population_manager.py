r"""Theory Population Manager & Multi-Generation Registry for MECH.

Maintains multi-generation populations of competing mechanistic theories:
    G_0 (Initial Candidates) -> G_1 (Post-Perturbation Branches) -> G_2 (Speciated Substrate Families)
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .theory_version_registry import TheoryVersion, TransportabilityLawEquation


@dataclass
class GenerationSnapshot:
    generation_index: int
    theories: List[TheoryVersion]
    posteriors: Dict[str, float]
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation_index": self.generation_index,
            "theories": [t.to_dict() for t in self.theories],
            "posteriors": {k: round(v, 4) for k, v in self.posteriors.items()},
            "timestamp_utc": self.timestamp_utc,
        }


class TheoryPopulationManager:
    """Tracks population state across evolutionary generations."""

    def __init__(self) -> None:
        self.generations: List[GenerationSnapshot] = []
        self._initialize_generation_zero()

    def _initialize_generation_zero(self) -> None:
        """Constructs G_0: 3 initial competing theory families."""
        ts = "2026-08-16T12:00:00Z"

        # Theory 1: Simple Linear Transportability
        t1 = TheoryVersion(
            version_id="T_G0_SIMPLE_LINEAR",
            parent_version_id=None,
            equation=TransportabilityLawEquation(
                w_role=1.20,
                w_linear=1.00,
                w_poly=0.00,
                w_dim=-0.50,
                w_routing=0.00,
                bias=-0.40,
            ),
            assumptions=["Linear Subspace Invariance"],
            validity_domain=["Standard Dense Transformers"],
            known_boundaries=["Polysemantic Interference", "MoE Routing"],
            evidence_experiment_ids=["EXP_CALIB_1"],
            timestamp_utc=ts,
        )

        # Theory 2: Substrate-Conditioned Multi-Term Law
        t2 = TheoryVersion(
            version_id="T_G0_SUBSTRATE_CONDITIONED",
            parent_version_id=None,
            equation=TransportabilityLawEquation(
                w_role=1.40,
                w_linear=1.20,
                w_poly=-1.80,
                w_dim=-1.20,
                w_routing=-1.45,
                bias=-0.60,
            ),
            assumptions=["Dense Homogeneity", "MoE Routing Dispersion Penalty", "Polysemantic Damping"],
            validity_domain=["Standard Dense Transformers", "Sparse MoE Transformers"],
            known_boundaries=["SSM Recurrent Architectures"],
            evidence_experiment_ids=["EXP_CALIB_1", "EXP_CALIB_2"],
            timestamp_utc=ts,
        )

        # Theory 3: Universal Primitive Non-Linear Alignment
        t3 = TheoryVersion(
            version_id="T_G0_UNIVERSAL_NONLINEAR_PRIMITIVE",
            parent_version_id=None,
            equation=TransportabilityLawEquation(
                w_role=1.60,
                w_linear=0.80,
                w_poly=-2.20,
                w_dim=-1.50,
                w_routing=-0.80,
                bias=-0.80,
            ),
            assumptions=["Universal Non-Linear Manifold Isomorphism"],
            validity_domain=["All Transformer Architectures"],
            known_boundaries=["SSM Recurrent Architectures"],
            evidence_experiment_ids=["EXP_CALIB_1", "EXP_CALIB_3"],
            timestamp_utc=ts,
        )

        g0 = GenerationSnapshot(
            generation_index=0,
            theories=[t1, t2, t3],
            posteriors={t1.version_id: 0.333, t2.version_id: 0.333, t3.version_id: 0.334},
            timestamp_utc=ts,
        )
        self.generations.append(g0)

    def advance_generation(
        self,
        new_theories: List[TheoryVersion],
        posteriors: Dict[str, float],
    ) -> GenerationSnapshot:
        """Adds a newly evolved generation to the lineage."""
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        next_gen_idx = len(self.generations)
        snapshot = GenerationSnapshot(
            generation_index=next_gen_idx,
            theories=new_theories,
            posteriors=posteriors,
            timestamp_utc=ts,
        )
        self.generations.append(snapshot)
        return snapshot

    def get_latest_generation(self) -> GenerationSnapshot:
        return self.generations[-1]
