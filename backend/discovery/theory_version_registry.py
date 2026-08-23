r"""Theory Version Registry & Lineage Engine for MECH.

Maintains first-class versioned theory objects (T_0, T_1, T_2, ...):
1. Immutable SHA-256 theory hashes
2. Directed acyclic parent lineage
3. Explicit validity domains and boundary definitions
4. Evidence and causal experiment provenance tracking
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class TransportabilityLawEquation:
    w_role: float
    w_linear: float
    w_poly: float
    w_dim: float
    w_routing: float
    bias: float

    def predict_transfer(
        self,
        s_role: float,
        a_linear: float,
        h_poly: float,
        delta_dim: float,
        r_routing: float = 0.0,
    ) -> float:
        z = (
            self.w_role * s_role
            + self.w_linear * a_linear
            + self.w_poly * h_poly
            + self.w_dim * delta_dim
            + self.w_routing * r_routing
            + self.bias
        )
        # Sigmoid activation
        return 1.0 / (1.0 + math.exp(-z))


import math


@dataclass
class TheoryVersion:
    version_id: str
    parent_version_id: Optional[str]
    equation: TransportabilityLawEquation
    assumptions: List[str]
    validity_domain: List[str]
    known_boundaries: List[str]
    evidence_experiment_ids: List[str]
    timestamp_utc: str
    theory_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        self.theory_sha256 = self.compute_hash()

    def compute_hash(self) -> str:
        payload = {
            "version_id": self.version_id,
            "parent_id": self.parent_version_id,
            "eq": asdict(self.equation),
            "assumptions": sorted(self.assumptions),
            "validity_domain": sorted(self.validity_domain),
            "known_boundaries": sorted(self.known_boundaries),
            "evidence": sorted(self.evidence_experiment_ids),
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "parent_version_id": self.parent_version_id,
            "equation": asdict(self.equation),
            "assumptions": self.assumptions,
            "validity_domain": self.validity_domain,
            "known_boundaries": self.known_boundaries,
            "evidence_experiment_ids": self.evidence_experiment_ids,
            "timestamp_utc": self.timestamp_utc,
            "theory_sha256": self.theory_sha256,
        }


class TheoryVersionRegistry:
    """Manages the immutable evolutionary lineage of mechanistic theories."""

    def __init__(self) -> None:
        self.theories: Dict[str, TheoryVersion] = {}
        self._initialize_base_theory()

    def _initialize_base_theory(self) -> None:
        t0 = TheoryVersion(
            version_id="T_0_BASE",
            parent_version_id=None,
            equation=TransportabilityLawEquation(
                w_role=1.40,
                w_linear=1.20,
                w_poly=-1.80,
                w_dim=-1.20,
                w_routing=0.00,  # Unaware of MoE dynamic routing
                bias=-0.60,
            ),
            assumptions=[
                "Dense Transformer Self-Attention Homogeneity",
                "Linear Inter-Substrate Projection",
                "Negligible Routing Perturbation",
            ],
            validity_domain=["Standard Dense Transformers"],
            known_boundaries=["SSM Recurrent Architectures", "High-Superposition Polysemanticity"],
            evidence_experiment_ids=["EXP_CALIBRATION_PHASE48", "EXP_BLIND_PHASE49"],
            timestamp_utc="2026-08-16T10:00:00Z",
        )
        self.theories[t0.version_id] = t0

    def register_version(self, theory: TheoryVersion) -> None:
        if theory.version_id in self.theories:
            raise ValueError(f"Theory {theory.version_id} already exists! Theories are immutable.")
        self.theories[theory.version_id] = theory

    def get_version(self, version_id: str) -> TheoryVersion:
        if version_id not in self.theories:
            raise KeyError(f"Theory version {version_id} not found in registry.")
        return self.theories[version_id]

    def get_lineage(self, version_id: str) -> List[str]:
        lineage = []
        curr: Optional[str] = version_id
        while curr:
            lineage.append(curr)
            curr = self.theories[curr].parent_version_id
        return list(reversed(lineage))
