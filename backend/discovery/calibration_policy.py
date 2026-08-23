"""Calibration Policy Configuration & Provenance for MECH."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class CalibrationPolicyConfig:
    """Configurable and auditable parameters governing model criticism and epistemic abstention."""
    policy_version: str = "2.0.0-phase25"
    valid_confident_threshold: float = 0.12
    valid_uncertain_threshold: float = 0.20
    min_samples_required: int = 5
    uncertain_utility_penalty_multiplier: float = 2.50
    uncertain_likelihood_attenuation_power: float = 0.50

    def compute_policy_hash(self) -> str:
        s = f"{self.policy_version}_{self.valid_confident_threshold}_{self.valid_uncertain_threshold}_{self.min_samples_required}_{self.uncertain_utility_penalty_multiplier}_{self.uncertain_likelihood_attenuation_power}"
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "valid_confident_threshold": self.valid_confident_threshold,
            "valid_uncertain_threshold": self.valid_uncertain_threshold,
            "min_samples_required": self.min_samples_required,
            "uncertain_utility_penalty_multiplier": self.uncertain_utility_penalty_multiplier,
            "uncertain_likelihood_attenuation_power": self.uncertain_likelihood_attenuation_power,
            "policy_hash": self.compute_policy_hash(),
        }
