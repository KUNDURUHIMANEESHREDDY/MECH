"""Claim Calibration Engine for MECH.

Enforces strict epistemic mapping between evidence types and allowed claim strengths.
Prevents overclaiming (e.g. promoting attention observations to causal claims).
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("MECH.science.claim_calibration")


class EvidenceType(str, Enum):
    ATTENTION_OBSERVATION = "ATTENTION_OBSERVATION"
    ACTIVATION_CORRELATION = "ACTIVATION_CORRELATION"
    PROBE_DECODABILITY = "PROBE_DECODABILITY"
    CAUSAL_ABLATION_SINGLE = "CAUSAL_ABLATION_SINGLE"
    CAUSAL_PATCHING_REPLICATION = "CAUSAL_PATCHING_REPLICATION"
    CONTROLLED_CIRCUIT_VERIFICATION = "CONTROLLED_CIRCUIT_VERIFICATION"


class AllowedClaimStrength(str, Enum):
    OBSERVATIONAL_CLAIM = "OBSERVATIONAL_CLAIM"
    ASSOCIATIONAL_CLAIM = "ASSOCIATIONAL_CLAIM"
    CAUSAL_COMPONENT_CLAIM = "CAUSAL_COMPONENT_CLAIM"
    MECHANISTIC_CIRCUIT_CLAIM = "MECHANISTIC_CIRCUIT_CLAIM"
    REJECTED_OVERCLAIM = "REJECTED_OVERCLAIM"


class ClaimCalibrationVerdict(BaseModel):
    claimed_text: str
    underlying_evidence_type: EvidenceType
    allowed_strength: AllowedClaimStrength
    is_claim_valid: bool
    calibrated_statement: str
    epistemic_rationale: str
    limitations: List[str] = Field(default_factory=list)
    timestamp: float = Field(default_factory=time.time)


class ClaimCalibrationEngine:
    """Audits scientific claims and calibrates phrasing to empirical evidence levels."""

    def calibrate_claim(
        self,
        claim_text: str,
        evidence_type: EvidenceType,
        target_component: str,
        has_negative_control: bool = False,
        has_replication: bool = False,
    ) -> ClaimCalibrationVerdict:
        """Determines whether a proposed claim exceeds empirical evidence boundaries."""
        lower_claim = claim_text.lower()
        contains_strong_causal = any(
            w in lower_claim for w in ["causes", "proves", "discovered the mechanism", "controls the circuit", "guarantees"]
        )

        limitations = []

        if evidence_type in [EvidenceType.ATTENTION_OBSERVATION, EvidenceType.ACTIVATION_CORRELATION, EvidenceType.PROBE_DECODABILITY]:
            allowed = AllowedClaimStrength.OBSERVATIONAL_CLAIM
            limitations.append("Attention routing and activation correlations do not establish causal mediation.")
            if contains_strong_causal:
                is_valid = False
                calibrated = f"Identified component {target_component} exhibiting activation patterns consistent with the phenomenon."
                rationale = "Claim rejected: Observational attention patterns cannot support causal mechanism assertions."
            else:
                is_valid = True
                calibrated = claim_text
                rationale = "Claim calibrated: Observational correlation properly articulated without causal overclaiming."

        elif evidence_type == EvidenceType.CAUSAL_ABLATION_SINGLE:
            if not has_negative_control:
                allowed = AllowedClaimStrength.ASSOCIATIONAL_CLAIM
                is_valid = not contains_strong_causal
                limitations.append("Lacks negative control to isolate component-specific effect from general disruption.")
                calibrated = f"Ablation of {target_component} resulted in logit change, but control baseline is required."
                rationale = "Claim constrained: Uncontrolled single ablation cannot assert specific mechanism."
            else:
                allowed = AllowedClaimStrength.CAUSAL_COMPONENT_CLAIM
                is_valid = not any(w in lower_claim for w in ["complete mechanism", "discovered the mechanism", "entire circuit"])
                calibrated = f"Independently verified causal mediation for component {target_component} over negative control baseline."
                rationale = "Claim supported: Causal intervention effect demonstrated over negative control."

        elif evidence_type in [EvidenceType.CAUSAL_PATCHING_REPLICATION, EvidenceType.CONTROLLED_CIRCUIT_VERIFICATION]:
            if has_negative_control and has_replication:
                allowed = AllowedClaimStrength.MECHANISTIC_CIRCUIT_CLAIM
                is_valid = True
                calibrated = claim_text
                rationale = "Claim fully supported: Multi-stage causal intervention, negative controls, and replications verified."
            else:
                allowed = AllowedClaimStrength.CAUSAL_COMPONENT_CLAIM
                is_valid = not ("complete mechanism" in lower_claim)
                calibrated = f"Validated causal contribution of {target_component} across test prompts."
                rationale = "Claim adjusted: Replication or control incomplete for full circuit assertion."

        else:
            allowed = AllowedClaimStrength.OBSERVATIONAL_CLAIM
            is_valid = False
            calibrated = f"Candidate component {target_component} identified."
            rationale = "Unknown evidence type: default to conservative observational phrasing."

        return ClaimCalibrationVerdict(
            claimed_text=claim_text,
            underlying_evidence_type=evidence_type,
            allowed_strength=allowed,
            is_claim_valid=is_valid,
            calibrated_statement=calibrated,
            epistemic_rationale=rationale,
            limitations=limitations,
        )


# Global claim calibration engine singleton
claim_calibration_engine = ClaimCalibrationEngine()
