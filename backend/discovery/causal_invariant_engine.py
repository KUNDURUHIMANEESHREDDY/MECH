r"""Causal Invariant Engine & Mechanism Ontology for MECH.

Represents abstract causal mechanisms and evaluates invariant hypotheses:
    I_1 = Causal Ordering
    I_2 = Mediation Topology
    I_3 = Input-Output Transformation
    I_4 = Intervention Response
    I_5 = Composition Law

Classifies invariants:
    UNIVERSAL_INVARIANT | SUBSTRATE_CONDITIONED_INVARIANT | MODALITY_CONDITIONED_INVARIANT | MODEL_SPECIFIC
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class InvariantScope(str, Enum):
    UNIVERSAL_INVARIANT = "UNIVERSAL_INVARIANT"
    SUBSTRATE_CONDITIONED_INVARIANT = "SUBSTRATE_CONDITIONED_INVARIANT"
    MODALITY_CONDITIONED_INVARIANT = "MODALITY_CONDITIONED_INVARIANT"
    MODEL_SPECIFIC = "MODEL_SPECIFIC"
    UNRESOLVED = "UNRESOLVED"


@dataclass
class CausalMechanism:
    mechanism_id: str
    mechanism_name: str
    computational_program: str
    input_signature: str
    output_signature: str
    target_substrates: List[str]
    target_modalities: List[str]
    invariant_scope: InvariantScope
    causal_signature_fidelity: float
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mechanism_id": self.mechanism_id,
            "mechanism_name": self.mechanism_name,
            "computational_program": self.computational_program,
            "input_signature": self.input_signature,
            "output_signature": self.output_signature,
            "target_substrates": self.target_substrates,
            "target_modalities": self.target_modalities,
            "invariant_scope": self.invariant_scope.value,
            "causal_signature_fidelity": round(self.causal_signature_fidelity, 4),
            "timestamp_utc": self.timestamp_utc,
        }


class CausalInvariantEngine:
    """Evaluates causal mechanisms across substrates and modalities to classify their scope."""

    def classify_invariant_scope(
        self,
        cross_arch_fidelity: float,
        cross_modal_fidelity: float,
        is_substrate_divergent: bool,
        is_modality_divergent: bool,
    ) -> InvariantScope:
        """Classifies invariant scope while preventing false universalization."""
        if cross_arch_fidelity >= 0.90 and cross_modal_fidelity >= 0.85 and not is_substrate_divergent and not is_modality_divergent:
            return InvariantScope.UNIVERSAL_INVARIANT
        elif cross_arch_fidelity >= 0.90 and is_modality_divergent:
            return InvariantScope.MODALITY_CONDITIONED_INVARIANT
        elif cross_modal_fidelity >= 0.85 and is_substrate_divergent:
            return InvariantScope.SUBSTRATE_CONDITIONED_INVARIANT
        elif cross_arch_fidelity >= 0.70:
            return InvariantScope.SUBSTRATE_CONDITIONED_INVARIANT
        else:
            return InvariantScope.MODEL_SPECIFIC
