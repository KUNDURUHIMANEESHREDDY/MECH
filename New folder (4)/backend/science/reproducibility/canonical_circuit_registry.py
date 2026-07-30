"""Canonical Circuit Registry.

Stores gold-standard circuit definitions (nodes and edges) for validating
discovery algorithms. Supports required vs optional components.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class CircuitReference:
    """Gold standard reference for a mechanistic circuit."""
    circuit_id: str
    paper_title: str
    model_id: str
    dataset_name: str
    required_nodes: Set[str]
    optional_nodes: Set[str] = field(default_factory=set)
    required_edges: Set[Tuple[str, str]] = field(default_factory=set)
    optional_edges: Set[Tuple[str, str]] = field(default_factory=set)
    expected_functional_recovery: float = 0.90
    metric_type: str = "logit_diff"


# Canonical GPT-2 Small IOI Circuit (Wang et al. 2022)
IOI_CANONICAL = CircuitReference(
    circuit_id="ioi_gpt2_small",
    paper_title="Interpretability in the Wild (2022)",
    model_id="gpt2-small",
    dataset_name="IOI-100",
    # Primary Name Movers and critical components
    required_nodes={
        "L9H6", "L9H9", "L10H0", "L10H7", # Name Movers
        "L7H3", "L8H6",                   # S-Inhibition
        "L5H1", "L5H5",                   # Induction
        "L0H1", "L0H10"                   # Duplicate Token
    },
    optional_nodes={
        "L11H10", "L10H10", "L11H2",      # Backup Name Movers / Negative NM
        "L6H9"                             # Weak Induction
    },
    # Edge examples: (Source, Target)
    required_edges={
        ("L0H1", "L5H1"),
        ("L5H1", "L7H3"),
        ("L7H3", "L9H9")
    },
    expected_functional_recovery=0.97
)

class CanonicalCircuitRegistry:
    """Registry engine for accessing ground-truth circuits."""

    def __init__(self) -> None:
        self._circuits = {
            "ioi": IOI_CANONICAL
        }

    def get_circuit(self, circuit_id: str) -> Optional[CircuitReference]:
        return self._circuits.get(circuit_id)

    def list_circuits(self) -> List[str]:
        return list(self._circuits.keys())
