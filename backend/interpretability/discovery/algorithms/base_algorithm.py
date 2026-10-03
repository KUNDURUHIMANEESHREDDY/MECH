"""Abstract Base Class for Discovery Algorithms.

All discovery algorithms (ACDC, Path Patching, Sparse Feature Search, etc.)
must inherit from this class to be executable by the Discovery Engine.
"""

from __future__ import annotations

import datetime as _dt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.science.models.adapter_base import ModelAdapter
from .configs import DiscoveryAlgorithmConfig


@dataclass
class DiscoveryReport:
    """Standardized output schema for all discovery algorithms."""
    algorithm: str
    dataset_id: str
    model_id: str
    runtime_ms: float
    statistics: Dict[str, Any]
    evidence: Dict[str, Any]
    confidence: float
    graph: Dict[str, Any]
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "dataset_id": self.dataset_id,
            "model_id": self.model_id,
            "runtime_ms": self.runtime_ms,
            "statistics": self.statistics,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "graph": self.graph,
            "artifacts": self.artifacts,
            "provenance": self.provenance,
            "timestamp": self.timestamp,
        }


class DiscoveryAlgorithm(ABC):
    """Abstract interface for a mechanistic discovery algorithm."""

    #: False in any subclass that does not implement the method its name and
    #: paper citation advertise. The reported graph and confidence then
    #: describe a simulation, not a measurement, and must say so.
    implements_published_method: bool = True

    #: Why the published method is not implemented, for subclasses that set the
    #: flag above to False.
    not_implemented_reason: str = ""

    def __init__(self, adapter: ModelAdapter) -> None:
        self.adapter = adapter

    def provenance_block(self) -> Dict[str, Any]:
        """Provenance for a DiscoveryReport derived from this algorithm's state.

        The `source` key is load-bearing, not decorative. `evidence_policy.
        provenance_of` reads a report's `provenance` dict by looking for
        `source`, `kind`, `type` or `status`, in that order, and returns
        "unavailable" if it finds none. A block that nests the label under
        `provenance` is therefore invisible to the evidence gate.

        That is why a live ACDC run reported `provenance_of(...) == "unavailable"`
        while carrying `"provenance": "live"` in the very same dict: the reader
        found no key it recognised and defaulted. The nested key is kept too,
        because it is what a human reads, but `source` is what the policy sees.

        For the unimplemented case the answer happens to coincide -- "reference"
        is not "live" either way -- so the shape had been wrong without ever
        producing a wrong *unavailable* verdict, only an unusable live one. That
        asymmetry is why it survived.
        """
        block: Dict[str, Any] = {
            "algorithm": self.__class__.__name__,
            "implements_published_method": self.implements_published_method,
        }
        if not self.implements_published_method:
            block.update({
                "source": "reference",
                "status": "unavailable",
                "measured": False,
                "provenance": "reference",
                "validation_eligible": False,
                "publication_eligible": False,
                "reason": self.not_implemented_reason or (
                    f"{self.__class__.__name__} does not implement the method "
                    "it is named for; its output is a simulation."),
            })
        return block

    @abstractmethod
    def run(self, dataset: Dict[str, Any], config: Optional[DiscoveryAlgorithmConfig] = None) -> DiscoveryReport:
        """Executes the discovery algorithm.
        
        Args:
            dataset: A dictionary containing the dataset (e.g., clean/corrupted prompts).
            config: Algorithm-specific configuration parameters.
            
        Returns:
            A unified DiscoveryReport object.
        """
        ...

