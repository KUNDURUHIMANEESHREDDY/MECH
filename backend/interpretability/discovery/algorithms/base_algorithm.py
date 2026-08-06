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

    def __init__(self, adapter: ModelAdapter) -> None:
        self.adapter = adapter

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

