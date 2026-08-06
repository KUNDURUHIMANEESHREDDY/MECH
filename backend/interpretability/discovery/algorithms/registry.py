"""Discovery Algorithm Registry.

Provides a decorator and registry for dynamically loading discovery algorithms
with rich metadata (paper, search space, requirements, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Type, Optional, Any

from .base_algorithm import DiscoveryAlgorithm
from backend.science.models.adapter_base import ModelAdapter


@dataclass
class AlgorithmMetadata:
    name: str
    paper: str
    authors: str
    year: int
    supported_models: list[str]
    required_capabilities: list[str]
    estimated_runtime: str
    search_space: str
    output_schema: str = "DiscoveryReport"


_ALGORITHM_REGISTRY: Dict[str, Type[DiscoveryAlgorithm]] = {}
_METADATA_REGISTRY: Dict[str, AlgorithmMetadata] = {}


def register_algorithm(metadata: AlgorithmMetadata):
    """Decorator to register a new discovery algorithm with metadata.
    
    Example:
        @register_algorithm(AlgorithmMetadata(name="acdc", ...))
        class ACDCAlgorithm(DiscoveryAlgorithm):
            ...
    """
    def wrapper(cls: Type[DiscoveryAlgorithm]) -> Type[DiscoveryAlgorithm]:
        _ALGORITHM_REGISTRY[metadata.name] = cls
        _METADATA_REGISTRY[metadata.name] = metadata
        return cls
    return wrapper


def get_algorithm_metadata(name: str) -> AlgorithmMetadata:
    if name not in _METADATA_REGISTRY:
        raise ValueError(f"Metadata for '{name}' not found.")
    return _METADATA_REGISTRY[name]


def get_algorithm_names() -> list[str]:
    """Returns a list of registered algorithm names."""
    return list(_ALGORITHM_REGISTRY.keys())


def get_algorithm(name: str, adapter: ModelAdapter) -> DiscoveryAlgorithm:
    """Instantiates and returns a registered algorithm.
    
    Raises:
        ValueError: If the algorithm is not registered.
    """
    if name not in _ALGORITHM_REGISTRY:
        raise ValueError(f"Algorithm '{name}' not found in registry. Available: {get_algorithm_names()}")
    return _ALGORITHM_REGISTRY[name](adapter)

