"""Discovery Algorithms Package."""

from .base_algorithm import DiscoveryAlgorithm, DiscoveryReport
from .configs import (
    DiscoveryAlgorithmConfig,
    ACDCConfig,
    PathPatchingConfig,
    SparseFeatureClusteringConfig,
    CausalScrubbingConfig,
    AttributionPatchingConfig,
    TranscoderConfig,
    FeatureUniversalityConfig,
)
from .registry import register_algorithm, get_algorithm, get_algorithm_names, get_algorithm_metadata, AlgorithmMetadata

# Import to trigger registration
from . import acdc
from . import path_patching
from . import sparse_feature_clustering
from . import causal_scrubbing
from . import attribution_patching
from . import transcoders
from . import feature_universality

__all__ = [
    "DiscoveryAlgorithm",
    "DiscoveryReport",
    "DiscoveryAlgorithmConfig",
    "ACDCConfig",
    "PathPatchingConfig",
    "SparseFeatureClusteringConfig",
    "CausalScrubbingConfig",
    "AttributionPatchingConfig",
    "TranscoderConfig",
    "FeatureUniversalityConfig",
    "register_algorithm",
    "get_algorithm",
    "get_algorithm_names",
    "get_algorithm_metadata",
    "AlgorithmMetadata",
]




