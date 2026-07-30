"""Typed Configurations for Discovery Algorithms."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DiscoveryAlgorithmConfig:
    """Base configuration for all discovery algorithms."""
    max_iterations: int = 100
    batch_size: int = 1
    device: str = "cpu"
    seed: int = 42


@dataclass
class ACDCConfig(DiscoveryAlgorithmConfig):
    """Configuration for the ACDC Algorithm."""
    threshold: float = 0.05
    search_space: str = "attention_heads" # Options: attention_heads, mlps, neurons
    metric: str = "logit_difference"

@dataclass
class PathPatchingConfig(DiscoveryAlgorithmConfig):
    """Configuration for the Path Patching Algorithm."""
    sender_nodes: str = "all_heads"
    receiver_nodes: str = "all_heads"
    metric: str = "logit_difference"
    patch_type: str = "activation" # Options: activation, attention_pattern

@dataclass
class SparseFeatureClusteringConfig(DiscoveryAlgorithmConfig):
    """Configuration for the Sparse Feature Clustering Algorithm."""
    num_features: int = 512          # Number of SAE features to cluster
    num_clusters: int = 16           # Target number of clusters
    similarity_metric: str = "cosine"  # Options: cosine, jaccard, pearson
    min_cluster_size: int = 3        # Minimum features per cluster
    activation_threshold: float = 0.1  # Min activation to count as 'firing'
    num_samples: int = 256           # Number of dataset samples for co-activation


@dataclass
class CausalScrubbingConfig(DiscoveryAlgorithmConfig):
    """Configuration for the Causal Scrubbing Algorithm (Redwood Research)."""
    equivalence_class: str = "token_type"  # Options: token_type, position, semantic_category
    scrub_paths: str = "non_circuit_heads"  # Options: non_circuit_heads, mlp_outputs, all_unmapped
    resample_count: int = 10               # Number of resampled reference runs per sample
    metric: str = "behavior_preservation"  # Options: behavior_preservation, kl_divergence
    tolerance: float = 0.05                 # Max allowed performance loss for hypothesis validation


@dataclass
class AttributionPatchingConfig(DiscoveryAlgorithmConfig):
    """Configuration for the Attribution Patching Algorithm (Neel Nanda, AtP)."""
    approximation_order: str = "first_order_taylor"  # Options: first_order_taylor, second_order
    search_space: str = "all_components"            # Options: all_components, attention_heads, mlps, neurons
    metric: str = "logit_difference"                 # Options: logit_difference, kl_divergence, probability
    top_k: int = 20                                  # Return top K highest attributed components
    threshold: float = 0.01                          # Minimum attribution score to include in graph


@dataclass
class TranscoderConfig(DiscoveryAlgorithmConfig):
    """Configuration for Transcoder Dictionary Learning (Anthropic, 2024)."""
    dict_size: int = 512                   # Number of transcoder features
    l1_alpha: float = 1e-3                 # L1 sparsity penalty coefficient
    source_layer: int = 4                  # Input layer to transcode from
    target_layer: int = 5                  # Output layer to transcode to
    reconstruction_target: float = 0.95    # Target fraction of variance explained (FVE)


@dataclass
class FeatureUniversalityConfig(DiscoveryAlgorithmConfig):
    """Configuration for Cross-Model / Cross-Layer Feature Universality Analysis."""
    target_models: List[str] = field(default_factory=lambda: ["gpt2", "gemma"])  # Models/layers to align
    matching_metric: str = "cosine_sim"     # Options: cosine_sim, jaccard_overlap, mutual_information
    min_alignment_score: float = 0.65       # Minimum threshold to declare a universal feature pair
    top_matches: int = 15                   # Top universal feature alignments to report


