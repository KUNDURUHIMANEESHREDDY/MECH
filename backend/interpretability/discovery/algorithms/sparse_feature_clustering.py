"""Sparse Feature Clustering Discovery Algorithm.

Groups SAE (Sparse Autoencoder) features into semantically meaningful clusters
by analyzing co-activation patterns across a dataset. Features that consistently
fire on similar inputs are clustered together, revealing latent structure in
the learned feature space.

Algorithm:
1. Run N dataset samples through the SAE encoder.
2. Build an (F x N) binary activation matrix (feature fires? yes/no).
3. Compute pairwise similarity between all features (cosine/jaccard/pearson).
4. Hierarchical agglomerative clustering to group features.
5. For each cluster, compute summary statistics and representative examples.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base_algorithm import DiscoveryAlgorithm, DiscoveryReport
from .registry import register_algorithm, AlgorithmMetadata
from .configs import DiscoveryAlgorithmConfig, SparseFeatureClusteringConfig


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors. Pure Python, zero dependencies."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def _jaccard_similarity(a: List[float], b: List[float]) -> float:
    """Jaccard similarity: |intersection| / |union| of firing sets."""
    both = sum(1 for x, y in zip(a, b) if x > 0 and y > 0)
    either = sum(1 for x, y in zip(a, b) if x > 0 or y > 0)
    return both / max(1, either)


def _pearson_correlation(a: List[float], b: List[float]) -> float:
    """Pearson correlation coefficient. Pure Python."""
    n = len(a)
    if n < 2:
        return 0.0
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    std_a = math.sqrt(sum((x - mean_a) ** 2 for x in a))
    std_b = math.sqrt(sum((y - mean_b) ** 2 for y in b))
    if std_a < 1e-12 or std_b < 1e-12:
        return 0.0
    return cov / (std_a * std_b)


SIMILARITY_FNS = {
    "cosine": _cosine_similarity,
    "jaccard": _jaccard_similarity,
    "pearson": _pearson_correlation,
}


@dataclass
class LinkageNode:
    """A node in the hierarchical clustering dendrogram."""
    cluster_id: int
    left: Optional[LinkageNode] = None
    right: Optional[LinkageNode] = None
    distance: float = 0.0
    features: List[int] = field(default_factory=list)
    level: int = 0  # Depth in the tree

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def _hierarchical_clustering_dendrogram(sim_matrix: List[List[float]], num_features: int) -> LinkageNode:
    """Performs hierarchical clustering and returns the root of the dendrogram.
    
    Uses average-linkage (UPGMA) for more balanced semantic hierarchies.
    """
    # Initialize: each feature is a leaf node
    nodes = {i: LinkageNode(cluster_id=i, features=[i]) for i in range(num_features)}
    
    # Distance matrix (1 - similarity)
    dist_matrix = [[1.0 - s for s in row] for row in sim_matrix]
    
    next_id = num_features
    active_ids = list(nodes.keys())

    while len(active_ids) > 1:
        # Find the pair with minimum distance
        min_dist = 2.0
        best_pair = (0, 1)
        
        for i_idx in range(len(active_ids)):
            for j_idx in range(i_idx + 1, len(active_ids)):
                id_i, id_j = active_ids[i_idx], active_ids[j_idx]

                # Average-linkage distance
                d_sum = 0.0
                count = 0
                for fi in nodes[id_i].features:
                    for fj in nodes[id_j].features:
                        d_sum += dist_matrix[fi][fj]
                        count += 1

                d_avg = d_sum / count
                if d_avg < min_dist:
                    min_dist = d_avg
                    best_pair = (id_i, id_j)
        
        # Merge best pair
        id_i, id_j = best_pair
        left, right = nodes[id_i], nodes[id_j]

        new_node = LinkageNode(
            cluster_id=next_id,
            left=left,
            right=right,
            distance=min_dist,
            features=left.features + right.features,
            level=max(left.level, right.level) + 1
        )

        nodes[next_id] = new_node
        active_ids.remove(id_i)
        active_ids.remove(id_j)
        active_ids.append(next_id)
        next_id += 1

    return nodes[active_ids[0]]


def _prune_dendrogram(root: LinkageNode, target_clusters: int) -> List[LinkageNode]:
    """Prunes the dendrogram to obtain a specific number of clusters."""
    clusters = [root]
    
    while len(clusters) < target_clusters:
        # Find the cluster with the largest distance to split
        # Only split if it's not a leaf
        splittable = [c for c in clusters if not c.is_leaf()]
        if not splittable:
            break

        to_split = max(splittable, key=lambda x: x.distance)
        clusters.remove(to_split)
        clusters.append(to_split.left)
        clusters.append(to_split.right)

    return clusters


@register_algorithm(AlgorithmMetadata(
    name="sparse_feature_clustering",
    paper="Sparse Feature Circuits (Marks et al.)",
    authors="Samuel Marks, Can Rager, Eric J. Michaud, et al.",
    year=2024,
    supported_models=["gpt2", "gemma", "llama", "mistral"],
    required_capabilities=["get_activations"],
    estimated_runtime="5s-30s",
    search_space="sae_features",
    output_schema="DiscoveryReport"
))
class SparseFeatureClusteringAlgorithm(DiscoveryAlgorithm):
    """Discovers hierarchical feature clusters by analyzing co-activation patterns."""

    def run(self, dataset: Dict[str, Any], config: Optional[DiscoveryAlgorithmConfig] = None) -> DiscoveryReport:
        """Runs hierarchical feature clustering on SAE activations."""
        t0 = time.time()
        
        if not isinstance(config, SparseFeatureClusteringConfig):
            config = SparseFeatureClusteringConfig()
        
        prompts = dataset.get("prompts", [])
        if not prompts:
            prompts = [{"clean": dataset.get("clean", "")}]
        
        num_features = config.num_features
        num_samples = min(config.num_samples, len(prompts))
        threshold = config.activation_threshold
        sim_fn = SIMILARITY_FNS.get(config.similarity_metric, _cosine_similarity)
        
        # ── Step 1: Build activation matrix (F x N) ──────────────────────
        activation_matrix: List[List[float]] = [[0.0] * num_samples for _ in range(num_features)]
        
        for sample_idx in range(num_samples):
            prompt_text = prompts[sample_idx].get("clean", "") if isinstance(prompts[sample_idx], dict) else str(prompts[sample_idx])
            
            for feat_idx in range(num_features):
                acts = self.adapter.get_activations(prompt_text, layer=0, neuron_index=feat_idx)
                if acts and len(acts) > 0:
                    val = acts[0].activation_value
                    activation_matrix[feat_idx][sample_idx] = 1.0 if abs(val) > threshold else 0.0
        
        # ── Step 2: Compute pairwise similarity ──────────────────────────
        sim_matrix: List[List[float]] = [[0.0] * num_features for _ in range(num_features)]
        for i in range(num_features):
            sim_matrix[i][i] = 1.0
            for j in range(i + 1, num_features):
                s = sim_fn(activation_matrix[i], activation_matrix[j])
                sim_matrix[i][j] = s
                sim_matrix[j][i] = s

        # ── Step 3: Hierarchical Clustering ──────────────────────────────
        dendrogram_root = _hierarchical_clustering_dendrogram(sim_matrix, num_features)
        
        # Obtain specific cluster levels (Concept Family vs Species)
        species_clusters = _prune_dendrogram(dendrogram_root, config.num_clusters)
        family_clusters = _prune_dendrogram(dendrogram_root, max(1, config.num_clusters // 4))
        
        # ── Step 4: Build cluster summaries ──────────────────────────────
        cluster_stats = []
        for i, node in enumerate(species_clusters):
            members = node.features
            if len(members) < config.min_cluster_size: continue
            
            # Cohesion
            pairs = [(members[i], members[j]) for i in range(len(members)) for j in range(i + 1, len(members))]
            cohesion = sum(sim_matrix[a][b] for a, b in pairs) / max(1, len(pairs)) if pairs else 1.0

            # Firing rate
            firing_rate = sum(sum(activation_matrix[m]) for m in members) / (len(members) * num_samples)
            
            cluster_stats.append({
                "cluster_id": node.cluster_id,
                "num_features": len(members),
                "members": members[:20],
                "cohesion": round(cohesion, 4),
                "avg_firing_rate": round(firing_rate, 4),
                "level": "species",
                "depth": node.level
            })

        # ── Step 5: Build Hierarchy Graph ────────────────────────────────
        nodes = []
        edges = []

        # Hierarchy: Link species to families
        for s in species_clusters:
            if len(s.features) < config.min_cluster_size: continue
            nodes.append({"id": f"S_{s.cluster_id}", "type": "Concept", "label": f"Concept {s.cluster_id}"})

            for f in family_clusters:
                if set(s.features).issubset(set(f.features)) and s.cluster_id != f.cluster_id:
                    nodes.append({"id": f"F_{f.cluster_id}", "type": "ConceptFamily", "label": f"Family {f.cluster_id}"})
                    edges.append({
                        "source": f"S_{s.cluster_id}", "target": f"F_{f.cluster_id}",
                        "relationship": "specializes"
                    })
                    break

        avg_cohesion = sum(s["cohesion"] for s in cluster_stats) / max(1, len(cluster_stats))
        runtime_ms = (time.time() - t0) * 1000
        
        return DiscoveryReport(
            algorithm="sparse_feature_clustering",
            dataset_id=dataset.get("id", "unknown"),
            model_id=self.adapter.spec.model_id if self.adapter else "mock",
            runtime_ms=runtime_ms,
            statistics={
                "num_features": num_features,
                "num_clusters_found": len(species_clusters),
                "avg_cohesion": round(avg_cohesion, 4),
                "hierarchy_depth": dendrogram_root.level
            },
            evidence={
                "clusters": cluster_stats,
                "dendrogram_distance": dendrogram_root.distance
            },
            confidence=min(0.99, avg_cohesion * 0.8 + 0.1),
            graph={
                "nodes": nodes,
                "edges": edges,
                "score": round(avg_cohesion, 3),
            },
            provenance={
                "search_space": "sae_features",
                "metric": config.similarity_metric,
                "clustering_type": "hierarchical_average_linkage"
            },
        )
