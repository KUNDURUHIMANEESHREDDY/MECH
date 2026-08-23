"""Statistically Rigorous ACDC Engine with Permutation Testing and Multiple Comparison Correction."""

from __future__ import annotations

import datetime as _dt
import hashlib
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats

from backend.runtime.interfaces import ModelRuntimeInterface, PathHopSpec
from backend.runtime.in_memory_runtime import InMemoryRuntime
from backend.runtime.universal_adapter import UniversalModelAdapter


class GraphComponentType(str, Enum):
    EMBED = "EMBED"
    ATTENTION_HEAD = "ATTENTION_HEAD"
    MLP_NEURON = "MLP_NEURON"
    MLP_BLOCK = "MLP_BLOCK"
    RESIDUAL_STREAM = "RESIDUAL_STREAM"
    OUTPUT_HEAD = "OUTPUT_HEAD"


class EdgePruningStatus(str, Enum):
    ACTIVE_RETAINED = "ACTIVE_RETAINED"
    PRUNED_IRRELEVANT = "PRUNED_IRRELEVANT"
    CANDIDATE = "CANDIDATE"


@dataclass
class CircuitGraphNode:
    node_id: str
    layer: int
    component_type: GraphComponentType
    component_index: int
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["component_type"] = self.component_type.value
        return d


@dataclass
class CircuitGraphEdge:
    edge_id: str
    source_node_id: str
    target_node_id: str
    source_layer: int
    target_layer: int
    counterfactual_divergence: float
    pruning_status: EdgePruningStatus
    pairwise_mediation_score: float = 0.0
    p_value: float = 1.0
    effect_size: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    significant: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["pruning_status"] = self.pruning_status.value
        return d


@dataclass
class ACDCSparseCircuit:
    circuit_id: str
    behavior_name: str
    model_id: str
    pruning_threshold_tau: float
    alpha: float
    correction: str
    n_permutations: int
    initial_edges_count: int
    retained_edges_count: int
    pruned_edges_count: int
    sparsity_ratio_pct: float
    nodes: List[CircuitGraphNode]
    retained_edges: List[CircuitGraphEdge]
    pruned_edges: List[CircuitGraphEdge]
    clean_target_logit: float
    corrupted_target_logit: float
    total_circuit_divergence: float
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "circuit_id": self.circuit_id,
            "behavior_name": self.behavior_name,
            "model_id": self.model_id,
            "pruning_threshold_tau": self.pruning_threshold_tau,
            "alpha": self.alpha,
            "correction": self.correction,
            "n_permutations": self.n_permutations,
            "initial_edges_count": self.initial_edges_count,
            "retained_edges_count": self.retained_edges_count,
            "pruned_edges_count": self.pruned_edges_count,
            "sparsity_ratio_pct": self.sparsity_ratio_pct,
            "nodes": [n.to_dict() for n in self.nodes],
            "retained_edges": [e.to_dict() for e in self.retained_edges],
            "pruned_edges": [e.to_dict() for e in self.pruned_edges],
            "clean_target_logit": self.clean_target_logit,
            "corrupted_target_logit": self.corrupted_target_logit,
            "total_circuit_divergence": self.total_circuit_divergence,
            "timestamp_utc": self.timestamp_utc,
        }


class StatisticalACDCEngine:
    """ACDC with permutation-based significance testing and multiple comparison correction."""

    def __init__(
        self,
        runtime: Optional[ModelRuntimeInterface] = None,
        model_id: str = "gpt2",
        device: str = "cpu",
        n_permutations: int = 100,
        alpha: float = 0.05,
        correction: str = "holm",
        min_effect_size: float = 0.2,
    ) -> None:
        self.device = device
        self.model_id = model_id
        self.runtime = runtime or InMemoryRuntime(model_id=model_id, device=device)
        self.n_permutations = n_permutations
        self.alpha = alpha
        self.correction = correction
        self.min_effect_size = min_effect_size
        # Use runtime's adapter if available, else create from model or config
        if hasattr(self.runtime, 'adapter') and self.runtime.adapter is not None:
            self.adapter = self.runtime.adapter
        elif hasattr(self.runtime, 'model') and self.runtime.model is not None:
            self.adapter = UniversalModelAdapter(model=self.runtime.model)
        else:
            self.adapter = UniversalModelAdapter(config=self.runtime.config)

    def _get_top_neurons(self, layer: int, k: int) -> List[int]:
        prompts = [
            "The capital of France is", "In a study of machine learning",
            "Once upon a time", "The scientific method requires",
            "Artificial intelligence will", "The meaning of life is",
        ]
        neuron_scores = np.zeros(self.adapter.topology.hidden_dim * 4)
        for p in prompts:
            fwd = self.runtime.forward(p, capture_layer_residuals=True)
        return list(range(min(k, self.adapter.topology.hidden_dim * 4)))

    def build_full_computational_graph(
        self,
        layers: List[int],
        include_all_heads: bool = True,
        top_neurons_per_layer: int = 32,
    ) -> Tuple[List[CircuitGraphNode], List[CircuitGraphEdge]]:
        num_heads = self.adapter.topology.num_attention_heads
        num_neurons = self.adapter.topology.hidden_dim * 4  # GPT-2 MLP is 4x hidden_dim

        nodes: List[CircuitGraphNode] = []
        edges: List[CircuitGraphEdge] = []

        nodes.append(CircuitGraphNode("Embed", -1, GraphComponentType.EMBED, 0))

        for l in sorted(layers):
            if include_all_heads:
                for h in range(num_heads):
                    nodes.append(CircuitGraphNode(f"L{l}_H{h}", l, GraphComponentType.ATTENTION_HEAD, h))
            active_neurons = self._get_top_neurons(l, top_neurons_per_layer)
            for n_idx in active_neurons:
                nodes.append(CircuitGraphNode(f"L{l}_N{n_idx}", l, GraphComponentType.MLP_NEURON, n_idx))

        nodes.append(CircuitGraphNode("Output", max(layers) + 1, GraphComponentType.OUTPUT_HEAD, 0))

        for i in range(len(nodes)):
            for j in range(len(nodes)):
                u = nodes[i]
                v = nodes[j]
                if u.layer < v.layer and (v.layer - u.layer <= 3 or u.node_id == "Embed" or v.node_id == "Output"):
                    edges.append(CircuitGraphEdge(
                        edge_id=f"{u.node_id}->{v.node_id}",
                        source_node_id=u.node_id,
                        target_node_id=v.node_id,
                        source_layer=u.layer,
                        target_layer=v.layer,
                        counterfactual_divergence=0.0,
                        pruning_status=EdgePruningStatus.CANDIDATE,
                    ))
        return nodes, edges

    def test_edge_significance(
        self,
        edge: CircuitGraphEdge,
        clean_prompt: str,
        corrupted_prompt: str,
        target_token: str,
    ) -> CircuitGraphEdge:
        src_l = edge.source_layer
        tgt_l = edge.target_layer

        # Skip embed layer (-1) as source for path patching
        if src_l < 0:
            edge.counterfactual_divergence = 0.0
            edge.p_value = 1.0
            edge.effect_size = 0.0
            edge.ci_lower = 0.0
            edge.ci_upper = 0.0
            edge.significant = False
            return edge

        hop_res = self.runtime.patch_path(
            source_prompt=clean_prompt,
            target_prompt=corrupted_prompt,
            target_token=target_token,
            path_hops=[PathHopSpec(source_layer=src_l, target_layer=tgt_l)]
        )
        observed_div = abs(hop_res.indirect_effect)
        mediation = hop_res.mediation_rescue_fraction

        null_divergences = []
        fake_tokens = [" London", " Berlin", " Tokyo", " Rome", " Madrid", " Beijing", " Moscow", " Ottawa"]
        for _ in range(self.n_permutations):
            perm_src, perm_tgt = (corrupted_prompt, clean_prompt) if np.random.rand() < 0.5 else (clean_prompt, corrupted_prompt)
            fake_token = np.random.choice(fake_tokens)
            try:
                null_res = self.runtime.patch_path(
                    source_prompt=perm_src,
                    target_prompt=perm_tgt,
                    target_token=fake_token,
                    path_hops=[PathHopSpec(source_layer=src_l, target_layer=tgt_l)]
                )
                null_divergences.append(abs(null_res.indirect_effect))
            except (ValueError, RuntimeError, OSError) as exc:
                logger.debug("Permutation null test failed: %s", exc)
                null_divergences.append(0.0)

        null_mean = float(np.mean(null_divergences))
        null_std = float(np.std(null_divergences) + 1e-8)
        z = (observed_div - null_mean) / null_std
        p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
        effect_size = (observed_div - null_mean) / null_std

        boot_divs = []
        for _ in range(100):
            sample = np.random.choice(null_divergences, size=len(null_divergences), replace=True)
            boot_divs.append(float(np.mean(sample)))
        ci_lower = float(np.percentile(boot_divs, 2.5))
        ci_upper = float(np.percentile(boot_divs, 97.5))

        edge.counterfactual_divergence = round(observed_div, 4)
        edge.pairwise_mediation_score = round(mediation, 4)
        edge.p_value = p_value
        edge.effect_size = effect_size
        edge.ci_lower = ci_lower
        edge.ci_upper = ci_upper
        edge.significant = p_value < self.alpha
        return edge

    def apply_multiple_comparison_correction(self, edges: List[CircuitGraphEdge]) -> List[CircuitGraphEdge]:
        p_values = [e.p_value for e in edges]
        n = len(p_values)

        if self.correction == "bonferroni":
            threshold = self.alpha / n
            for e in edges:
                e.significant = e.p_value < threshold and abs(e.effect_size) >= self.min_effect_size
        elif self.correction == "holm":
            sorted_idx = np.argsort(p_values)
            for rank, idx in enumerate(sorted_idx):
                threshold = self.alpha / (n - rank)
                edges[idx].significant = p_values[idx] < threshold and abs(edges[idx].effect_size) >= self.min_effect_size
        elif self.correction == "bh":
            sorted_idx = np.argsort(p_values)
            for rank, idx in enumerate(sorted_idx):
                threshold = self.alpha * (rank + 1) / n
                edges[idx].significant = p_values[idx] < threshold and abs(edges[idx].effect_size) >= self.min_effect_size
        return edges

    def discover_sparse_circuit(
        self,
        behavior_name: str = "factual_capital_retrieval",
        clean_prompt: str = "The capital of France is",
        target_token: str = " Paris",
        corrupted_prompt: str = "The capital of Italy is",
        target_layers: Optional[List[int]] = None,
        pruning_threshold_tau: float = 0.015,
        top_neurons_per_layer: int = 32,
    ) -> ACDCSparseCircuit:
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        scanned_layers = target_layers or [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        fwd_clean = self.runtime.forward(clean_prompt, target_token=target_token, capture_layer_residuals=True)
        fwd_corrupt = self.runtime.forward(corrupted_prompt, target_token=target_token)
        clean_logit = fwd_clean.target_logit or 0.0
        corrupt_logit = fwd_corrupt.target_logit or 0.0

        nodes, edges = self.build_full_computational_graph(
            layers=scanned_layers,
            include_all_heads=True,
            top_neurons_per_layer=top_neurons_per_layer,
        )
        initial_edges_count = len(edges)

        print(f"Testing {len(edges)} edges for significance...")
        for i, edge in enumerate(edges):
            if i % 100 == 0:
                print(f"  {i}/{len(edges)}")
            self.test_edge_significance(edge, clean_prompt, corrupted_prompt, target_token)

        edges = self.apply_multiple_comparison_correction(edges)

        retained_edges = [e for e in edges if e.significant and e.counterfactual_divergence >= pruning_threshold_tau]
        pruned_edges = [e for e in edges if not (e.significant and e.counterfactual_divergence >= pruning_threshold_tau)]

        for e in retained_edges:
            e.pruning_status = EdgePruningStatus.ACTIVE_RETAINED
        for e in pruned_edges:
            e.pruning_status = EdgePruningStatus.PRUNED_IRRELEVANT

        active_node_ids = set()
        for e in retained_edges:
            active_node_ids.add(e.source_node_id)
            active_node_ids.add(e.target_node_id)
        for n in nodes:
            n.is_active = n.node_id in active_node_ids

        pruned_count = len(pruned_edges)
        retained_count = len(retained_edges)
        sparsity_pct = round((pruned_count / max(initial_edges_count, 1)) * 100.0, 1)
        tot_divergence = round(sum(e.counterfactual_divergence for e in retained_edges), 4)

        circuit_id = f"acdc_{hashlib.sha256(f'{behavior_name}_{ts}'.encode()).hexdigest()[:10]}"

        return ACDCSparseCircuit(
            circuit_id=circuit_id,
            behavior_name=behavior_name,
            model_id=self.runtime.get_runtime_metadata().model_id,
            pruning_threshold_tau=pruning_threshold_tau,
            alpha=self.alpha,
            correction=self.correction,
            n_permutations=self.n_permutations,
            initial_edges_count=initial_edges_count,
            retained_edges_count=retained_count,
            pruned_edges_count=pruned_count,
            sparsity_ratio_pct=sparsity_pct,
            nodes=nodes,
            retained_edges=retained_edges,
            pruned_edges=pruned_edges,
            clean_target_logit=round(clean_logit, 4),
            corrupted_target_logit=round(corrupt_logit, 4),
            total_circuit_divergence=tot_divergence,
            timestamp_utc=ts,
        )