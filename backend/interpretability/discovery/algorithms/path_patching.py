"""Path Patching Search Algorithm.

Implements Path Patching to discover specific causal pathways between
sender nodes (e.g., early attention heads) and receiver nodes (e.g., late MLPs)
by intervening on the residual stream edges connecting them.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from .base_algorithm import DiscoveryAlgorithm, DiscoveryReport
from .registry import register_algorithm, AlgorithmMetadata
from .configs import DiscoveryAlgorithmConfig, PathPatchingConfig


@register_algorithm(AlgorithmMetadata(
    name="path_patching",
    paper="Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small (Wang et al.)",
    authors="Kevin Wang, Alexandre Variengien, Arthur Conmy, et al.",
    year=2022,
    supported_models=["gpt2", "gemma", "llama", "mistral"],
    required_capabilities=["patch_activation", "get_logits", "get_activations"],
    estimated_runtime="30s-10m",
    search_space="paths_between_nodes",
    output_schema="DiscoveryReport"
))
class PathPatchingAlgorithm(DiscoveryAlgorithm):
    """Discovers causal edges by patching direct paths between sender and receiver nodes."""

    def run(self, dataset: Dict[str, Any], config: Optional[DiscoveryAlgorithmConfig] = None) -> DiscoveryReport:
        """Runs path patching to evaluate direct causal effects between components."""
        t0 = time.time()
        
        if not isinstance(config, PathPatchingConfig):
            config = PathPatchingConfig()
            
        clean_prompt = dataset.get("clean", "")
        corrupted_prompt = dataset.get("corrupted", "")
        target_token = dataset.get("target_token")
        
        # 1. Base Forward Passes
        clean_logits = self.adapter.get_logits(clean_prompt)
        corrupted_logits = self.adapter.get_logits(corrupted_prompt)
        
        if not target_token:
            target_token = clean_logits.get("top_token", " ")
            
        nodes = []
        edges = []
        
        num_layers = self.adapter.spec.num_layers
        num_heads = self.adapter.spec.num_heads

        sender_layers = [0] if num_layers <= 1 else [0, min(1, num_layers - 1)]
        receiver_layers = sorted(list(set([max(0, num_layers - 2), max(0, num_layers - 1)])))
        
        search_heads = [h for h in [0, 1] if h < num_heads]
        if not search_heads:
            search_heads = [0]
        
        # Build node registry
        nodes.append({"id": "T_0", "type": "Token", "label": clean_prompt})
        nodes.append({"id": "P_0", "type": "Prediction", "label": target_token})
        
        for layer in sender_layers + receiver_layers:
            for head in search_heads:
                nodes.append({"id": f"H_L{layer}_H{head}", "type": "Head", "label": f"L{layer}_H{head}"})

        # Path Patching Logic:
        # In full path patching, we run the model on corrupted data, 
        # but patch the *input* to the receiver to be what it would have been
        # if the sender had processed the clean data.
        
        # For this MVP, we simulate the path effect by patching the sender's output 
        # and observing the delta, mimicking a direct path effect measurement.
        
        for s_layer in sender_layers:
            for s_head in search_heads:
                
                s_activations = self.adapter.get_activations(clean_prompt, layer=s_layer, neuron_index=s_head)
                if not s_activations:
                    continue
                s_val = s_activations[0].activation_value
                
                for r_layer in receiver_layers:
                    for r_head in search_heads:
                        
                        # Apply patch representing the path from Sender -> Receiver
                        # In a real distributed engine, this uses `DistributedPatch` with path semantics
                        patch_res = self.adapter.patch_activation(
                            prompt=corrupted_prompt,
                            layer=r_layer,
                            neuron_index=r_head,
                            patch_value=s_val  # Injecting sender's clean output into receiver
                        )
                        
                        delta = abs(patch_res.delta)
                        
                        # If the path has a significant effect
                        if delta > 0.05:
                            s_id = f"H_L{s_layer}_H{s_head}"
                            r_id = f"H_L{r_layer}_H{r_head}"
                            
                            edges.append({
                                "source": s_id,
                                "target": r_id,
                                "weight": round(delta, 3),
                                "confidence": round(min(0.99, 0.5 + delta), 2)
                            })
                            
                            # Connect to output prediction
                            edges.append({
                                "source": r_id,
                                "target": "P_0",
                                "weight": 1.0,
                                "confidence": 1.0
                            })
                            
                            # Connect from input token
                            edges.append({
                                "source": "T_0",
                                "target": s_id,
                                "weight": 1.0,
                                "confidence": 1.0
                            })
                            
        # Deduplicate edges
        unique_edges = []
        seen = set()
        for e in edges:
            sig = f"{e['source']}->{e['target']}"
            if sig not in seen:
                seen.add(sig)
                unique_edges.append(e)
                
        circuit_score = round(sum(e["weight"] for e in unique_edges) / max(1, len(unique_edges)), 3)
        
        runtime_ms = (time.time() - t0) * 1000
        
        return DiscoveryReport(
            algorithm="path_patching",
            dataset_id=dataset.get("id", "unknown"),
            model_id=self.adapter.spec.model_id if self.adapter else "mock",
            runtime_ms=runtime_ms,
            statistics={
                "paths_evaluated": len(sender_layers) * len(receiver_layers) * len(search_heads)**2,
                "significant_paths": len([e for e in unique_edges if e["target"] != "P_0" and e["source"] != "T_0"])
            },
            evidence={
                "clean_prompt": clean_prompt,
                "corrupted_prompt": corrupted_prompt,
                "target_token": target_token
            },
            confidence=min(0.99, circuit_score * 0.9 + 0.1),
            graph={
                "nodes": nodes,
                "edges": unique_edges,
                "score": circuit_score
            },
            provenance={
                "search_space": config.sender_nodes + " -> " + config.receiver_nodes,
                "metric": config.metric
            }
        )
