"""Path Patching Search Algorithm.

Discovers causal pathways between sender nodes (early attention heads) and
receiver nodes (late heads) by intervening on the residual-stream edges that
connect them.

The procedure requires *two simultaneous interventions* on the corrupted run:
the sender's output is frozen at its clean value, and the receiver reads the
clean value that arrived at its position from the sender. Only then is the
change in logit difference attributable to the direct route rather than to the
sender's ordinary contribution.

The previous implementation patched just the receiver with the sender's
activation while the sender ran normally, which measures the sender's *total*
effect, not the path effect -- and it attached a fabricated `0.5 + delta`
confidence to every edge it drew. That is why `implements_published_method` was
False.

Real path patching now lives in `live_measure.path_patch`, which installs both
hooks on the live TransformerLens model. This algorithm delegates to it when
the dataset supplies the IOI/subject token ids the logit difference is measured
against. Without those, it refuses rather than reporting a conflated number.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set

from .base_algorithm import DiscoveryAlgorithm, DiscoveryReport
from .registry import register_algorithm, AlgorithmMetadata
from .configs import DiscoveryAlgorithmConfig, PathPatchingConfig


@register_algorithm(AlgorithmMetadata(
    name="path_patching",
    paper="Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small (Wang et al.)",
    authors="Kevin Wang, Alexandre Variengien, Arthur Conmy, et al.",
    year=2022,
    supported_models=["gpt2"],
    required_capabilities=["patch_activation", "get_logits", "get_activations"],
    estimated_runtime="30s-10m",
    search_space="paths_between_nodes",
    output_schema="DiscoveryReport"
))
class PathPatchingAlgorithm(DiscoveryAlgorithm):
    """True path patching via `live_measure.path_patch`, when reachable.

    A single-site `patch_activation` cannot express this: the procedure needs
    the sender frozen *and* the receiver swapped in one forward pass. So the
    generic adapter path is not a fallback that produces a slightly-worse
    number -- it produces a different quantity entirely, and this class reports
    that plainly rather than reporting edges it cannot justify.
    """

    implements_published_method = True
    not_implemented_reason = ""

    def run(self, dataset: Dict[str, Any],
            config: Optional[DiscoveryAlgorithmConfig] = None) -> DiscoveryReport:
        """Evaluate isolated causal paths between candidate senders/receivers."""
        t0 = time.time()

        if not isinstance(config, PathPatchingConfig):
            config = PathPatchingConfig()

        clean_prompt = dataset.get("clean", "")
        corrupted_prompt = dataset.get("corrupted", "")
        target_token = dataset.get("target_token") or " "
        io_id = dataset.get("io_id")
        subj_id = dataset.get("subject_id")

        num_layers = self.adapter.spec.num_layers
        num_heads = self.adapter.spec.num_heads
        sender_layers = [0] if num_layers <= 1 else [0, min(1, num_layers - 1)]
        receiver_layers = sorted({max(0, num_layers - 2), max(0, num_layers - 1)})
        search_heads = [h for h in (0, 1) if h < num_heads] or [0]

        if io_id is None or subj_id is None:
            return DiscoveryReport(
                algorithm="path_patching",
                dataset_id=dataset.get("id", "unknown"),
                model_id=self.adapter.spec.model_id if self.adapter else "mock",
                runtime_ms=round((time.time() - t0) * 1000, 2),
                statistics={"sender_receivers_evaluated": 0},
                evidence={
                    "clean_prompt": clean_prompt,
                    "corrupted_prompt": corrupted_prompt,
                    "target_token": target_token,
                },
                confidence=0.0,
                graph={"nodes": [], "edges": []},
                provenance={
                    "implements_published_method": True,
                    "status": "unavailable",
                    "measured": False,
                    "provenance": "unavailable",
                    "validation_eligible": False,
                    "publication_eligible": False,
                    "search_space": (
                        config.sender_nodes + " -> " + config.receiver_nodes),
                    "metric": config.metric,
                    "reason": (
                        "The dataset did not supply io_id/subject_id, so there "
                        "is no logit difference to isolate a path against. The "
                        "only measurement available here is a single receiver "
                        "patch, which conflates the path with the sender's "
                        "total effect, so no path is claimed."
                    ),
                },
            )

        from backend.interpretability.discovery.live_measure import (
            head_label,
            parse_head,
            path_patch,
        )

        measurements: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        graph_nodes: List[Dict[str, Any]] = [
            {"id": "T_0", "type": "Token", "label": clean_prompt},
            {"id": "P_0", "type": "Prediction", "label": target_token},
        ]
        node_ids: Set[str] = {"T_0", "P_0"}

        def ensure_node(layer: int, head: int) -> str:
            node_id = f"H_L{layer}_H{head}"
            if node_id not in node_ids:
                node_ids.add(node_id)
                graph_nodes.append({
                    "id": node_id, "type": "Head",
                    "label": head_label(layer, head),
                })
            return node_id

        for s_layer in sender_layers:
            for s_head in search_heads:
                sender_id = ensure_node(s_layer, s_head)
                receivers = [(rl, rh) for rl in receiver_layers
                             for rh in search_heads]
                if not receivers:
                    continue
                measured = path_patch(
                    clean_prompt, corrupted_prompt,
                    int(io_id), int(subj_id), (s_layer, s_head), receivers)
                measurements.append(measured)

                for row in measured["receivers"]:
                    effect = row.get("path_effect")
                    if effect is None:
                        continue
                    parsed = parse_head(row["receiver"]) or (0, 0)
                    receiver_id = ensure_node(parsed[0], parsed[1])
                    # Report an edge only when the isolated route moved the
                    # logit gap by more than the noise threshold.
                    if abs(effect) > 0.05:
                        edges.append({
                            "source": sender_id,
                            "target": receiver_id,
                            "weight": round(abs(effect), 4),
                            "confidence": round(min(0.99, abs(effect)), 4),
                            "path_effect": effect,
                        })

        # Deduplicate edges, keeping the strongest measured path effect.
        strongest: Dict[str, Dict[str, Any]] = {}
        for edge in edges:
            key = f"{edge['source']}->{edge['target']}"
            existing = strongest.get(key)
            if existing is None or abs(edge["path_effect"]) > abs(existing["path_effect"]):
                strongest[key] = edge
        unique_edges = list(strongest.values())

        all_effects = [abs(r["path_effect"])
                       for m in measurements for r in m["receivers"]
                       if r.get("path_effect") is not None]
        circuit_score = round(
            sum(all_effects) / len(all_effects), 4) if all_effects else 0.0

        return DiscoveryReport(
            algorithm="path_patching",
            dataset_id=dataset.get("id", "unknown"),
            model_id=self.adapter.spec.model_id if self.adapter else "mock",
            runtime_ms=round((time.time() - t0) * 1000, 2),
            statistics={
                "paths_evaluated": sum(len(m["receivers"]) for m in measurements),
                "senders_evaluated": len(measurements),
                "significant_paths": len(unique_edges),
            },
            evidence={
                "clean_prompt": clean_prompt,
                "corrupted_prompt": corrupted_prompt,
                "target_token": target_token,
                "measurements": measurements,
            },
            # A measured quantity: the mean absolute isolated path effect.
            # Not a rescaled graph score, and not floored at some constant.
            confidence=round(min(0.99, circuit_score), 4),
            graph={
                "nodes": graph_nodes,
                "edges": unique_edges,
                "score": circuit_score,
                "score_meaning": "mean_absolute_isolated_path_effect",
            },
            provenance={
                "implements_published_method": True,
                "measured": True,
                "provenance": "live",
                "sender_frozen": True,
                "receiver_input_swapped": True,
                "isolates_direct_path": True,
                "search_space": (
                    config.sender_nodes + " -> " + config.receiver_nodes),
                "metric": config.metric,
                "confidence_basis": "mean absolute isolated path effect",
                "paper_citation": (
                    "Wang et al. 2022: path patching freezes the sender and "
                    "swaps the receiver input in one corrupted forward pass"),
            },
        )
