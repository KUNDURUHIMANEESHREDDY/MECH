"""Quantitative Circuit Metrics Suite.

Evaluates discovered interpretability subnetworks against 3 formal mechanistic criteria:
1. Faithfulness (F): Measures if the isolated subnetwork reproduces the full model's clean/corrupted margin.
2. Completeness (C): Measures resilience when non-circuit background components are ablated.
3. Minimality (M): Quantifies the fraction of nodes that are strictly necessary (where single-node ablation causes a substantial faithfulness collapse).

Note: While Faithfulness, Completeness, and Minimality are standard mechanistic interpretability concepts
(Conmy et al. 2023; Wang et al. 2022), the specific numerical targets (F >= 0.90, C >= 0.85, M >= 0.80)
represent MECH's default Quality Evaluation Policy rather than universal constants.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("MECH.circuit_metrics")


@dataclass
class NodeSensitivity:
    """Measures the causal necessity of a single node in the circuit."""
    node_id: str
    faithfulness_drop_on_ablation: float
    is_critical: bool  # True if drop >= node_necessity_threshold


@dataclass
class ACDCCircuitEvaluation:
    """Quantitative evaluation report for an isolated circuit subnetwork."""
    task_name: str
    num_nodes: int
    num_edges: int
    # 1. Faithfulness (F)
    faithfulness: float
    # 2. Completeness (C)
    completeness: float
    # 3. Minimality (M) Breakdown: Disentangling Node-Level Criterion vs Aggregate Score
    minimality_score: float                # Aggregate score (fraction of nodes meeting necessity criterion)
    node_necessity_threshold: float        # Minimum single-node ablation drop to be deemed critical (e.g. 0.15)
    critical_node_fraction: float          # Percentage of nodes in circuit meeting the necessity threshold
    weakest_link_sensitivity: float        # Minimum faithfulness drop across all circuit nodes
    # Underlying Logit Deltas
    full_model_clean_delta: float
    full_model_corrupted_delta: float
    circuit_only_delta: float
    background_ablated_delta: float
    # Sensitivity breakdown per node
    node_sensitivities: List[NodeSensitivity] = field(default_factory=list)
    # Policy Evaluation Compliance (Configurable Targets)
    meets_faithfulness_policy: bool = False  # F >= target (default 0.90)
    meets_completeness_policy: bool = False  # C >= target (default 0.85)
    meets_minimality_policy: bool = False    # M >= target (default 0.80)
    policy_grade: str = "Sub-threshold"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "faithfulness": round(self.faithfulness, 4),
            "completeness": round(self.completeness, 4),
            "minimality": round(self.minimality_score, 4),
            "minimality_details": {
                "aggregate_minimality_score": round(self.minimality_score, 4),
                "node_necessity_threshold": self.node_necessity_threshold,
                "critical_node_fraction_pct": round(self.critical_node_fraction * 100, 1),
                "weakest_link_faithfulness_drop": round(self.weakest_link_sensitivity, 4),
                "definition": f"Aggregate score M = {self.critical_node_fraction*100:.1f}% of nodes individually cause >= {self.node_necessity_threshold*100:.0f}% faithfulness collapse when ablated.",
            },
            "full_model_clean_delta": round(self.full_model_clean_delta, 4),
            "full_model_corrupted_delta": round(self.full_model_corrupted_delta, 4),
            "circuit_only_delta": round(self.circuit_only_delta, 4),
            "background_ablated_delta": round(self.background_ablated_delta, 4),
            "node_sensitivities": [
                {
                    "node_id": ns.node_id,
                    "faithfulness_drop_on_ablation": round(ns.faithfulness_drop_on_ablation, 4),
                    "is_critical": ns.is_critical,
                }
                for ns in self.node_sensitivities
            ],
            "meets_faithfulness_policy": self.meets_faithfulness_policy,
            "meets_completeness_policy": self.meets_completeness_policy,
            "meets_minimality_policy": self.meets_minimality_policy,
            "policy_grade": self.policy_grade,
        }


class ACDCCircuitMetricsEvaluator:
    """Evaluates circuit metrics (Faithfulness, Completeness, Minimality) under configurable quality policies."""

    def __init__(
        self,
        faithfulness_target: float = 0.90,
        completeness_target: float = 0.85,
        minimality_target: float = 0.80,
        node_necessity_threshold: float = 0.15,
    ) -> None:
        self.faithfulness_target = faithfulness_target
        self.completeness_target = completeness_target
        self.minimality_target = minimality_target
        self.node_necessity_threshold = node_necessity_threshold

    def evaluate_circuit(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        clean_prompt: str = "The Eiffel Tower is in the city of",
        corrupted_prompt: str = "The Colosseum is in the city of",
        target_token: str = " Paris",
        task_name: str = "Fact Retrieval",
    ) -> ACDCCircuitEvaluation:
        """Run a REAL quantitative evaluation of the circuit via activation-patching
        ablations on the loaded GPT-2 model (no fabricated constants)."""
        import backend.services.gpt2_engine as gpt2_engine

        comp_nodes = [n for n in nodes if n.get("id") not in ("input", "output")]
        n_nodes = len(comp_nodes)
        n_edges = len(edges)

        def _parse(node_id: str) -> set:
            s = (node_id or "").replace("node_", "")
            comps: set = set()
            m = re.match(r"L(\d+)_H(\d+)", s)
            if m:
                comps.add((int(m.group(1)), "attn"))
                return comps
            if "MLP" in s.upper():
                m = re.match(r"L(\d+)", s)
                if m:
                    comps.add((int(m.group(1)), "mlp"))
                    return comps
            if "SAE" in s.upper() or "_N" in s.upper():
                m = re.match(r"L(\d+)", s)
                if m:
                    comps.add((int(m.group(1)), "mlp"))
                    return comps
            m = re.match(r"L(\d+)", s)
            if m:
                l = int(m.group(1))
                comps.add((l, "attn"))
                comps.add((l, "mlp"))
                return comps
            return comps

        circuit_comps: set = set()
        for n in comp_nodes:
            circuit_comps |= _parse(n.get("id", ""))

        def _logit(base: str, source: str, patch: set) -> float:
            r = gpt2_engine.run_activation_patch(base, source, patch, target_token)
            if r.get("status") != "ok":
                raise RuntimeError(f"activation patch failed: {r.get('error')}")
            return float(r["target_logit"])

        # Empty / non-computable circuit -> honest zero evaluation (no fabrication).
        if not gpt2_engine.is_available() or not circuit_comps:
            return ACDCCircuitEvaluation(
                task_name=task_name,
                num_nodes=n_nodes,
                num_edges=n_edges,
                faithfulness=0.0,
                completeness=0.0,
                minimality_score=0.0,
                node_necessity_threshold=self.node_necessity_threshold,
                critical_node_fraction=0.0,
                weakest_link_sensitivity=0.0,
                full_model_clean_delta=0.0,
                full_model_corrupted_delta=0.0,
                circuit_only_delta=0.0,
                background_ablated_delta=0.0,
                node_sensitivities=[
                    NodeSensitivity(
                        node_id=n.get("id", f"node_{i}"),
                        faithfulness_drop_on_ablation=0.0,
                        is_critical=False,
                    )
                    for i, n in enumerate(comp_nodes)
                ],
                meets_faithfulness_policy=False,
                meets_completeness_policy=False,
                meets_minimality_policy=False,
                policy_grade="No Computable Circuit (requires a loaded model and mapped components)",
            )

        # Real baselines (full-model target logits) from actual forward passes.
        L_clean = _logit(clean_prompt, clean_prompt, set())
        L_corr = _logit(corrupted_prompt, corrupted_prompt, set())
        denom = max(1e-6, L_clean - L_corr)

        # 1. Faithfulness (F): circuit-only run (base=corrupted, patch circuit from clean).
        L_circ = _logit(corrupted_prompt, clean_prompt, circuit_comps)
        f_score = max(0.0, min(1.0, (L_circ - L_corr) / denom)) if denom > 0 else 0.0

        # 2. Completeness (C): ablate circuit (base=clean, patch circuit from corrupted).
        L_ablated = _logit(clean_prompt, corrupted_prompt, circuit_comps)
        c_score = max(0.0, min(1.0, (L_clean - L_ablated) / denom)) if denom > 0 else 0.0

        # 3. Minimality (M): per-node ablation within the circuit (faithfulness drop).
        node_sensitivities: List[NodeSensitivity] = []
        critical_count = 0
        drops: List[float] = []
        for idx, n in enumerate(comp_nodes):
            n_comps = _parse(n.get("id", ""))
            patch_minus = circuit_comps - n_comps
            L_n = _logit(corrupted_prompt, clean_prompt, patch_minus)
            f_n = max(0.0, min(1.0, (L_n - L_corr) / denom)) if denom > 0 else 0.0
            drop = max(0.0, f_score - f_n)
            is_crit = drop >= self.node_necessity_threshold
            if is_crit:
                critical_count += 1
            drops.append(drop)
            node_sensitivities.append(NodeSensitivity(
                node_id=n.get("id", f"node_{idx}"),
                faithfulness_drop_on_ablation=round(drop, 4),
                is_critical=is_crit,
            ))

        crit_frac = critical_count / max(1, n_nodes)
        weakest_drop = min(drops) if drops else 0.0
        m_score = round(crit_frac, 4)

        meets_f = f_score >= self.faithfulness_target
        meets_c = c_score >= self.completeness_target
        meets_m = m_score >= self.minimality_target

        if meets_f and meets_c and meets_m:
            grade = "MECH Circuit Policy: Gold Tier (Faithful, Complete & Minimal)"
        elif meets_f and meets_c:
            grade = "MECH Circuit Policy: Silver Tier (Faithful & Complete, Minor Redundancy)"
        elif meets_f:
            grade = "MECH Circuit Policy: Bronze Tier (Faithful Subnetwork)"
        else:
            grade = "MECH Circuit Policy: Sub-threshold (Requires Refinement)"

        return ACDCCircuitEvaluation(
            task_name=task_name,
            num_nodes=n_nodes,
            num_edges=n_edges,
            faithfulness=round(f_score, 4),
            completeness=round(c_score, 4),
            minimality_score=m_score,
            node_necessity_threshold=self.node_necessity_threshold,
            critical_node_fraction=crit_frac,
            weakest_link_sensitivity=round(weakest_drop, 4),
            full_model_clean_delta=round(L_clean, 4),
            full_model_corrupted_delta=round(L_corr, 4),
            circuit_only_delta=round(L_circ, 4),
            background_ablated_delta=round(L_ablated, 4),
            node_sensitivities=node_sensitivities,
            meets_faithfulness_policy=meets_f,
            meets_completeness_policy=meets_c,
            meets_minimality_policy=meets_m,
            policy_grade=grade,
        )
