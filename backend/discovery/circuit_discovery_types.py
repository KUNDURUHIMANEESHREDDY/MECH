"""Epistemic Data Contracts and Types for Autonomous Circuit Discovery.

Enforces strict epistemic classification distinguishing VERIFIED_CIRCUIT,
CANDIDATE_CIRCUIT, INCOMPLETE_PATHWAY, and FALSIFIED hypotheses based on
quantitative multi-control specificity, mediation rescue, and cross-prompt replication.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EpistemicCircuitTier(str, Enum):
    """The epistemic verification tier of a discovered mechanistic circuit."""
    VERIFIED_CIRCUIT = "VERIFIED_CIRCUIT"         # Specificity >= 3.0x, Mediation >= 70%, Cross-Prompt >= 80%
    CANDIDATE_CIRCUIT = "CANDIDATE_CIRCUIT"       # Specificity >= 2.0x, Mediation >= 40%, Cross-Prompt >= 60%
    INCOMPLETE_PATHWAY = "INCOMPLETE_PATHWAY"     # Isolated causal nodes with unmediated or broken pathway
    FALSIFIED = "FALSIFIED"                       # Failed negative controls (specificity < 2.0x) or null p-val > 0.05


@dataclass
class DiscoveredCircuitNode:
    """An individual functional component within a discovered circuit candidate."""
    node_id: str                        # e.g., "L8_N412", "L7_H3", "SAE_L8_F1842"
    layer: int
    component_type: str                 # "neuron" | "sae_latent" | "attention_head" | "mlp"
    component_index: int
    attribution_score: float
    causal_delta_z: float
    clean_logit: float
    intervened_logit: float
    control_specificity_ratio: float    # |delta_z| / mean(|delta_z_controls|)
    control_battery_passed_count: int   # out of 4 (matched-norm, same-layer, same-mechanism, random)
    evidence_status: str                # "VERIFIED" | "SUPPORTED" | "CANDIDATE"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiscoveredCircuitEdge:
    """A directed causal connection between two nodes in a discovered circuit candidate."""
    edge_id: str                        # e.g., "L6_N104->L8_N412"
    source_node_id: str
    target_node_id: str
    source_layer: int
    target_layer: int
    mechanism_type: str                 # "residual_stream" | "attention_ov" | "mlp_projection"
    edge_causal_effect: float
    edge_evidence_tier: str             # "VERIFIED" | "SUPPORTED" | "CANDIDATE"
    pairwise_rescue_fraction: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CircuitMediationMetrics:
    """End-to-end path patching and mediation rescue metrics for the entire circuit candidate."""
    direct_effect: float
    indirect_effect: float
    total_causal_effect: float
    mediation_rescue_fraction: float    # (z_rescued - z_ablated_A) / (z_clean - z_ablated_A)
    null_distribution_percentile: float # percentile vs matched random path ensemble
    null_distribution_p_value: float
    end_to_end_status: str              # "FULLY_MEDIATED" | "PARTIALLY_MEDIATED" | "UNMEDIATED"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CrossPromptGeneralizationSummary:
    """Generalization robustness of the discovered circuit across held-out evaluation prompts."""
    total_prompts_tested: int
    replicated_count: int
    replication_rate_pct: float
    mean_causal_delta_z: float
    cross_prompt_passed: bool
    prompt_breakdown: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiscoveredCircuitCandidate:
    """A fully quantified, triaged candidate mechanistic circuit produced by autonomous discovery."""
    circuit_id: str
    behavior_name: str
    model_id: str
    architecture: str
    timestamp_utc: str
    epistemic_classification: EpistemicCircuitTier
    nodes: List[DiscoveredCircuitNode]
    edges: List[DiscoveredCircuitEdge]
    mediation: CircuitMediationMetrics
    cross_prompt: CrossPromptGeneralizationSummary
    mean_control_specificity_ratio: float
    immutable_experiment_run_id: Optional[str]
    discovery_narrative: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "circuit_id": self.circuit_id,
            "behavior_name": self.behavior_name,
            "model_id": self.model_id,
            "architecture": self.architecture,
            "timestamp_utc": self.timestamp_utc,
            "epistemic_classification": self.epistemic_classification.value,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "mediation": self.mediation.to_dict(),
            "cross_prompt": self.cross_prompt.to_dict(),
            "mean_control_specificity_ratio": self.mean_control_specificity_ratio,
            "immutable_experiment_run_id": self.immutable_experiment_run_id,
            "discovery_narrative": self.discovery_narrative,
        }
