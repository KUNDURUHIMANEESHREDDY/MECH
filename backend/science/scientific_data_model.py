"""Canonical Scientific Data Models and Types for MECH Platform.

Defines the shared epistemically rigorous data models across all backend engines
and frontend panels (Logit Lens, SAE Dictionary, Multi-Mechanism Graph, Evidence Fusion).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EvidenceLevel(str, Enum):
    """Epistemic Evidence Hierarchy."""
    OBSERVED = "OBSERVED"  # Observational correlation / attention routing score
    CANDIDATE = "CANDIDATE"  # Feature attribution / linear projection support
    SUPPORTED = "SUPPORTED"  # Activation knockout / ablation recovery
    CAUSALLY_VERIFIED = "CAUSALLY_VERIFIED"  # Controlled double-counterfactual path patching confirmation
    FALSIFIED = "FALSIFIED"  # Failed empirical controls / mediation counter-tests
    UNEXECUTED = "UNEXECUTED"  # Missing weights / model runtime unavailable


class MechanismType(str, Enum):
    """Transformer mechanism taxonomy."""
    ATTENTION_ROUTING = "attention_routing"  # Attention-mediated feature routing (OV/QK)
    MLP_PROJECTION = "mlp_projection"  # Non-linear associative memory & key-value lookup
    RESIDUAL_STREAM = "residual_stream"  # Direct linear accumulation across layers
    OV_CIRCUIT = "ov_circuit"  # Output-Value information transport
    QK_CIRCUIT = "qk_circuit"  # Query-Key contextual pattern matching


class TransitionType(str, Enum):
    """Predictive belief state transitions across depth."""
    EMERGENCE = "emergence"  # Target probability becomes non-negligible
    DIVERGENCE = "divergence"  # Target sharply separates from distractors (highest ΔP jump)
    STABILIZATION = "stabilization"  # Top prediction locked in with low entropy
    SUPPRESSION = "suppression"  # Intermediate candidate suppressed by late inhibition
    STABLE = "stable"  # Incremental accumulation without phase change


class SubstrateAnchor(BaseModel):
    """Physical hardware substrate reference (dense neuron coordinate)."""
    layer: int
    neuron_idx: int
    weight_norm: float
    is_polysemantic: bool = True
    role_description: str = "Dense polysemantic coordinate"


class FeatureEvidence(BaseModel):
    """Empirical quality evaluation of a sparse autoencoder (SAE) latent or neuron."""
    feature_id: str
    layer: int
    latent_idx: int
    semantic_label: str
    specificity: float = Field(ge=0.0, le=1.0, description="Empirical specificity score")
    consistency: float = Field(ge=0.0, le=1.0, description="Activation consistency across target context")
    cross_prompt_stability: float = Field(ge=0.0, le=1.0, description="Activation stability across paraphrased probes")
    activation_strength: float
    causal_effect: Optional[float] = Field(default=None, description="Δlogit drop upon ablation")
    confidence_score: float = Field(ge=0.0, le=1.0)
    linear_logit_delta: Dict[str, float] = Field(
        default_factory=dict,
        description="Linear projection behavior: Δz_i ≈ a_i * (W_U d_i)",
    )
    substrate_anchors: List[SubstrateAnchor] = Field(default_factory=list)
    evidence_level: EvidenceLevel = EvidenceLevel.CANDIDATE
    is_causally_mediating: bool = False


class LogitLensTransition(BaseModel):
    """Temporal prediction state at a single Transformer layer."""
    layer: int
    top_token: str
    probability: float
    delta_probability: float = Field(description="Change in probability relative to layer l-1")
    rank: int = 1
    is_predictive_transition: bool = False
    transition_type: TransitionType = TransitionType.STABLE
    predictions: List[Dict[str, Any]] = Field(default_factory=list)
    active_feature_projections: List[Dict[str, Any]] = Field(default_factory=list)


class EdgeEvidenceObject(BaseModel):
    """Full structured continuous evidence object attached to every graph edge."""
    logit_lens_stage: str
    sae_association: Optional[str] = None
    attention_routing_score: float
    attribution_score: float
    causal_effect: Optional[float] = None
    robust_specificity: Optional[float] = None
    control_results: List[Dict[str, Any]] = Field(default_factory=list)
    cross_prompt_replicated: bool = False
    epistemic_scope: str = "Tested on factual probe suite with 4-control battery"


class CircuitPathwayEdge(BaseModel):
    """Multi-mechanism directed pathway connecting candidate features and components."""
    edge_id: str
    source_id: str
    target_id: str
    mechanism_type: MechanismType
    evidence_level: EvidenceLevel = EvidenceLevel.OBSERVED
    attention_routing_score: float = Field(ge=0.0, le=1.0, description="Effective attention-mediated routing score")
    attribution_score: float = Field(ge=0.0, le=1.0, description="Gradient/activation attribution score")
    causal_mediation_effect: Optional[float] = Field(default=None, description="Δlogit under path patching")
    value_flow_description: str = Field(description="Forward information transfer (Source -> Target)")
    query_relation_description: str = Field(description="Backward query mechanism (Target -> Source)")
    evidence_object: Optional[EdgeEvidenceObject] = None
    falsification_report: Optional[str] = None


class CircuitCompositionReport(BaseModel):
    """Evaluates full-circuit causal composition under the Weakest-Link Principle."""
    pathway_id: str
    edges: List[CircuitPathwayEdge] = Field(default_factory=list)
    composed_causal_tier: EvidenceLevel
    weakest_link_edge_id: str
    weakest_link_evidence_tier: EvidenceLevel
    composition_principle: str = "Weakest Necessary Link: A multi-step circuit is bounded by its least-verified causal link."
    end_to_end_path_patching_effect: Optional[float] = None


class UnifiedScientificReport(BaseModel):
    """Canonical interpretability report container shared across all MECH views."""
    investigation_id: str
    model_id: str
    clean_prompt: str
    corrupted_prompt: Optional[str] = None
    target_token: str
    predictive_divergence_layer: int
    logit_lens_trajectory: List[LogitLensTransition] = Field(default_factory=list)
    candidate_features: List[FeatureEvidence] = Field(default_factory=list)
    circuit_nodes: List[Dict[str, Any]] = Field(default_factory=list)
    circuit_edges: List[CircuitPathwayEdge] = Field(default_factory=list)
    circuit_composition: Optional[CircuitCompositionReport] = None
    falsified_hypotheses: List[Dict[str, Any]] = Field(default_factory=list)
    summary_verdict: str = ""
