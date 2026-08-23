"""Canonical Scientific Research Entities and Schemas for MECH Platform.

Epistemically grounded data models for investigations, hypotheses, experiments,
causal interventions, evidence graphs, mechanisms, provenance chains, compute jobs,
and the strict ScientificResult / ScientificResponseEnvelope type system.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class KnowledgeType(str, Enum):
    OBSERVATION = "OBSERVATION"        # Directly measured tensor activations / attention
    INFERENCE = "INFERENCE"            # Inferred pattern / candidate mechanism
    CAUSAL_EVIDENCE = "CAUSAL_EVIDENCE"  # Statistically significant effect from intervention
    CLAIM = "CLAIM"                    # Researcher-level scientific conclusion


class HypothesisStatus(str, Enum):
    UNTESTED = "UNTESTED"
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    FALSIFIED = "FALSIFIED"
    INCONCLUSIVE = "INCONCLUSIVE"


class EvidenceProvenanceSource(str, Enum):
    COMPUTED = "COMPUTED"
    OBSERVED = "OBSERVED"
    USER_DEFINED = "USER_DEFINED"
    AI_GENERATED = "AI_GENERATED"
    IMPORTED = "IMPORTED"
    DERIVED = "DERIVED"
    SYNTHETIC = "SYNTHETIC"


class InterventionType(str, Enum):
    ACTIVATION_PATCHING = "ACTIVATION_PATCHING"
    PATCHING = "PATCHING"
    ABLATION_ZERO = "ABLATION_ZERO"
    ABLATION_MEAN = "ABLATION_MEAN"
    ABLATION_GAUSSIAN = "ABLATION_GAUSSIAN"
    ABLATION_NOISE = "ABLATION_NOISE"
    STEERING = "STEERING"
    SCALING = "SCALING"
    CLAMPING = "CLAMPING"
    COUNTERFACTUAL = "COUNTERFACTUAL"
    RESTORATION = "RESTORATION"


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    PREPARING = "PREPARING"
    LOADING_MODEL = "LOADING_MODEL"
    RUNNING_INFERENCE = "RUNNING_INFERENCE"
    CAPTURING_ACTIVATIONS = "CAPTURING_ACTIVATIONS"
    COMPUTING_INTERVENTION = "COMPUTING_INTERVENTION"
    WRITING_ARTIFACTS = "WRITING_ARTIFACTS"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    FAILED_RESOURCE_LIMIT = "FAILED_RESOURCE_LIMIT"
    CANCELLED = "CANCELLED"


class ExperimentExecutionStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    NOT_EXECUTABLE = "NOT_EXECUTABLE"  # Cannot execute (e.g., no model available)


# ---------------------------------------------------------------------------
# Strict Result Type System with Provenance
# ---------------------------------------------------------------------------
class ScientificResult(BaseModel):
    id: str = Field(default_factory=lambda: f"res_{uuid.uuid4().hex[:10]}")
    metric_name: str
    value: float
    unit: str = "delta_logit"
    baseline_value: Optional[float] = None
    control_value: Optional[float] = None
    effect_size_cohens_d: Optional[float] = None
    uncertainty_ci_95: Optional[tuple[float, float]] = None
    sample_size: int = 1
    methodology: str = ""
    model_id: str = "gpt2"
    model_hash: str = ""
    dataset_id: str = "ioi"
    dataset_version: str = "1.0.0"
    experiment_id: str = ""
    run_id: str = ""
    manifest_id: str = ""
    manifest_sha256: str = ""
    code_version: str = "2.0.0"
    execution_environment: str = "PyTorch 2.4.0"
    knowledge_type: KnowledgeType = KnowledgeType.CAUSAL_EVIDENCE
    is_reproducible: bool = True
    timestamp: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Investigation
# ---------------------------------------------------------------------------
class Investigation(BaseModel):
    id: str = Field(default_factory=lambda: f"inv_{uuid.uuid4().hex[:10]}")
    title: str
    research_question: str
    model_id: str = "gpt2"
    dataset_id: str = "ioi"
    current_hypothesis_id: Optional[str] = None
    status: str = "active"  # active, completed, archived
    tags: List[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Hypothesis with Multidimensional Evidence & Falsification
# ---------------------------------------------------------------------------
class AlternativeExplanation(BaseModel):
    id: str = Field(default_factory=lambda: f"alt_{uuid.uuid4().hex[:8]}")
    component: str  # e.g. "L8H4", "MLP_L8", "Residual_Stream"
    description: str
    status: str = "UNRESOLVED"  # UNRESOLVED, RULED_OUT, PLAUSIBLE


class Hypothesis(BaseModel):
    id: str = Field(default_factory=lambda: f"hyp_{uuid.uuid4().hex[:10]}")
    investigation_id: str
    title: str
    statement: str  # claim statement, e.g. "L9H9 causally mediates IOI"
    target_component: str = ""  # e.g. "L9H9", "L8_MLP", "Feature_412"
    prediction: str = ""  # What should happen if true (e.g. "Δlogit > 1.0 upon ablation")
    expected_evidence: str = ""  # What measurements would support it
    falsification_condition: str = ""  # What measurement disproves it (e.g. "Δlogit < 0.2")
    falsification_threshold: float = 0.2
    knowledge_type: KnowledgeType = KnowledgeType.INFERENCE
    status: HypothesisStatus = HypothesisStatus.UNTESTED

    # Multidimensional Evidence Scorecard
    observational_support: str = "NONE"   # NONE, WEAK, MODERATE, STRONG
    causal_support: str = "NONE"          # NONE, WEAK, MODERATE, STRONG
    control_contrast: str = "NONE"        # NONE, WEAK, MODERATE, STRONG
    replication_count: int = 0
    falsification_tested: bool = False
    alternative_explanations: List[AlternativeExplanation] = Field(default_factory=list)

    evidence_count_supporting: int = 0
    evidence_count_contradicting: int = 0
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    contradicting_evidence_ids: List[str] = Field(default_factory=list)
    methodology: str = ""
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class ExperimentExecutionStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# ---------------------------------------------------------------------------
# Experiment Specification
# ---------------------------------------------------------------------------
class Experiment(BaseModel):
    id: str = Field(default_factory=lambda: f"exp_{uuid.uuid4().hex[:10]}")
    investigation_id: str
    hypothesis_id: Optional[str] = None
    name: str
    description: str = ""
    model_id: str = "gpt2"
    dataset_id: str = "ioi"
    clean_prompt: str
    corrupted_prompt: Optional[str] = None
    target_token: str
    distractor_token: Optional[str] = None
    intervention_type: InterventionType = InterventionType.ACTIVATION_PATCHING
    source_component: str  # e.g. "L9H9"
    control_component: Optional[str] = None  # e.g. "L0H0" or "random"
    steering_coefficient: float = 1.0
    repeats: int = 1
    metric_target: str = "logit_difference"
    execution_status: ExperimentExecutionStatus = ExperimentExecutionStatus.PENDING
    created_at: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Experiment Run Record
# ---------------------------------------------------------------------------
class ExperimentRun(BaseModel):
    id: str = Field(default_factory=lambda: f"run_{uuid.uuid4().hex[:10]}")
    experiment_id: str
    investigation_id: str
    timestamp: float = Field(default_factory=time.time)
    model_id: str = "gpt2"
    model_hash: str = ""
    dataset_version: str = "1.0.0"
    code_version: str = "2.0.0"
    seed: int = 42
    hardware: str = "cpu"
    execution_time_ms: float = 0.0
    status: JobStatus = JobStatus.COMPLETED
    baseline_target_prob: float = 0.0
    intervened_target_prob: float = 0.0
    delta_target_prob: float = 0.0
    baseline_logit: float = 0.0
    intervened_logit: float = 0.0
    delta_logit: float = 0.0
    control_delta_logit: Optional[float] = None
    effect_size_cohens_d: Optional[float] = None
    p_value: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    top_predicted_tokens_clean: List[Dict[str, Any]] = Field(default_factory=list)
    top_predicted_tokens_intervened: List[Dict[str, Any]] = Field(default_factory=list)
    is_reproducible: bool = True
    manifest_id: Optional[str] = None
    provenance_hash: Optional[str] = None
    artifact_ids: List[str] = Field(default_factory=list)
    logs: List[str] = Field(default_factory=list)
    used_mock_data: bool = False  # Critical: True if results are from mock/synthetic data, not live model


# ---------------------------------------------------------------------------
# Evidence Node & Link in Knowledge Graph
# ---------------------------------------------------------------------------
class EvidenceRecord(BaseModel):
    id: str = Field(default_factory=lambda: f"evi_{uuid.uuid4().hex[:10]}")
    investigation_id: str
    hypothesis_id: Optional[str] = None
    experiment_run_id: Optional[str] = None
    source_type: EvidenceProvenanceSource = EvidenceProvenanceSource.COMPUTED
    claim: str
    evidence_level: str = "OBSERVED"  # OBSERVED, CANDIDATE, SUPPORTED, CAUSALLY_VERIFIED, FALSIFIED
    knowledge_type: KnowledgeType = KnowledgeType.OBSERVATION
    supports_hypothesis: bool = True
    metric_name: str
    metric_value: float
    baseline_value: Optional[float] = None
    control_value: Optional[float] = None
    sample_size: int = 1
    statistical_details: Dict[str, Any] = Field(default_factory=dict)
    provenance_chain: List[str] = Field(default_factory=list)
    methodology: str = ""
    created_at: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Mechanism Definition
# ---------------------------------------------------------------------------
class MechanismNode(BaseModel):
    id: str
    component_type: str  # input, attention_head, mlp, residual, output
    label: str
    layer: Optional[int] = None
    head: Optional[int] = None
    neuron: Optional[int] = None
    feature_id: Optional[str] = None
    evidence_level: str = "OBSERVED"


class MechanismEdge(BaseModel):
    id: str
    source_node_id: str
    target_node_id: str
    mechanism_type: str  # attention_routing, mlp_projection, residual_stream, qk_circuit, ov_circuit
    causal_effect: Optional[float] = None
    is_causally_verified: bool = False
    evidence_ids: List[str] = Field(default_factory=list)


class Mechanism(BaseModel):
    id: str = Field(default_factory=lambda: f"mech_{uuid.uuid4().hex[:10]}")
    investigation_id: str
    name: str
    description: str = ""
    nodes: List[MechanismNode] = Field(default_factory=list)
    edges: List[MechanismEdge] = Field(default_factory=list)
    is_evidence_backed: bool = False
    weakest_link_tier: str = "OBSERVED"
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Research Artifact Metadata (Tensor/Plot/Manifest)
# ---------------------------------------------------------------------------
class ResearchArtifact(BaseModel):
    id: str = Field(default_factory=lambda: f"art_{uuid.uuid4().hex[:10]}")
    investigation_id: str
    experiment_run_id: Optional[str] = None
    name: str
    artifact_type: str  # tensor, json_manifest, report, chart_data, csv
    file_path: str
    checksum_sha256: str
    size_bytes: int = 0
    storage_format: str = "safetensors"  # safetensors, pt, json, md, raw
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Compute Job
# ---------------------------------------------------------------------------
class ComputeJob(BaseModel):
    id: str = Field(default_factory=lambda: f"job_{uuid.uuid4().hex[:10]}")
    job_type: str
    name: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    current_stage: str = "QUEUED"
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    result_artifact_id: Optional[str] = None
    result_summary: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    logs: List[str] = Field(default_factory=list)
