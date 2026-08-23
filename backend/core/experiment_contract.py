"""Shared experiment contract between Agent 2 (Research Interface) and Agent 1 (Experimental Engine).

Defines the formal schemas that both agents must adhere to. No hardcoded, fabricated,
synthetic, or placeholder scientific results may be presented as real MECH findings.
If an experiment has not actually executed, the UI must explicitly show that it has not executed.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Execution Status
# ---------------------------------------------------------------------------

class ExperimentExecutionStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    NOT_EXECUTABLE = "NOT_EXECUTABLE"  # Cannot execute (e.g., no model available)


# ---------------------------------------------------------------------------
# Intervention
# ---------------------------------------------------------------------------

class InterventionType(str, Enum):
    ACTIVATION_PATCHING = "ACTIVATION_PATCHING"
    ABLATION_ZERO = "ABLATION_ZERO"
    ABLATION_MEAN = "ABLATION_MEAN"
    ABLATION_GAUSSIAN = "ABLATION_GAUSSIAN"
    STEERING = "STEERING"
    COUNTERFACTUAL = "COUNTERFACTUAL"
    RESTORATION = "RESTORATION"


class Intervention(BaseModel):
    id: str = Field(default_factory=lambda: f"inv_{uuid.uuid4().hex[:8]}")
    experiment_run_id: str
    intervention_type: InterventionType
    target_component: str  # e.g. "L9H9", "L8_MLP", "Feature_412"
    control_component: Optional[str] = None  # e.g. "L0H0" or "random"
    steering_coefficient: float = 1.0
    parameter: Optional[Dict[str, Any]] = None  # intervention-specific parameters
    applied_at_timestamp: float = Field(default_factory=time.time)
    description: str = ""


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

class Metric(BaseModel):
    name: str  # e.g. "logit_difference", "accuracy", "perplexity"
    value: float
    unit: str = "delta_logit"
    baseline_value: Optional[float] = None
    intervention_value: Optional[float] = None
    effect_size: Optional[float] = None  # Cohen's d or similar
    p_value: Optional[float] = None
    confidence_interval: Optional[tuple[float, float]] = None
    sample_size: int = 1
    measurement_method: str = ""
    conditions: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    id: str = Field(default_factory=lambda: f"obs_{uuid.uuid4().hex[:8]}")
    experiment_run_id: str
    metric: Metric
    intervention: Optional[Intervention] = None
    baseline_measurement: Optional[Dict[str, Any]] = None
    intervened_measurement: Optional[Dict[str, Any]] = None
    raw_data_hash: str = ""  # SHA256 of the raw activation/tensor data
    recorded_at: float = Field(default_factory=time.time)
    provenance_chain: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

class ProvenanceStep(BaseModel):
    step_id: str = Field(default_factory=lambda: f"prov_{uuid.uuid4().hex[:6]}")
    description: str
    executed_by: str  # e.g. "experiment_runner", "analysis_pipeline", "human_researcher"
    executed_at: float = Field(default_factory=time.time)
    input_state: Dict[str, Any] = Field(default_factory=dict)
    output_state: Dict[str, Any] = Field(default_factory=dict)
    transformation: str = ""  # e.g. "ablation", "steering", "activation_patching"
    verification_status: str = "PENDING"  # PENDING, VERIFIED, FALSIFIED, INCONCLUSIVE


class ProvenanceChain(BaseModel):
    id: str = Field(default_factory=lambda: f"provchain_{uuid.uuid4().hex[:8]}")
    experiment_run_id: str
    steps: List[ProvenanceStep] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    verified_at: Optional[float] = None


# ---------------------------------------------------------------------------
# Experiment (specification - what will be run)
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
    updated_at: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Experiment Run (execution record - what was actually run)
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
    execution_status: ExperimentExecutionStatus = ExperimentExecutionStatus.PENDING
    execution_time_ms: float = 0.0
    # Measurement outcomes
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
    # Critical: marker for whether real execution occurred
    used_mock_data: bool = False  # True if results are from mock/synthetic data


# ---------------------------------------------------------------------------
# Hypothesis (with multidimensional evidence & falsification)
# ---------------------------------------------------------------------------

class HypothesisStatus(str, Enum):
    UNTESTED = "UNTESTED"
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    FALSIFIED = "FALSIFIED"
    INCONCLUSIVE = "INCONCLUSIVE"


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


# Evidence level (shared between Python and TypeScript)
class KnowledgeType(str, Enum):
    OBSERVATION = "OBSERVATION"
    INFERENCE = "INFERENCE"
    CAUSAL_EVIDENCE = "CAUSAL_EVIDENCE"
    CLAIM = "CLAIM"


class EvidenceLevel(str, Enum):
    OBSERVED = "OBSERVED"
    CANDIDATE = "CANDIDATE"
    SUPPORTED = "SUPPORTED"
    CAUSALLY_VERIFIED = "CAUSALLY_VERIFIED"
    FALSIFIED = "FALSIFIED"


# ---------------------------------------------------------------------------
# Evidence Record
# ---------------------------------------------------------------------------

class EvidenceRecord(BaseModel):
    id: str = Field(default_factory=lambda: f"evi_{uuid.uuid4().hex[:10]}")
    investigation_id: str
    hypothesis_id: Optional[str] = None
    experiment_run_id: Optional[str] = None
    source_type: str = "COMPUTED"  # COMPUTED, OBSERVED, USER_DEFINED, AI_GENERATED, IMPORTED, DERIVED
    claim: str
    evidence_level: EvidenceLevel = EvidenceLevel.OBSERVED
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
    status: ExperimentExecutionStatus = ExperimentExecutionStatus.QUEUED
    progress: float = 0.0
    current_stage: str = "QUEUED"
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    result_artifact_id: Optional[str] = None
    result_summary: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    logs: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Experiment Result (runtime container, parallel to ExperimentResult in experiment_runner.py)
# ---------------------------------------------------------------------------

class ExperimentResult(BaseModel):
    """Container for a single experiment's results with full provenance."""

    name: str
    experiment_type: str  # e.g. "attention", "neuron", "residual", "cross_layer"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    parameters: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    experiment_run_id: str = ""
    used_mock_data: bool = False  # Critical: tracks if results are synthetic

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "experiment_type": self.experiment_type,
            "timestamp": self.timestamp,
            "parameters": self.parameters,
            "results": self.results,
            "experiment_run_id": self.experiment_run_id,
            "used_mock_data": self.used_mock_data,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)