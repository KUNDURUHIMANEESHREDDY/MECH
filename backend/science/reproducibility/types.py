"""Immutable Data Contracts and Types for Reproducible Scientific Experiments.

Every experiment run is treated as a cryptographically signed, immutable snapshot
capturing exact model identity, execution environment, specification, full provenance
chain, quantitative measurements, and scientific verdict.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ModelIdentity:
    """Exact model checkpoint, parameterization, and execution strategy identity."""
    model_id: str
    architecture: str
    parameter_count: int
    weights_hash: str
    revision_or_commit: str
    tokenizer_hash: str
    config_hash: str
    source: str = "huggingface"
    precision: str = "fp32"
    execution_strategy: str = "in_memory"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionEnvironment:
    """Complete runtime environment specification ensuring reproducible execution."""
    python_version: str
    pytorch_version: str
    transformers_version: str
    cuda_version: Optional[str]
    gpu_name: Optional[str]
    gpu_compute_capability: Optional[str]
    os_platform: str
    os_release: str
    mech_version: str
    git_commit_sha: str
    dependency_lock_hash: str
    deterministic_mode: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InterventionImplementation:
    """Exact hook implementation identity to ensure physical execution equivalence."""
    type: str = "zero_ablation"
    target_module: str = "transformer.h.{layer}.mlp.c_fc"
    hook_location: str = "forward_hook_activation"
    hook_impl_version: str = "v1.0.0_torch_native"
    clamp_value: Optional[float] = 0.0
    dtype: str = "torch.float32"
    execution_mode: str = "deterministic"
    source_code_hash: str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentSpecification:
    """The exact scientific specification of the experiment."""
    clean_prompt: str
    target_token: str
    corrupted_prompt: Optional[str] = None
    distractor_token: Optional[str] = None
    target_component: str = "L8_N412"
    component_type: str = "neuron"  # "neuron" | "mlp" | "sae_latent" | "attention_head"
    layer: int = 8
    component_index: int = 412
    intervention_type: str = "zero_ablation"
    ablation_scale: float = 0.0
    random_seed: int = 42
    intervention_impl: InterventionImplementation = field(default_factory=InterventionImplementation)
    control_battery_spec: List[Dict[str, Any]] = field(default_factory=list)
    cross_prompt_suite: Optional[List[Tuple[str, str]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProvenanceChain:
    """The entire causal provenance chain leading to the scientific claim."""
    logit_lens_divergence_layer: int
    maximum_predictive_gain_layer: int
    active_sae_candidates: List[str]
    dense_substrate_anchors: List[str]
    linear_projection_delta: Dict[str, float]
    edge_causal_effect: float
    four_control_results: List[Dict[str, Any]]
    cross_prompt_stability: float
    mediation_rescue_fraction: float
    null_distribution_percentile: float
    null_distribution_p_value: float
    final_evidence_tier: str  # "CAUSALLY_VERIFIED" | "SUPPORTED" | "CANDIDATE" | "FALSIFIED"
    epistemic_scope: List[str] = field(default_factory=lambda: ["single-prompt"])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeStrategy:
    """Declared execution strategy and hardware budget configuration for an experiment."""
    execution_runtime: str = "out_of_core"  # "in_memory" | "out_of_core"
    max_active_layers: int = 1
    ram_budget_mb: float = 8192.0
    vram_budget_mb: float = 4096.0
    enforcement_mode: str = "strict"        # "strict" | "best_effort"
    violation_action: str = "evict"         # "evict" | "spill" | "abort"
    prefetch_enabled: bool = True
    prefetch_queue_depth: int = 1
    quantization: str = "fp32"
    virtual_unembedding_chunk_size: int = 4096

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionTelemetry:
    """Physical hardware and execution telemetry captured during experiment run."""
    peak_ram_mb: float = 0.0
    peak_vram_mb: float = 0.0
    total_load_time_ms: float = 0.0
    total_eviction_time_ms: float = 0.0
    total_compute_time_ms: float = 0.0
    prefetch_requests: int = 0
    prefetch_hits: int = 0
    prefetch_misses: int = 0
    prefetch_hit_rate_pct: float = 0.0
    compute_utilization_pct: float = 0.0
    io_stall_pct: float = 0.0
    memory_budget_satisfied: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImmutableExperimentRun:
    """An immutable, cryptographically hashed experimental record with execution strategy and hardware telemetry."""
    run_id: str
    parent_run_id: Optional[str]  # Lineage: Run #104 -> Run #105
    experiment_type: str          # "ORIGINAL" | "REPRODUCTION" | "REPLICATION"
    title: str
    timestamp_utc: str
    model: ModelIdentity
    environment: ExecutionEnvironment
    specification: ExperimentSpecification
    provenance_chain: ProvenanceChain
    measurements: Dict[str, Any]
    verdict: str
    runtime_strategy: RuntimeStrategy = field(default_factory=RuntimeStrategy)
    execution_telemetry: Optional[ExecutionTelemetry] = None
    manifest_sha256: str = ""
    canonical_spec_hash: str = ""

    def calculate_canonical_spec_hash(self) -> str:
        """Calculates H_experiment = SHA256(canonical_spec || model_identity || environment_identity)."""
        canonical_payload = {
            "specification": self.specification.to_dict(),
            "model": self.model.to_dict(),
            "environment": {
                "python_version": self.environment.python_version,
                "pytorch_version": self.environment.pytorch_version,
                "transformers_version": self.environment.transformers_version,
                "dependency_lock_hash": self.environment.dependency_lock_hash,
                "deterministic_mode": self.environment.deterministic_mode,
            },
            "runtime_strategy": self.runtime_strategy.to_dict(),
        }
        raw = json.dumps(canonical_payload, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def calculate_manifest_hash(self) -> str:
        """Calculates a deterministic SHA-256 hash over the full payload."""
        payload = {
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "experiment_type": self.experiment_type,
            "title": self.title,
            "timestamp_utc": self.timestamp_utc,
            "model": self.model.to_dict(),
            "environment": self.environment.to_dict(),
            "specification": self.specification.to_dict(),
            "provenance_chain": self.provenance_chain.to_dict(),
            "measurements": self.measurements,
            "verdict": self.verdict,
            "runtime_strategy": self.runtime_strategy.to_dict(),
            "execution_telemetry": self.execution_telemetry.to_dict() if self.execution_telemetry else None,
            "canonical_spec_hash": self.canonical_spec_hash or self.calculate_canonical_spec_hash(),
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["canonical_spec_hash"] = self.canonical_spec_hash or self.calculate_canonical_spec_hash()
        d["manifest_sha256"] = self.manifest_sha256 or self.calculate_manifest_hash()
        return d




@dataclass
class ComponentToleranceResult:
    """Tolerance comparison for an individual measured component."""
    metric_name: str
    expected_value: Any
    observed_value: Any
    abs_diff: Optional[float]
    rel_diff: Optional[float]
    tolerance_threshold: str
    passed: bool
    status: str  # "EXACT" | "WITHIN_TOLERANCE" | "MISMATCH"
    details: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReproductionComparisonReport:
    """Detailed reproduction report evaluating numerical fidelity across all components."""
    original_run_id: str
    reproduction_run_id: str
    timestamp_utc: str
    overall_reproduced: bool
    numerical_tolerance_verdict: str
    component_comparisons: List[ComponentToleranceResult]
    execution_duration_ms: float
    model_matched: bool
    environment_matched: bool
    reproduction_category: str = "STRICT_IDENTICAL_RUNTIME"  # "STRICT_IDENTICAL_RUNTIME" | "CROSS_STRATEGY_REPLICATION"
    strategy_matched: bool = True
    strategy_divergence_summary: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_run_id": self.original_run_id,
            "reproduction_run_id": self.reproduction_run_id,
            "timestamp_utc": self.timestamp_utc,
            "overall_reproduced": self.overall_reproduced,
            "numerical_tolerance_verdict": self.numerical_tolerance_verdict,
            "component_comparisons": [c.to_dict() for c in self.component_comparisons],
            "execution_duration_ms": self.execution_duration_ms,
            "model_matched": self.model_matched,
            "environment_matched": self.environment_matched,
            "reproduction_category": self.reproduction_category,
            "strategy_matched": self.strategy_matched,
            "strategy_divergence_summary": self.strategy_divergence_summary,
        }

