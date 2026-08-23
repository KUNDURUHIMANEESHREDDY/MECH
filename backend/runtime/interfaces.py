"""Abstract Model Runtime Interfaces for In-Memory and Out-of-Core Execution.

Decouples high-level mechanistic interpretability methodologies (Logit Lens,
SAE feature probing, causal interventions, pathway verification) from the
underlying tensor residency, memory tiering, and hardware execution strategy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class PrecisionProfile:
    """Exact numerical precision and quantization settings."""
    weight_dtype: str = "float32"       # "float32" | "float16" | "bfloat16" | "int8" | "int4"
    activation_dtype: str = "float32"
    accumulation_dtype: str = "float32"
    quantization_scheme: Optional[str] = None  # "none" | "dynamic_int8" | "bitsandbytes_8bit" | "bitsandbytes_4bit"
    dequantization_method: Optional[str] = None
    tolerance_profile: str = "standard_fp32"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelArchitectureTopology:
    """Detailed architectural structure and layer naming topology."""
    architecture_family: str            # "gpt2" | "llama" | "mistral" | "qwen" | "gemma" | "pythia" | "generic"
    num_layers: int
    hidden_dim: int
    num_attention_heads: int
    vocab_size: int
    norm_type: str                      # "layernorm" | "rmsnorm"
    has_rotary_embeddings: bool = False
    has_absolute_pos_embeddings: bool = True
    mlp_activation_fn: str = "gelu"     # "gelu" | "silu" | "swiglu" | "relu"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



@dataclass(frozen=True)
class RuntimeMetadata:
    """Hardware and residency strategy provenance metadata."""
    runtime_type: str                   # "in_memory" | "out_of_core"
    model_id: str
    architecture: str
    num_layers: int
    hidden_dimension: int
    vocab_size: int
    precision: PrecisionProfile
    weight_storage: str                 # "resident_ram" | "safetensors_mmap"
    vram_budget_mb: float
    ram_budget_mb: float
    residency_policy: str               # "demand_paged" | "eager_all"
    device: str                         # "cpu" | "cuda:0"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["precision"] = self.precision.to_dict()
        return d


@dataclass
class RuntimeForwardOutput:
    """Output of a forward pass through the runtime."""
    prompt: str
    tokens: List[str]
    token_ids: List[int]
    top_predicted_token: str
    top_predicted_id: int
    target_token: Optional[str]
    target_logit: Optional[float]
    target_probability: Optional[float]
    target_rank: Optional[int]
    layer_residuals: Optional[Dict[int, Any]] = None
    runtime_metadata: Optional[RuntimeMetadata] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "prompt": self.prompt,
            "tokens": self.tokens,
            "token_ids": self.token_ids,
            "top_predicted_token": self.top_predicted_token,
            "top_predicted_id": self.top_predicted_id,
            "target_token": self.target_token,
            "target_logit": self.target_logit,
            "target_probability": self.target_probability,
            "target_rank": self.target_rank,
            "runtime_metadata": self.runtime_metadata.to_dict() if self.runtime_metadata else None,
        }
        return d


@dataclass
class RuntimeInterventionOutput:
    """Output of an intervention forward pass."""
    clean_logit: float
    intervened_logit: float
    delta_logit: float
    clean_probability: float
    intervened_probability: float
    delta_probability: float
    clean_rank: int
    intervened_rank: int
    delta_rank: int
    verdict: str
    runtime_metadata: Optional[RuntimeMetadata] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.runtime_metadata:
            d["runtime_metadata"] = self.runtime_metadata.to_dict()
        return d


@dataclass(frozen=True)
class PathHopSpec:
    """Specification of a path-patching hop."""
    source_layer: int
    target_layer: int
    source_component: str = "residual"  # "residual" | "mlp" | "attention"
    target_component: str = "residual"
    sequence_position: int = -1         # -1 for last token


@dataclass
class PathPatchingOutput:
    """Output of an out-of-core multi-hop path patching experiment."""
    source_prompt: str
    target_prompt: str
    target_token: str
    clean_target_logit: float
    corrupted_target_logit: float
    patched_target_logit: float
    indirect_effect: float
    mediation_rescue_fraction: float
    clean_target_rank: int
    patched_target_rank: int
    path_hops: List[PathHopSpec]
    artifacts: List[Dict[str, Any]]
    runtime_metadata: Optional[RuntimeMetadata] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.runtime_metadata:
            d["runtime_metadata"] = self.runtime_metadata.to_dict()
        return d


class ModelRuntimeInterface(ABC):
    """Abstract interface satisfied by both InMemory and OutOfCore runtimes."""

    @abstractmethod
    def get_runtime_metadata(self) -> RuntimeMetadata:
        """Returns hardware and residency execution strategy metadata."""
        pass

    @abstractmethod
    def forward(
        self,
        prompt: str,
        target_token: Optional[str] = None,
        capture_layer_residuals: bool = False,
    ) -> RuntimeForwardOutput:
        """Executes a standard forward pass."""
        pass

    @abstractmethod
    def compute_logit_lens_trajectory(
        self,
        prompt: str,
        target_token: str,
        distractor_token: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Computes layer-by-layer logit predictions and vocabulary rank trajectories."""
        pass

    @abstractmethod
    def project_to_vocabulary(
        self,
        hidden_state: Any,
        top_k: int = 10,
        target_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Projects a hidden state tensor onto vocabulary logits via virtual unembedding."""
        pass

    @abstractmethod
    def apply_intervention(
        self,
        prompt: str,
        target_token: str,
        layer: int,
        component_type: str,
        component_index: int,
        ablation_scale: float = 0.0,
    ) -> RuntimeInterventionOutput:
        """Executes a controlled intervention forward pass."""
        pass

    @abstractmethod
    def capture_activation(
        self,
        prompt: str,
        layer: int,
        component: str = "residual",
        sequence_position: int = -1,
        experiment_hash: Optional[str] = None,
    ) -> Any:
        """Captures an intermediate activation as a first-class ActivationArtifact."""
        pass

    @abstractmethod
    def patch_activation(
        self,
        target_prompt: str,
        target_token: str,
        layer: int,
        artifact: Any,
        component: str = "residual",
        sequence_position: int = -1,
    ) -> RuntimeForwardOutput:
        """Executes forward pass while injecting a captured ActivationArtifact at target layer."""
        pass

    @abstractmethod
    def patch_path(
        self,
        source_prompt: str,
        target_prompt: str,
        target_token: str,
        path_hops: List[PathHopSpec],
        experiment_hash: Optional[str] = None,
    ) -> PathPatchingOutput:
        """Executes multi-hop causal path patching and calculates mediation rescue fraction."""
        pass

