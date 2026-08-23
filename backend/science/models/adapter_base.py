"""Abstract Model Adapter Base Class.

Defines the unified interface all model families must implement.
Real HuggingFace model loading is used when weights are available;
mock_mode=True is used for unit tests without GPU/weights.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging
logger = logging.getLogger(__name__)



@dataclass
class ActivationResult:
    layer: int
    token_index: int
    neuron_index: int
    activation_value: float
    context_prompt: str
    top_k_tokens: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AttentionPattern:
    layer: int
    head: int
    pattern_matrix: List[List[float]]          # [seq_len, seq_len]
    tokens: List[str]
    attn_entropy: float


@dataclass
class PatchResult:
    original_logit: float
    patched_logit: float
    delta: float
    top_token_before: str
    top_token_after: str
    layer: int
    neuron_index: int
    patch_value: float


@dataclass
class ModelSpec:
    """Static model metadata."""
    model_id: str
    family: str                    # "gpt2", "gemma", "llama", "qwen", "mistral", "deepseek"
    num_layers: int
    num_heads: int
    d_model: int
    d_mlp: int
    vocab_size: int
    context_length: int
    hf_repo_id: str               # HuggingFace repo, e.g. "gpt2"
    supports_sae: bool = True
    mock_mode: bool = False       # True during unit tests (no real weights loaded)

    @property
    def name(self) -> str:
        return self.model_id



class ModelAdapter(ABC):
    """Unified interface for mechanistic interpretability across model families."""

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._model = None
        self._tokenizer = None
        self._manager = None
        if not spec.mock_mode:
            self._load_model()

    def _load_model(self) -> None:
        """Load real HuggingFace model weights via ModelManager. Skipped in mock_mode."""
        try:
            from .model_manager import ModelManager
            self._manager = ModelManager()
            self._model, self._tokenizer = self._manager.get_model_and_tokenizer(
                self.spec.hf_repo_id
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Swallowed exception: %s", exc)
            # When in live mode, leave _model as None to fail closed on downstream queries
            self._model = None
            self._tokenizer = None


    @abstractmethod
    def get_activations(
        self,
        prompt: str,
        layer: int,
        neuron_index: Optional[int] = None,
    ) -> List[ActivationResult]:
        """Return residual stream / MLP activations for the given layer."""
        ...

    @abstractmethod
    def get_attention_patterns(
        self,
        prompt: str,
        layer: int,
    ) -> List[AttentionPattern]:
        """Return all attention head patterns for the given layer."""
        ...

    @abstractmethod
    def get_logits(self, prompt: str) -> Dict[str, Any]:
        """Return final token logits and top-k predictions."""
        ...

    @abstractmethod
    def patch_activation(
        self,
        prompt: str,
        layer: int,
        neuron_index: int,
        patch_value: float,
    ) -> PatchResult:
        """Apply activation patching and return before/after comparison."""
        ...

    @abstractmethod
    def get_residual_stream(self, prompt: str) -> List[Dict[str, Any]]:
        """Return the residual stream state at each layer."""
        ...

    def get_model_spec(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "model_id": self.spec.model_id,
            "family": self.spec.family,
            "num_layers": self.spec.num_layers,
            "num_heads": self.spec.num_heads,
            "d_model": self.spec.d_model,
            "d_mlp": self.spec.d_mlp,
            "vocab_size": self.spec.vocab_size,
            "context_length": self.spec.context_length,
            "hf_repo_id": self.spec.hf_repo_id,
            "mock_mode": self.spec.mock_mode,
        }
