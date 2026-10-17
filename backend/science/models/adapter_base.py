"""Abstract Model Adapter Base Class.

Defines the unified interface all model families must implement.
Real HuggingFace model loading is used when weights are available;
mock_mode=True is used for unit tests without GPU/weights.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ActivationResult:
    layer: int
    token_index: int
    neuron_index: int
    activation_value: float
    context_prompt: str
    top_k_tokens: List[Dict[str, Any]] = field(default_factory=list)
    #: Shape of the model that produced this result, read from the config that
    #: was actually loaded rather than from the spec table. Optional because
    #: only ``HFAdapterMixin`` records it, and ``None`` means the construction
    #: site did not. A hardcoded head count is how a Llama-3 gets measured as a
    #: 2B, so where the loaded value is available it is the one recorded.
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None
    d_model: Optional[int] = None
    #: How this result was produced: ``"live"``, ``"seeded"``, or
    #: ``"unavailable"``.
    #:
    #: ``None`` means the construction site did not record it -- the same
    #: contract as ``PatchResult.model_fingerprint`` below, so an unrecorded
    #: value is visibly absent rather than silently plausible. This exists
    #: because ``GPT2Adapter``'s mock branch returns fixed tokens and
    #: activations, and a caller holding only the result had no way to tell
    #: those from measurements.
    provenance: Optional[str] = None


@dataclass
class AttentionPattern:
    layer: int
    head: int
    pattern_matrix: List[List[float]]          # [seq_len, seq_len]
    tokens: List[str]
    attn_entropy: float


class LiveUnavailable(RuntimeError):
    """No live model is available, so the measurement cannot be made.

    Callers must fail closed on this rather than substituting a value. It lives
    here, in the shared adapter layer, because both the model adapters and the
    discovery layer need it -- defining it in either one inverted the import
    direction. ``live_measure`` re-exports this name, so existing
    ``except LiveUnavailable`` handlers are unaffected.
    """


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
    #: Fingerprint of the weights this result was measured against.
    #:
    #: Optional only because three adapter classes construct `PatchResult` at
    #: nine sites and the field cannot be required until all of them populate
    #: it -- a required field with unpopulated constructors raises at runtime
    #: rather than being absent, which trades a silent gap for a crash.
    #:
    #: `None` means "no identity was recorded", and `tests/pytest/
    #: test_model_identity_seal.py` holds this red until that is impossible.
    #: Identity must be compared by fingerprint and never by shape: two
    #: checkpoints of one family agree on `layer` and `neuron_index` by
    #: construction, and every tensor-level check passes across a swap.
    model_fingerprint: Optional[str] = None
    #: How this result was produced: ``"live"``, ``"seeded"``, or
    #: ``"unavailable"``. ``None`` means the construction site did not record
    #: it; see ``ActivationResult.provenance``.
    provenance: Optional[str] = None

    def same_model_as(self, other: "PatchResult") -> Optional[bool]:
        """Whether two results may be compared. `None` if either is unsealed.

        Deliberately not a bool. `False` would mean "definitely different
        models" and `True` would mean "safe to compare"; collapsing the
        unrecorded case into either is how a gap becomes a result.
        """
        if self.model_fingerprint is None or other.model_fingerprint is None:
            return None
        return self.model_fingerprint == other.model_fingerprint


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


class ModelAdapter(ABC):
    """Unified interface for mechanistic interpretability across model families."""

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._model = None
        self._tokenizer = None
        self._manager = None
        #: True when this adapter fabricates its output instead of running a
        #: model. Subclasses that never execute weights must set this, because
        #: downstream provenance checks treat it as "not a measurement".
        self.simulated = False
        self.simulation_reason: str = ""
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
        except Exception as exc:
            # Graceful degradation: fall back to mock mode if weights unavailable.
            #
            # `simulated` is set alongside `mock_mode`, not replaced by it. The
            # comment on `simulated` above says downstream provenance checks
            # treat it as "not a measurement", and this path does fabricate its
            # output -- flipping only `mock_mode` left an adapter that could not
            # load its weights reporting `simulated is False`.
            self.spec = ModelSpec(**{**self.spec.__dict__, "mock_mode": True})
            self.simulated = True
            self.simulation_reason = (
                f"Weights for {self.spec.hf_repo_id} could not be loaded "
                f"({type(exc).__name__}: {exc}). This adapter returns its mock "
                f"fixtures, not model output."
            )

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
