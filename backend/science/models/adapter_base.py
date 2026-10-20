"""Abstract Model Adapter Base Class.

Defines the unified interface all model families must implement.
Real HuggingFace model loading is used when weights are available;
mock_mode=True is used for unit tests without GPU/weights.
"""

from __future__ import annotations

import hashlib
import json
import re
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


#: The shape of a fingerprint this module produces: ``sha256:<32 hex>``.
#:
#: A string that is not of this shape was not produced by the fingerprinting
#: code -- it is a placeholder, a truncation, or a hand-written value. It
#: carries no evidence of anything, so a comparison against one cannot be
#: answered from it.
_FINGERPRINT_PATTERN = re.compile(r"^sha256:[0-9a-f]{32}$")


def is_canonical_fingerprint(value: Optional[str]) -> bool:
    """Whether `value` is a digest produced by this module."""
    return isinstance(value, str) and _FINGERPRINT_PATTERN.match(value) is not None


def hash_weights(source: Any) -> Optional[str]:
    """A content hash of a model's parameters, or `None` if it has none.

    The digest covers each parameter's name, shape and values. The name
    matters because two checkpoints can hold the same values under
    different names; the shape matters because a resharded checkpoint is a
    different model even when the values line up.
    """
    if source is None:
        return None
    try:
        named = sorted(source.named_parameters(), key=lambda item: item[0])
    except Exception:
        return None
    if not named:
        return None

    import torch  # local: this module imports torch lazily by design

    digest = hashlib.sha256()
    for name, param in named:
        digest.update(name.encode("utf-8"))
        try:
            tensor = param.detach().to("cpu")
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.to(torch.float32).contiguous().numpy().tobytes())
        except Exception:
            return None
    return f"sha256:{digest.hexdigest()[:32]}"


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
    #: Every construction site in `backend/science/models/` passes it, and
    #: `tests/pytest/test_model_identity_seal.py` counts the ones that do
    #: not, so a new site cannot quietly drop the identity again. The field
    #: stays optional because a result can also be built by hand -- and one
    #: built by hand is exactly the case that has to be *refused* rather
    #: than compared, which is what `same_model_as` is for.
    #:
    #: `None` means "no identity was recorded". It is not a placeholder for
    #: an unknown-but-real checkpoint.
    #:
    #: Identity must be compared by fingerprint and never by shape: two
    #: checkpoints of one family agree on `layer` and `neuron_index` by
    #: construction, and every tensor-level check passes across a swap.
    model_fingerprint: Optional[str] = None
    #: How this result was produced: ``"live"``, ``"seeded"``, or
    #: ``"unavailable"``. ``None`` means the construction site did not record
    #: it; see ``ActivationResult.provenance``.
    provenance: Optional[str] = None

    def same_model_as(self, other: "PatchResult") -> Optional[bool]:
        """Whether two results may be compared. `None` if the answer is unknown.

        Deliberately not a bool. `False` means "definitely different models"
        and `True` means "safe to compare"; collapsing the unrecorded case
        into either is how a gap becomes a result.

        Three cases:

        * both sealed -- `True` when the fingerprints agree, `False` when
          they do not. Identity is compared by fingerprint and never by
          shape: two checkpoints of one family agree on `layer` and
          `neuron_index` by construction, so the fingerprint is the only
          thing that can tell them apart.
        * neither sealed -- `None`. Nothing was recorded, so nothing can be
          asserted.
        * one sealed, one not -- `False` when the sealed side carries a real
          digest, because a measured identity is definitely not an
          unrecorded one; and `None` when it carries a string that is not a
          digest, because a placeholder is not an identity and cannot
          answer the question either.
        """
        mine, theirs = self.model_fingerprint, other.model_fingerprint
        if mine is not None and theirs is not None:
            return mine == theirs
        if mine is None and theirs is None:
            return None
        recorded = mine if mine is not None else theirs
        return False if is_canonical_fingerprint(recorded) else None


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
        #: Cached identities of what is loaded: the weights, the tokenizer
        #: and the configuration. Each is a content hash, not a label.
        self._fingerprint: Optional[str] = None
        self._tokenizer_fingerprint: Optional[str] = None
        self._config_fingerprint: Optional[str] = None
        #: True when this adapter fabricates its output instead of running a
        #: model. Subclasses that never execute weights must set this, because
        #: downstream provenance checks treat it as "not a measurement".
        self.simulated = False
        self.simulation_reason: str = ""
        if not spec.mock_mode:
            self._load_model()
        self._publish_model_identity()

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

    # ------------------------------------------------------------------ #
    #  Model identity                                                      #
    # ------------------------------------------------------------------ #

    def _weight_source(self) -> Any:
        """The object whose parameters are this model's weights, or None."""
        model = getattr(self, "_model", None)
        if model is not None:
            return model
        # TransformerLens keeps the HuggingFace model on `params`.
        hooked = getattr(self, "_tl_model", None)
        if hooked is not None:
            return getattr(hooked, "params", None)
        return None

    def model_fingerprint(self) -> Optional[str]:
        """A content hash of the weights currently loaded, or `None`.

        Derived from tensor values, not from `self.spec.model_id`. A label
        is an assertion about provenance; a hash is evidence of it. The
        distinction is the whole point: a swapped checkpoint keeps its model
        id, keeps its shapes, and passes every tensor-level check, while the
        fingerprint changes. That is precisely the case this exists to
        catch -- activations captured under one checkpoint and patched under
        another are shape-compatible, so nothing downstream notices.

        Cached after the first call: the weights are fixed for the life of
        the adapter, and rehashing ~124M parameters on every result would be
        a real cost for no additional guarantee. A caller that mutates
        weights in place must invalidate the cache itself, which no current
        code path does -- recorded here rather than defended against.

        `None` when there are no weights to describe. Returning a string
        derived from a mock's construction would be worse than returning
        nothing: it would fingerprint a fiction.
        """
        if self._fingerprint is not None:
            return self._fingerprint
        self._fingerprint = hash_weights(self._weight_source())
        return self._fingerprint

    def tokenizer_fingerprint(self) -> Optional[str]:
        """A content hash of the loaded tokenizer's vocabulary, or `None`.

        A different tokenizer over the same weights produces different token
        ids, and therefore a different experiment: the weight hash answers
        "which checkpoint", this answers "which text-to-ids mapping reached
        it". A vocabulary that cannot be read is reported as absent rather
        than guessed at.
        """
        if self._tokenizer_fingerprint is not None:
            return self._tokenizer_fingerprint
        tokenizer = getattr(self, "_tokenizer", None)
        if tokenizer is None:
            return None
        try:
            vocabulary = tokenizer.get_vocab()
        except Exception:
            return None
        if not vocabulary:
            return None
        digest = hashlib.sha256()
        for token, index in sorted(vocabulary.items(), key=lambda item: str(item[0])):
            digest.update(f"{token}\t{index}\n".encode("utf-8"))
        self._tokenizer_fingerprint = f"sha256:{digest.hexdigest()[:32]}"
        return self._tokenizer_fingerprint

    def config_fingerprint(self) -> Optional[str]:
        """A content hash of the loaded configuration, or `None`.

        Read from the model that was actually loaded, not from the spec
        table: a hardcoded head count is how a Llama-3 gets measured as a
        2B, and a configuration change alters every index a result names
        without altering a single parameter.
        """
        if self._config_fingerprint is not None:
            return self._config_fingerprint
        config = getattr(self._weight_source(), "config", None)
        if config is None:
            return None
        fields: Dict[str, str] = {}
        for name in (
            "n_layer", "n_head", "n_embd", "n_positions", "vocab_size",
            "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
            "hidden_size", "intermediate_size", "max_position_embeddings",
            "torch_dtype",
        ):
            value = getattr(config, name, None)
            if value is not None:
                fields[name] = str(value)
        if not fields:
            return None
        payload = json.dumps(fields, sort_keys=True).encode("utf-8")
        self._config_fingerprint = f"sha256:{hashlib.sha256(payload).hexdigest()[:32]}"
        return self._config_fingerprint

    def model_identity(self) -> Optional[Dict[str, Any]]:
        """The identity of the model this adapter currently holds.

        `None` when there are no weights: without a checkpoint there is no
        experiment to identify, and an identity assembled from a mock's
        construction would be a fingerprint of a fiction.
        """
        weights = self.model_fingerprint()
        if weights is None:
            return None
        return {
            "model_fingerprint": weights,
            "tokenizer_fingerprint": self.tokenizer_fingerprint(),
            "config_fingerprint": self.config_fingerprint(),
        }

    def _publish_model_identity(self) -> None:
        """Make this adapter the model the process is currently working with.

        The activation cache has to answer "was this captured under the
        model that is loaded now?", and it cannot answer that from a stored
        entry: two checkpoints of one family have identical shapes, so
        every field an entry records still matches after a swap. The answer
        is process state, so the adapter holding the weights publishes it.

        The reference is weak: a published adapter that is garbage
        collected must not keep half a gigabyte of weights alive.
        """
        if self._weight_source() is None:
            return
        try:
            from backend.runtime.activation_cache import (
                register_model_identity_provider,
            )
        except Exception:
            return
        reference = weakref.ref(self)

        def provider() -> Optional[Dict[str, Any]]:
            adapter = reference()
            return None if adapter is None else adapter.model_identity()

        register_model_identity_provider(provider)

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
