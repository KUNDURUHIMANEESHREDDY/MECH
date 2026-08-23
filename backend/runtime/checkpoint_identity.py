"""Checkpoint Identity Recorder & Verifier for MECH.

Records an immutable snapshot of a loaded model's identity at load time
and provides deterministic comparison against a stored baseline.

Fields captured:
    model_id        — registry name (e.g. "gpt2")
    hf_id           — HuggingFace hub id (e.g. "gpt2")
    architecture    — model architecture class name
    vocab_size      — tokenizer vocabulary size
    hidden_size     — embedding dimension
    num_layers      — transformer depth
    num_heads       — attention heads
    dtype           — primary weight dtype
    device          — device of first parameter
    weights_hash    — SHA-256 digest over first 200 (name, shape, dtype) tuples
    config_hash     — SHA-256 digest of model config dict (sorted keys)
    tokenizer_hash  — SHA-256 digest of tokenizer vocab (sorted)

Invariant enforced:
    RUNTIME_VALIDATED ≠ MODEL_SCIENTIFICALLY_VALIDATED

    compute_checkpoint_identity() may succeed even when the wrong weights
    are loaded.  Callers must compare the returned identity against a known
    good baseline before trusting mechanistic results.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class CheckpointIdentity:
    """Immutable snapshot of a model checkpoint's identity at load time."""
    model_id: str
    hf_id: str
    architecture: str           # e.g. "GPT2LMHeadModel"
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    dtype: str                  # e.g. "torch.float32"
    device: str                 # e.g. "cpu" or "cuda:0"
    weights_hash: str           # 64-char SHA-256 hex
    config_hash: str            # 64-char SHA-256 hex of sorted config dict
    tokenizer_hash: str         # 64-char SHA-256 hex of sorted vocab

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def matches(self, other: "CheckpointIdentity") -> bool:
        """True iff all hash fields and structural fields agree."""
        return (
            self.model_id      == other.model_id
            and self.architecture  == other.architecture
            and self.vocab_size    == other.vocab_size
            and self.hidden_size   == other.hidden_size
            and self.num_layers    == other.num_layers
            and self.weights_hash  == other.weights_hash
            and self.config_hash   == other.config_hash
            and self.tokenizer_hash == other.tokenizer_hash
        )

    def diff(self, other: "CheckpointIdentity") -> Dict[str, Any]:
        """Returns a dict of fields that differ between self and other."""
        mismatches: Dict[str, Any] = {}
        for field_name in (
            "model_id", "architecture", "vocab_size", "hidden_size",
            "num_layers", "num_heads", "dtype", "weights_hash",
            "config_hash", "tokenizer_hash",
        ):
            a = getattr(self, field_name)
            b = getattr(other, field_name)
            if a != b:
                mismatches[field_name] = {"expected": a, "observed": b}
        return mismatches


def _hash_weights(model) -> str:
    """SHA-256 over first 200 (param_name, shape, dtype) tuples."""
    parts = []
    for name, param in model.named_parameters():
        parts.append(f"{name}:{list(param.shape)}:{param.dtype}")
        if len(parts) >= 200:
            break
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _hash_config(config) -> str:
    """SHA-256 over the model config dict with sorted keys."""
    try:
        cfg_dict = config.to_dict() if hasattr(config, "to_dict") else vars(config)
        # Exclude non-deterministic fields like _name_or_path which may vary by load path
        clean = {
            k: v for k, v in cfg_dict.items()
            if not k.startswith("_") and isinstance(v, (int, float, str, bool, list, type(None)))
        }
        serialised = json.dumps(clean, sort_keys=True, default=str)
        return hashlib.sha256(serialised.encode()).hexdigest()
    except (TypeError, ValueError, AttributeError) as exc:
        logger.warning("Failed to compute config hash, marking unavailable: %s", exc)
        return "HASH_UNAVAILABLE"


def _hash_tokenizer(tokenizer) -> str:
    """SHA-256 over the sorted vocabulary mapping."""
    try:
        vocab = tokenizer.get_vocab()          # {token_str: id, ...}
        serialised = json.dumps(vocab, sort_keys=True)
        return hashlib.sha256(serialised.encode()).hexdigest()
    except (AttributeError, OSError, ValueError, TypeError) as exc:
        logger.warning("Failed to compute tokenizer hash, marking unavailable: %s", exc)
        return "HASH_UNAVAILABLE"


def compute_checkpoint_identity(
    model,
    tokenizer,
    model_id: str,
    hf_id: str,
) -> CheckpointIdentity:
    """
    Derives a CheckpointIdentity snapshot from a loaded model and tokenizer.

    Parameters
    ----------
    model      : loaded nn.Module (e.g. GPT2LMHeadModel)
    tokenizer  : loaded PreTrainedTokenizer
    model_id   : registry key (e.g. "gpt2")
    hf_id      : HuggingFace hub id (e.g. "gpt2")

    Returns
    -------
    CheckpointIdentity — all hashes computed at call time
    """
    import torch

    config = model.config
    arch = type(model).__name__

    # Structural metadata
    vocab_size  = getattr(config, "vocab_size",
                          getattr(tokenizer, "vocab_size", -1))
    hidden_size = getattr(config, "n_embd",  getattr(config, "hidden_size", -1))
    num_layers  = getattr(config, "n_layer", getattr(config, "num_hidden_layers", -1))
    num_heads   = getattr(config, "n_head",  getattr(config, "num_attention_heads", -1))

    # dtype and device from first parameter
    try:
        first_param = next(iter(model.parameters()))
        dtype  = str(first_param.dtype)
        device = str(first_param.device)
    except StopIteration:
        dtype  = "unknown"
        device = "unknown"

    return CheckpointIdentity(
        model_id=model_id,
        hf_id=hf_id,
        architecture=arch,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        dtype=dtype,
        device=device,
        weights_hash=_hash_weights(model),
        config_hash=_hash_config(config),
        tokenizer_hash=_hash_tokenizer(tokenizer),
    )


def verify_checkpoint_identity(
    live: CheckpointIdentity,
    baseline: CheckpointIdentity,
) -> Dict[str, Any]:
    """
    Compares a live identity against a stored baseline.

    Returns
    -------
    dict with keys:
        verified  : bool
        mismatches: dict of differing fields (empty on success)
        summary   : human-readable verdict string
    """
    mismatches = baseline.diff(live)
    verified   = len(mismatches) == 0
    if verified:
        summary = f"Checkpoint identity VERIFIED for '{live.model_id}' — all hashes match."
    else:
        fields = ", ".join(mismatches.keys())
        summary = (
            f"Checkpoint identity MISMATCH for '{live.model_id}' — "
            f"differing fields: [{fields}]. "
            f"Do not trust mechanistic results until the correct checkpoint is loaded."
        )
    return {"verified": verified, "mismatches": mismatches, "summary": summary}
