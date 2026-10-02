"""Model Fingerprint Engine — Scientific Provenance for Weights & Config.

Captures SHA-256 of the actual loaded weights, configuration, and tokenizer
vocabulary so silent drift becomes detectable. The hashes are computed from
the real artifacts in memory; nothing here is a placeholder.

The engine fails closed. If the adapter is running in ``mock_mode`` or never
loaded real weights, ``capture()`` returns a fingerprint with ``attested=False``
and an explicit reason, and ``verify()`` refuses to report a match. An
unattested fingerprint is not evidence of anything.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import weakref
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

_CHUNK = 1 << 20  # 1 MiB


@dataclass
class ModelFingerprint:
    """Immutable snapshot of a model's identity and configuration."""
    model_id: str
    hf_repo_id: str
    revision: str
    weights_sha256: str
    config_sha256: str
    tokenizer_sha256: str
    parameter_count: int
    precision: str                # e.g., "float32", "bfloat16"
    quantization: Optional[str]   # e.g., "4bit", "8bit", None
    architecture: str
    recorded_at: str
    source_url: str
    # Fail-closed attestation state. `attested` is True only when the three
    # hashes above were computed from real loaded artifacts.
    attested: bool = False
    attestation_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelFingerprintEngine:
    """Generates and verifies model fingerprints from real loaded weights."""

    # Hashing GPT-2 small costs ~500 MB of reads, so cache per model object. A
    # WeakKeyDictionary is required rather than an id()-keyed dict: CPython
    # recycles ids after collection, which would let a different model hit a
    # stale entry and silently mask weight drift.
    _cache: "weakref.WeakKeyDictionary[Any, ModelFingerprint]" = (
        weakref.WeakKeyDictionary()
    )

    @staticmethod
    def _sha256_state_dict(model: Any) -> str:
        """Hash every tensor's name, dtype, shape, and raw bytes, in key order."""
        digest = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            flat = tensor.detach().to("cpu").contiguous().view(-1)
            buf = flat.numpy().tobytes()
            for start in range(0, len(buf), _CHUNK):
                digest.update(buf[start:start + _CHUNK])
        return "sha256:" + digest.hexdigest()

    @staticmethod
    def _sha256_config(model: Any) -> str:
        """Hash the model config canonically (sorted keys)."""
        cfg = model.config.to_dict()
        payload = json.dumps(cfg, sort_keys=True, default=str)
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _sha256_tokenizer(tokenizer: Any) -> str:
        """Hash the tokenizer vocabulary canonically.

        Hashing the vocab rather than the on-disk merge/vocab file keeps this
        portable across tokenizer implementations while still detecting any
        change in token->id mapping, which is what actually affects results.
        """
        vocab = tokenizer.get_vocab()
        digest = hashlib.sha256()
        for token in sorted(vocab):
            digest.update(token.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(str(vocab[token]).encode("ascii"))
            digest.update(b"\x01")
        return "sha256:" + digest.hexdigest()

    def _unattested(self, adapter: Any, reason: str) -> ModelFingerprint:
        """Build a fingerprint that is explicitly NOT evidence of real weights."""
        spec = adapter.spec
        return ModelFingerprint(
            model_id=spec.model_id,
            hf_repo_id=spec.hf_repo_id,
            revision="unknown",
            weights_sha256="unattested",
            config_sha256="unattested",
            tokenizer_sha256="unattested",
            parameter_count=0,
            precision="unknown",
            quantization=None,
            architecture="unknown",
            recorded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            source_url=f"https://huggingface.co/{spec.hf_repo_id}",
            attested=False,
            attestation_reason=reason,
        )

    def capture(self, adapter: Any) -> ModelFingerprint:
        """Capture a fingerprint from a ModelAdapter backed by real weights."""
        spec = adapter.spec

        if getattr(spec, "mock_mode", False):
            return self._unattested(
                adapter, "Adapter is in mock_mode; no weights were loaded.")
        model = getattr(adapter, "_model", None)
        if model is None:
            return self._unattested(
                adapter, "Adapter has no loaded model object; nothing to hash.")
        tokenizer = getattr(adapter, "_tokenizer", None)
        if tokenizer is None:
            return self._unattested(
                adapter, "Adapter has no loaded tokenizer; vocab cannot be hashed.")

        try:
            cached = self._cache.get(model)
        except TypeError:
            cached = None  # not weak-referenceable; fall through and re-hash
        if cached is not None:
            return cached

        try:
            weights_sha = self._sha256_state_dict(model)
            config_sha = self._sha256_config(model)
            tokenizer_sha = self._sha256_tokenizer(tokenizer)
        except Exception as exc:
            return self._unattested(
                adapter, f"Hashing failed, so identity cannot be attested: {exc}")

        param_count = sum(p.numel() for p in model.parameters())
        dtypes = {str(p.dtype) for p in model.parameters()}
        precision = dtypes.pop() if len(dtypes) == 1 else (
            "mixed(" + ",".join(sorted(dtypes)) + ")")
        architecture = (getattr(model.config, "architectures", None) or
                        [type(model).__name__])[0]
        quant = getattr(model.config, "quantization_config", None)
        quantization = type(quant).__name__ if quant is not None else None

        fingerprint = ModelFingerprint(
            model_id=spec.model_id,
            hf_repo_id=spec.hf_repo_id,
            revision=getattr(model.config, "_commit_hash", None) or "unknown",
            weights_sha256=weights_sha,
            config_sha256=config_sha,
            tokenizer_sha256=tokenizer_sha,
            parameter_count=param_count,
            precision=precision,
            quantization=quantization,
            architecture=architecture,
            recorded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            source_url=f"https://huggingface.co/{spec.hf_repo_id}",
            attested=True,
            attestation_reason="",
        )
        try:
            self._cache[model] = fingerprint
        except TypeError:
            pass  # not weak-referenceable; the next call simply re-hashes
        return fingerprint

    def verify(self, current: ModelFingerprint,
               reference: ModelFingerprint) -> Dict[str, Any]:
        """Compare two fingerprints to detect model drift.

        Never reports a match unless both sides are attested; an unattested
        fingerprint cannot corroborate anything.
        """
        if not (getattr(current, "attested", False)
                and getattr(reference, "attested", False)):
            reasons = [
                f"{label}: {getattr(fp, 'attestation_reason', 'unattested') or 'unattested'}"
                for label, fp in (("current", current), ("reference", reference))
                if not getattr(fp, "attested", False)
            ]
            return {
                "is_match": False,
                "mismatches": ["attestation"],
                "drift_severity": "UNVERIFIABLE",
                "reason": "; ".join(reasons),
            }

        mismatches = []
        if current.weights_sha256 != reference.weights_sha256:
            mismatches.append("weights_sha256")
        if current.config_sha256 != reference.config_sha256:
            mismatches.append("config_sha256")
        if current.tokenizer_sha256 != reference.tokenizer_sha256:
            mismatches.append("tokenizer_sha256")
        if current.parameter_count != reference.parameter_count:
            mismatches.append("parameter_count")

        return {
            "is_match": len(mismatches) == 0,
            "mismatches": mismatches,
            "drift_severity": "CRITICAL" if "weights_sha256" in mismatches else "LOW",
        }
