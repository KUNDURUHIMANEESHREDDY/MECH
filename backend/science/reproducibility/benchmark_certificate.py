"""Benchmark Certificate — The Scientific Proof of Validity.

Contains the definitive results of a benchmark run, signed with hashes
of the model, dataset, and environment to ensure provenance.
"""

from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkCertificate:
    """A signed certificate of benchmark performance and validity."""
    certificate_id: str
    benchmark_id: str
    benchmark_version: str

    # Provenance Fingerprints.
    #   Every field is Optional. They defaulted to the string "unknown", which is
    #   indistinguishable from a recorded value once it is in a certificate.
    #
    #   `dataset_hash` and `environment_hash` previously held a dataset id and a
    #   snapshot id respectively -- identifiers stored under names ending in
    #   `_hash`, so a reader checking "which weights produced this" was looking at
    #   a snapshot label. `dataset_id` and `environment_id` now carry those, and
    #   the `_hash` fields carry None unless a real digest was supplied.
    dataset_hash: Optional[str]
    tokenizer_hash: Optional[str]
    model_sha256: Optional[str]
    environment_hash: Optional[str]

    # Reference Comparison
    #   `published_reference` is Optional and defaults to None, not 0.0. The
    #   caller used to be handed `results.get("published", 0.0)`, so a run that
    #   never supplied a literature figure produced a certificate asserting a
    #   comparison against 0.0 -- a comparison nobody made. None means "no
    #   published reference was supplied".
    published_reference: Optional[float]
    reference_baseline: Optional[float]
    current_result: float

    # Verdicts
    #   `regression` is Optional for the same reason: the caller did not pass it,
    #   so the old `results.get("regression", False)` reported "no regression
    #   detected" on every certificate, which is a clean bill of health produced
    #   by a default rather than by a comparison. None means "not assessed".
    regression: Optional[bool]
    passed: bool
    reproducibility_score: float

    # Metadata
    validator_version: str
    registry_version: str
    schema_version: str = "1.1.0"
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    #: Labels, kept out of the `*_hash` fields above.
    dataset_id: str = ""
    environment_id: str = ""


def _digest(value: Any) -> Optional[str]:
    """Return a hex digest, or None.

    Rejects placeholder and non-digest values. A field named `*_hash` holding
    `"unknown"`, a dataset id, or a snapshot label tells a reader checking which
    weights produced a result that nothing was recorded, while looking like a
    recorded value.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"unknown", "none", "null"}:
        return None
    if "placeholder" in text.lower():
        return None
    body = text[len("sha256:"):] if text.lower().startswith("sha256:") else text
    if len(body) == 64 and all(c in "0123456789abcdefABCDEF" for c in body):
        return body
    return None


class CertificateEngine:
    """Generates benchmark certificates.

    Does not sign anything, despite the class docstring's former claim: signing
    happens in `ResearchManifestEngine.sign_manifest` over the manifest's Merkle
    root, which in turn covers this certificate's hash.
    """

    def generate(
        self,
        benchmark_id: str,
        results: Dict[str, Any],
        hashes: Dict[str, str],
        repro_score: float,
        verdict: bool
    ) -> BenchmarkCertificate:
        """Assembles the final scientific certificate."""
        import uuid
        cert_id = f"CERT-{uuid.uuid4().hex[:8].upper()}"

        return BenchmarkCertificate(
            certificate_id=cert_id,
            benchmark_id=benchmark_id,
            benchmark_version=results.get("version", "v1.0"),
            dataset_hash=_digest(hashes.get("dataset_hash")),
            tokenizer_hash=_digest(hashes.get("tokenizer")),
            model_sha256=_digest(hashes.get("model")),
            environment_hash=_digest(hashes.get("env_hash")),
            dataset_id=str(hashes.get("dataset_id") or hashes.get("dataset") or ""),
            environment_id=str(hashes.get("environment_id") or hashes.get("env") or ""),
            published_reference=results.get("published"),
            reference_baseline=results.get("baseline"),
            current_result=results.get("current", 0.0),
            regression=results.get("regression"),
            passed=verdict,
            reproducibility_score=repro_score,
            validator_version="1.5.0", # Hardcoded engine version
            registry_version="1.1.0"
        )
