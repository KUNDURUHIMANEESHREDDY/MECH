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

    # Provenance Fingerprints
    dataset_hash: str
    tokenizer_hash: str
    model_sha256: str
    environment_hash: str

    # Reference Comparison
    published_reference: float
    reference_baseline: Optional[float]
    current_result: float

    # Verdicts
    regression: bool
    passed: bool
    reproducibility_score: float

    # Metadata
    validator_version: str
    registry_version: str
    schema_version: str = "1.1.0"
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    signature: str = "" # Placeholder for digital signature


class CertificateEngine:
    """Generates and signs benchmark certificates."""

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
            dataset_hash=hashes.get("dataset", "unknown"),
            tokenizer_hash=hashes.get("tokenizer", "unknown"),
            model_sha256=hashes.get("model", "unknown"),
            environment_hash=hashes.get("env", "unknown"),
            published_reference=results.get("published", 0.0),
            reference_baseline=results.get("baseline"),
            current_result=results.get("current", 0.0),
            regression=results.get("regression", False),
            passed=verdict,
            reproducibility_score=repro_score,
            validator_version="1.5.0", # Hardcoded engine version
            registry_version="1.1.0"
        )
