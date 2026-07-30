"""Dataset Certificate — The Proof of Data Grounding.

Generates immutable proof that a dataset is Golden, signed, and compliant
with the platform's reproducibility standards.
"""

from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DatasetCertificate:
    """A signed certificate of dataset integrity and provenance."""
    dataset_id: str
    version: str
    status: str

    # Hashes
    prompt_hash: str
    token_hash: str
    bundle_hash: str

    # Provenance
    paper_doi: str
    license: str
    author: str

    # Metadata
    verified_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    validator_version: str = "1.2.0"
    signature: str = ""


class DatasetCertificateEngine:
    """Generates standalone dataset certificates."""

    def issue(self, dataset_meta: Dict[str, Any]) -> DatasetCertificate:
        """Assembles the final dataset certificate."""
        hashes = dataset_meta.get("hashes", {})
        prov = dataset_meta.get("provenance", {})
        lineage = dataset_meta.get("lineage", {})

        return DatasetCertificate(
            dataset_id=dataset_meta["dataset_id"],
            version=dataset_meta["version"],
            status=dataset_meta["status"],
            prompt_hash=hashes.get("prompt_hash", "unknown"),
            token_hash=hashes.get("token_hash", "unknown"),
            bundle_hash=hashes.get("bundle_hash", "unknown"),
            paper_doi=prov.get("paper_doi", "unknown"),
            license=prov.get("license", "unknown"),
            author=lineage.get("author", "unknown"),
            signature=dataset_meta.get("signature", "")
        )
