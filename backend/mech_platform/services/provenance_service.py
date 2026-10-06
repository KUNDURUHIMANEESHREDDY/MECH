"""Provenance Service.

Tracks pipeline, model, dataset, plugin versions, and reproducibility checksums.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from typing import Any, Dict


class ProvenanceService:
    """Service for recording experiment execution provenance."""

    def record_provenance(
        self,
        experiment_id: str,
        pipeline_version: str = "2.0.0",
        model_version: str = "GPT-2 Small",
        dataset_version: str = "OpenWebText v1",
    ) -> Dict[str, Any]:
        """Record what an experiment claims to have been run with.

        The checksum used to be

            f"sha256_{hash(experiment_id + pipeline_version) & 0xffffffff:08x}"

        which is not a SHA-256 digest. It is eight hex characters of Python's
        built-in `hash()`, which is SipHash-1-3 over a salted string: it differs
        between processes, it is reversible-ish by brute force over the input
        space, and it collides after ~2^16 distinct inputs by the birthday bound
        rather than 2^128. Labelling it `sha256_` asserted a cryptographic
        property it did not have, in a field whose entire purpose is to let
        someone check whether two records describe the same run.

        Now it is a real SHA-256 over the canonical field values, so it is
        stable across processes and machines and it actually detects a change to
        what it covers.
        """
        covered = {
            "experiment_id": str(experiment_id),
            "pipeline_version": str(pipeline_version),
            "model_version": str(model_version),
            "dataset_version": str(dataset_version),
        }
        canonical = "\n".join(f"{k}={covered[k]}" for k in sorted(covered))
        return {
            **covered,
            "checksum_algorithm": "sha256",
            "checksum": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "checksum_covers": sorted(covered),
            "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
        }
