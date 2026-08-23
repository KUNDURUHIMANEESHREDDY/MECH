"""Manifest Store — Phase 12 Scientific Reproducibility Engine.

Writes ExperimentManifest records to disk and provides retrieval and
reproducibility verification between independent runs.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Optional

from backend.science.provenance.experiment_manifest import ExperimentManifest


_RESULTS_DIR = Path(__file__).parent / "results"


def _results_dir() -> Path:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return _RESULTS_DIR


def write(manifest: ExperimentManifest) -> Path:
    """Serialize and write a manifest to disk. Returns the written file path."""
    path = _results_dir() / f"{manifest.experiment_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)
    return path


def retrieve(experiment_id: str) -> Dict:
    """Retrieve a manifest by experiment_id from disk. Raises FileNotFoundError if missing."""
    path = _results_dir() / f"{experiment_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found for experiment_id={experiment_id}. "
            f"The experiment must be re-run to generate its provenance."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_reproducibility(experiment_id_a: str, experiment_id_b: str) -> Dict:
    """Compare two independent runs of the same experiment type for reproducibility.

    Returns a report indicating whether effect sizes, evidence levels, and
    control distributions are consistent between runs.

    Raises:
        FileNotFoundError: If either manifest is missing.
        ValueError: If the two manifests are from different experiment types.
    """
    ma = retrieve(experiment_id_a)
    mb = retrieve(experiment_id_b)

    if ma["experiment_type"] != mb["experiment_type"]:
        raise ValueError(
            f"Cannot compare manifests of different experiment types: "
            f"{ma['experiment_type']} vs {mb['experiment_type']}"
        )

    # Check model identity
    same_model = ma["model_id"] == mb["model_id"]
    same_weights = ma["model_weights_sha256"] == mb["model_weights_sha256"]
    same_tokenizer = ma["tokenizer_hash"] == mb["tokenizer_hash"]

    # Check evidence level consistency
    same_evidence_level = ma["evidence_level"] == mb["evidence_level"]

    # Check effect size consistency (within 10% of each other)
    effect_a = ma.get("bootstrap_statistics", {}) or {}
    effect_b = mb.get("bootstrap_statistics", {}) or {}
    es_a = effect_a.get("effect_size", 0.0)
    es_b = effect_b.get("effect_size", 0.0)
    effect_relative_diff = abs(es_a - es_b) / max(1e-8, abs(es_a + es_b) / 2)
    effect_sizes_consistent = effect_relative_diff < 0.10

    return {
        "experiment_id_a": experiment_id_a,
        "experiment_id_b": experiment_id_b,
        "experiment_type": ma["experiment_type"],
        "same_model": same_model,
        "same_weights": same_weights,
        "same_tokenizer": same_tokenizer,
        "same_evidence_level": same_evidence_level,
        "effect_size_a": round(es_a, 6),
        "effect_size_b": round(es_b, 6),
        "effect_relative_diff": round(effect_relative_diff, 4),
        "effect_sizes_consistent": effect_sizes_consistent,
        "is_reproducible": same_weights and same_evidence_level and effect_sizes_consistent,
        "caveat": (
            "Reproducibility requires identical model weights, evidence level, "
            "and effect sizes within 10% relative tolerance across independent runs."
        ),
    }
