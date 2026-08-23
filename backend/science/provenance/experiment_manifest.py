"""Experiment Provenance Manifest — Phase 12 Scientific Reproducibility Engine.

Every MECH scientific result must be fully reconstructible from its manifest.
If any required field is missing, the experiment returns UNEXECUTED.

Provenance chain requirement:
    Result
      ↓
    ExperimentManifest
      ↓
    Raw measurements + control distribution
      ↓
    Live model execution
      ↓
    Model / checkpoint hash
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_str(s: str) -> str:
    return _sha256_bytes(s.encode("utf-8"))


def _library_versions() -> Dict[str, str]:
    """Capture versions of all scientific dependencies at execution time."""
    versions: Dict[str, str] = {}
    for lib in ["torch", "transformer_lens", "sae_lens", "bertviz", "circuitsvis", "numpy", "scipy"]:
        try:
            mod = __import__(lib)
            versions[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[lib] = "NOT_INSTALLED"
    versions["python"] = sys.version.split()[0]
    return versions


def _hash_model_state_dict(model) -> str:
    """Compute SHA-256 of the serialized model state_dict for provenance."""
    import io
    import torch
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return _sha256_bytes(buf.getvalue())


def _hash_tokenizer_vocab(tokenizer) -> str:
    """Compute SHA-256 of tokenizer vocabulary for provenance."""
    vocab_str = json.dumps(tokenizer.get_vocab(), sort_keys=True)
    return _sha256_str(vocab_str)


def _hash_prompts(prompts: List[str]) -> str:
    """Compute SHA-256 of the prompt list used in the experiment."""
    combined = "\n".join(prompts)
    return _sha256_str(combined)


@dataclass
class ControlDistribution:
    """Empirical null/control distribution from matched controls."""
    n_controls: int
    raw_values: List[float]
    mean: float
    std: float
    median: float
    percentile_95: float
    percentile_99: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_controls": self.n_controls,
            "raw_values": [round(v, 6) for v in self.raw_values],
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
            "median": round(self.median, 6),
            "percentile_95": round(self.percentile_95, 6),
            "percentile_99": round(self.percentile_99, 6),
        }


@dataclass
class BootstrapStatistics:
    """Bootstrap confidence interval statistics."""
    n_bootstrap: int
    effect_size: float
    ci_lower: float
    ci_upper: float
    ci_width: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_bootstrap": self.n_bootstrap,
            "effect_size": round(self.effect_size, 6),
            "ci_lower": round(self.ci_lower, 6),
            "ci_upper": round(self.ci_upper, 6),
            "ci_width": round(self.ci_width, 6),
        }


@dataclass
class ExperimentManifest:
    """Complete provenance record for a single MECH scientific experiment.

    REQUIRED FIELDS: If any required field is None/missing, the experiment
    must return UNEXECUTED — never a default/zero measurement.
    """
    # Identity
    experiment_id: str
    experiment_type: str           # "induction_head" | "sae_feature" | "causal_circuit" | "dla_fidelity" | "logit_lens"
    timestamp_utc: str
    mech_version: str

    # Hardware & Runtime
    device: str
    precision: str                 # "float32" | "float16"
    library_versions: Dict[str, str]

    # Model Provenance
    model_id: str
    model_weights_sha256: str      # REQUIRED: SHA-256 of model state_dict

    # Tokenizer Provenance
    tokenizer_hash: str            # REQUIRED: SHA-256 of tokenizer vocab

    # SAE Provenance (optional)
    sae_id: Optional[str]
    sae_weights_sha256: Optional[str]

    # Input Provenance
    prompt_dataset_hash: str       # REQUIRED: SHA-256 of all prompts used
    prompts_used: List[str]
    random_seeds: List[int]

    # Intervention Specification
    intervention_specification: Dict[str, Any]

    # Raw Measurements (ALL stored as measured floats, not assertions)
    baseline_measurements: Dict[str, float]       # e.g. clean logit, clean prob
    raw_effect_measurements: List[float]          # e.g. per-head IE scores
    control_distribution: Optional[ControlDistribution]   # empirical null
    replication_measurements: List[Dict[str, float]]      # per-template replications

    # Statistical Analysis
    bootstrap_statistics: Optional[BootstrapStatistics]
    spearman_rho: Optional[float]                 # For DLA fidelity
    spearman_p_value: Optional[float]
    pearson_r: Optional[float]
    dla_approximation_quality: Optional[str]      # "HIGH" | "MODERATE" | "POOR"

    # Epistemic Outcome
    evidence_level: str            # EvidenceLevel enum value
    falsification_status: str      # "FALSIFIED" | "SUPPORTED" | "CAUSALLY_VERIFIED" | "UNEXECUTED" | ...
    is_reproducible: bool
    statistical_caveat: str

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "experiment_id": self.experiment_id,
            "experiment_type": self.experiment_type,
            "timestamp_utc": self.timestamp_utc,
            "mech_version": self.mech_version,
            "device": self.device,
            "precision": self.precision,
            "library_versions": self.library_versions,
            "model_id": self.model_id,
            "model_weights_sha256": self.model_weights_sha256,
            "tokenizer_hash": self.tokenizer_hash,
            "sae_id": self.sae_id,
            "sae_weights_sha256": self.sae_weights_sha256,
            "prompt_dataset_hash": self.prompt_dataset_hash,
            "prompts_used": self.prompts_used,
            "random_seeds": self.random_seeds,
            "intervention_specification": self.intervention_specification,
            "baseline_measurements": {k: round(v, 6) for k, v in self.baseline_measurements.items()},
            "raw_effect_measurements": [round(v, 6) for v in self.raw_effect_measurements],
            "control_distribution": self.control_distribution.to_dict() if self.control_distribution else None,
            "replication_measurements": self.replication_measurements,
            "bootstrap_statistics": self.bootstrap_statistics.to_dict() if self.bootstrap_statistics else None,
            "spearman_rho": round(self.spearman_rho, 6) if self.spearman_rho is not None else None,
            "spearman_p_value": round(self.spearman_p_value, 6) if self.spearman_p_value is not None else None,
            "pearson_r": round(self.pearson_r, 6) if self.pearson_r is not None else None,
            "dla_approximation_quality": self.dla_approximation_quality,
            "evidence_level": self.evidence_level,
            "falsification_status": self.falsification_status,
            "is_reproducible": self.is_reproducible,
            "statistical_caveat": self.statistical_caveat,
        }
        return d


def build_control_distribution(raw_values: List[float]) -> ControlDistribution:
    """Compute empirical null distribution statistics from raw control measurements."""
    import numpy as np
    arr = np.array(raw_values, dtype=np.float64)
    return ControlDistribution(
        n_controls=len(arr),
        raw_values=list(arr),
        mean=float(arr.mean()),
        std=float(arr.std()),
        median=float(np.median(arr)),
        percentile_95=float(np.percentile(arr, 95)),
        percentile_99=float(np.percentile(arr, 99)),
    )


def build_bootstrap_statistics(
    effect_values: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> BootstrapStatistics:
    """Compute bootstrap CI around the mean of effect values."""
    import numpy as np
    arr = np.array(effect_values, dtype=np.float64)
    rng = np.random.default_rng(42)
    bootstrap_means = [
        float(rng.choice(arr, size=len(arr), replace=True).mean())
        for _ in range(n_bootstrap)
    ]
    alpha = (1 - ci) / 2
    ci_lower = float(np.percentile(bootstrap_means, alpha * 100))
    ci_upper = float(np.percentile(bootstrap_means, (1 - alpha) * 100))
    return BootstrapStatistics(
        n_bootstrap=n_bootstrap,
        effect_size=float(arr.mean()),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_width=ci_upper - ci_lower,
    )
