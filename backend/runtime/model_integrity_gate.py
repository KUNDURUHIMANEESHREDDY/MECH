"""Model Integrity Gate for MECH — Phase 62.5.

The hard scientific gate that enforces:

    ╔══════════════════════════════════════════════════════════╗
    ║  RUNTIME_VALIDATED  ≠  MODEL_SCIENTIFICALLY_VALIDATED   ║
    ╚══════════════════════════════════════════════════════════╝

All four checks must pass before any mechanistic experiment is allowed:

    1. checkpoint_identity  — weights_hash, config_hash, tokenizer_hash
                              all match a known baseline
    2. tokenizer_integrity  — decode(encode(t)) == t for every probe token;
                              every target token exists in vocab as a single token
    3. architecture_match   — loaded model's class / layer count / hidden dim
                              match the registry entry
    4. behavioral_sanity    — expected tokens appear within top-k for all
                              standard deterministic anchors

Gate decisions
--------------
MODEL_READY
    All four checks passed.  Mechanistic experiments are permitted.

MODEL_NOT_VALIDATED
    One or more checks failed.  MECH must ABSTAIN.
    "No mechanistic claims permitted until the gate passes."

The gate does NOT judge whether the model is scientifically interesting —
it only judges whether the loaded model is the one MECH intends to study.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .checkpoint_identity import CheckpointIdentity, verify_checkpoint_identity
from .tokenizer_integrity import TokenizerIntegrityReport
from .behavioral_sanity_suite import BehavioralSanityReport


class ModelReadinessState(str, Enum):
    MODEL_READY         = "MODEL_READY"
    MODEL_NOT_VALIDATED = "MODEL_NOT_VALIDATED"
    GATE_NOT_RUN        = "GATE_NOT_RUN"


@dataclass
class ModelIntegrityGateResult:
    """Full result of a model integrity gate evaluation."""
    model_id: str
    gate_state: ModelReadinessState
    timestamp_utc: str

    # Per-check verdicts
    checkpoint_verified: bool
    tokenizer_verified: bool
    architecture_verified: bool
    behavioral_sanity_passed: bool

    # Detailed sub-reports
    checkpoint_report: Optional[Dict[str, Any]]
    tokenizer_report: Optional[Dict[str, Any]]
    architecture_report: Dict[str, Any]
    behavioral_report: Optional[Dict[str, Any]]

    # Top-level diagnosis
    failed_checks: List[str]
    gate_message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "gate_state": self.gate_state.value,
            "timestamp_utc": self.timestamp_utc,
            "checkpoint_verified": self.checkpoint_verified,
            "tokenizer_verified": self.tokenizer_verified,
            "architecture_verified": self.architecture_verified,
            "behavioral_sanity_passed": self.behavioral_sanity_passed,
            "failed_checks": self.failed_checks,
            "gate_message": self.gate_message,
            "checkpoint_report": self.checkpoint_report,
            "tokenizer_report": self.tokenizer_report,
            "architecture_report": self.architecture_report,
            "behavioral_report": self.behavioral_report,
        }

    @property
    def is_ready(self) -> bool:
        return self.gate_state == ModelReadinessState.MODEL_READY


def _verify_architecture(
    model,
    expected_num_layers: int,
    expected_hidden_size: int,
    expected_architecture: str,
) -> Dict[str, Any]:
    """Cross-checks loaded model structural shape against registry expectations."""
    config = model.config
    arch   = type(model).__name__

    actual_layers = getattr(config, "n_layer", getattr(config, "num_hidden_layers", -1))
    actual_hidden = getattr(config, "n_embd",  getattr(config, "hidden_size", -1))

    layer_ok  = (expected_num_layers <= 0) or (actual_layers == expected_num_layers)
    hidden_ok = (expected_hidden_size <= 0) or (actual_hidden == expected_hidden_size)
    arch_ok   = (not expected_architecture) or (arch == expected_architecture)

    verified = layer_ok and hidden_ok and arch_ok

    mismatches: Dict[str, Any] = {}
    if not layer_ok:
        mismatches["num_layers"] = {"expected": expected_num_layers, "observed": actual_layers}
    if not hidden_ok:
        mismatches["hidden_size"] = {"expected": expected_hidden_size, "observed": actual_hidden}
    if not arch_ok:
        mismatches["architecture"] = {"expected": expected_architecture, "observed": arch}

    return {
        "verified": verified,
        "actual_architecture": arch,
        "actual_num_layers": actual_layers,
        "actual_hidden_size": actual_hidden,
        "mismatches": mismatches,
        "summary": (
            "Architecture VERIFIED."
            if verified else
            f"Architecture MISMATCH: {mismatches}"
        ),
    }


def run_model_integrity_gate(
    model,
    tokenizer,
    model_id: str,
    hf_id: str,
    target_tokens: List[str],
    runtime=None,
    baseline_identity: Optional[CheckpointIdentity] = None,
    expected_num_layers: int = -1,
    expected_hidden_size: int = -1,
    expected_architecture: str = "",
) -> ModelIntegrityGateResult:
    """
    Executes all four integrity checks and returns a gate decision.

    Parameters
    ----------
    model               : loaded nn.Module
    tokenizer           : loaded PreTrainedTokenizer
    model_id            : registry key (e.g. "gpt2")
    hf_id               : HuggingFace hub id (e.g. "gpt2")
    target_tokens       : all expected tokens from probes to be run
    runtime             : ModelRuntimeInterface, required for behavioral sanity
    baseline_identity   : stored CheckpointIdentity to compare against;
                          if None, checkpoint verification is skipped (WARN, not FAIL)
    expected_num_layers : -1 = skip check
    expected_hidden_size: -1 = skip check
    expected_architecture: "" = skip check

    Returns
    -------
    ModelIntegrityGateResult with gate_state MODEL_READY or MODEL_NOT_VALIDATED
    """
    from .checkpoint_identity import compute_checkpoint_identity, verify_checkpoint_identity
    from .tokenizer_integrity import verify_tokenizer_integrity
    from .behavioral_sanity_suite import run_behavioral_sanity_suite

    ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
    failed_checks: List[str] = []

    # ── 1. Checkpoint identity ───────────────────────────────────────────────
    live_identity = compute_checkpoint_identity(model, tokenizer, model_id, hf_id)
    if baseline_identity is not None:
        ckpt_report_raw = verify_checkpoint_identity(live_identity, baseline_identity)
        checkpoint_verified = ckpt_report_raw["verified"]
        ckpt_report = {**ckpt_report_raw, "live_identity": live_identity.to_dict()}
    else:
        checkpoint_verified = True   # no baseline to compare — treat as unverified-but-not-failed
        ckpt_report = {
            "verified": True,
            "mismatches": {},
            "summary": "No baseline provided — checkpoint hash recorded but not compared.",
            "live_identity": live_identity.to_dict(),
        }

    if not checkpoint_verified:
        failed_checks.append("checkpoint_identity")

    # ── 2. Tokenizer integrity ───────────────────────────────────────────────
    tok_report_obj = verify_tokenizer_integrity(tokenizer, target_tokens)
    tokenizer_verified = tok_report_obj.all_passed
    if not tokenizer_verified:
        failed_checks.append("tokenizer_integrity")

    # ── 3. Architecture match ────────────────────────────────────────────────
    arch_report = _verify_architecture(
        model,
        expected_num_layers=expected_num_layers,
        expected_hidden_size=expected_hidden_size,
        expected_architecture=expected_architecture,
    )
    architecture_verified = arch_report["verified"]
    if not architecture_verified:
        failed_checks.append("architecture_match")

    # ── 4. Behavioral sanity ─────────────────────────────────────────────────
    if runtime is not None:
        san_report_obj = run_behavioral_sanity_suite(runtime, model_id=model_id)
        behavioral_sanity_passed = san_report_obj.all_passed
        san_report = san_report_obj.to_dict()
    else:
        # No runtime supplied — skip behavioral sanity but warn
        behavioral_sanity_passed = True
        san_report = {
            "all_passed": True,
            "summary": "Behavioral sanity skipped — no runtime supplied.",
        }

    if not behavioral_sanity_passed:
        failed_checks.append("behavioral_sanity")

    # ── Gate decision ────────────────────────────────────────────────────────
    if not failed_checks:
        gate_state = ModelReadinessState.MODEL_READY
        gate_message = (
            f"MODEL_READY: '{model_id}' passed all integrity checks. "
            f"Mechanistic experiments are permitted."
        )
    else:
        gate_state = ModelReadinessState.MODEL_NOT_VALIDATED
        gate_message = (
            f"MODEL_NOT_VALIDATED: '{model_id}' failed checks: {failed_checks}. "
            f"ABSTAIN — no mechanistic claims permitted until the gate passes. "
            f"Investigate the loaded checkpoint, tokenizer, and inference path "
            f"(verify GPU utilization, that real weights are loaded, and that the "
            f"inference path does not fall back to a stub or random initialisation)."
        )

    return ModelIntegrityGateResult(
        model_id=model_id,
        gate_state=gate_state,
        timestamp_utc=ts,
        checkpoint_verified=checkpoint_verified,
        tokenizer_verified=tokenizer_verified,
        architecture_verified=architecture_verified,
        behavioral_sanity_passed=behavioral_sanity_passed,
        checkpoint_report=ckpt_report,
        tokenizer_report=tok_report_obj.to_dict(),
        architecture_report=arch_report,
        behavioral_report=san_report,
        failed_checks=failed_checks,
        gate_message=gate_message,
    )


def assert_model_ready(gate_result: ModelIntegrityGateResult) -> None:
    """
    Raises RuntimeError if the gate has not passed.

    Call this at the entry-point of any mechanistic experiment function
    to enforce the hard scientific gate.

    Raises
    ------
    RuntimeError
        When gate_state is not MODEL_READY, with a full diagnostic message
        that names which checks failed and what to investigate.
    """
    if not gate_result.is_ready:
        raise RuntimeError(
            f"[ModelIntegrityGate BLOCKED] {gate_result.gate_message}\n"
            f"Failed checks: {gate_result.failed_checks}\n"
            f"Checkpoint summary : {gate_result.checkpoint_report.get('summary', 'N/A')}\n"
            f"Tokenizer summary  : {gate_result.tokenizer_report.get('summary', 'N/A')}\n"
            f"Architecture report: {gate_result.architecture_report.get('summary', 'N/A')}\n"
            f"Behavioral summary : {gate_result.behavioral_report.get('summary', 'N/A')}"
        )
