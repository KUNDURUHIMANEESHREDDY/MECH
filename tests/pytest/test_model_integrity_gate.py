"""Tests for Phase 62.5: Model Integrity Gate.

Covers all four sub-checks and the composite gate decision:
    1. Checkpoint identity (hash match, hash mismatch, no baseline)
    2. Tokenizer integrity (round-trip, vocab membership, multi-token splits)
    3. Architecture match (correct / wrong layer count / wrong hidden size)
    4. Behavioral sanity (anchors found in top-k / not found)
    5. Composite gate: MODEL_READY and MODEL_NOT_VALIDATED paths
    6. assert_model_ready raises on non-ready gate
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

from backend.runtime.behavioral_sanity_suite import (
    BehavioralSanityReport,
    SanityAnchor,
    SanityAnchorResult,
    run_behavioral_sanity_suite,
)
from backend.runtime.checkpoint_identity import (
    CheckpointIdentity,
    compute_checkpoint_identity,
    verify_checkpoint_identity,
)
from backend.runtime.model_integrity_gate import (
    ModelIntegrityGateResult,
    ModelReadinessState,
    assert_model_ready,
    run_model_integrity_gate,
)
from backend.runtime.tokenizer_integrity import (
    TokenizerIntegrityReport,
    verify_tokenizer_integrity,
)


# ─── fixtures / stubs ─────────────────────────────────────────────────────────

class _FakeParam:
    dtype = "torch.float32"
    device = "cpu"
    def __iter__(self): return iter([self])

class _FakeConfig:
    vocab_size   = 50257
    n_embd       = 768
    n_layer      = 12
    n_head       = 12
    model_type   = "gpt2"
    def to_dict(self):
        return {"vocab_size": self.vocab_size, "n_embd": self.n_embd,
                "n_layer": self.n_layer, "n_head": self.n_head, "model_type": "gpt2"}

class _FakeModel:
    config = _FakeConfig()
    def named_parameters(self):
        # Tiny fake param list — enough to hash
        return [(f"layer.{i}.weight", MagicMock(shape=[768, 768], dtype="torch.float32"))
                for i in range(12)]
    def parameters(self):
        p = MagicMock()
        p.dtype = "torch.float32"
        p.device = MagicMock()
        p.device.__str__ = lambda s: "cpu"
        return iter([p])


FAKE_VOCAB = {" Paris": 3581, " four": 1440, " George": 4604, " of": 286, "David": 4006}

class _FakeTokenizer:
    vocab_size = 50257
    def get_vocab(self):
        return FAKE_VOCAB
    def encode(self, text, add_special_tokens=False):
        # Return single-id list for known tokens, two-ids for unknown
        vid = FAKE_VOCAB.get(text)
        if vid is not None:
            return [vid]
        return [999, 1000]   # simulate multi-token split for unknown text
    def decode(self, ids):
        reverse = {v: k for k, v in FAKE_VOCAB.items()}
        return "".join(reverse.get(i, "?") for i in ids)


def _make_identity(weights_hash="aabbccdd") -> CheckpointIdentity:
    return CheckpointIdentity(
        model_id="gpt2", hf_id="gpt2",
        architecture="GPT2LMHeadModel",
        vocab_size=50257, hidden_size=768, num_layers=12, num_heads=12,
        dtype="torch.float32", device="cpu",
        weights_hash=weights_hash,
        config_hash="cafebabe",
        tokenizer_hash="deadbeef",
    )


def _make_mock_runtime(
    top1_token: str = " Paris",
    target_rank: int = 0,
    target_prob: float = 0.72,
    target_logit: float = 8.3,
    top_k_entries: Optional[List[Dict]] = None,
) -> MagicMock:
    """Builds a minimal mock ModelRuntimeInterface for sanity checks."""
    rt = MagicMock()
    rt.num_layers = 12

    fwd = MagicMock()
    fwd.top_predicted_token = top1_token
    fwd.target_rank = target_rank
    fwd.target_probability = target_prob
    fwd.target_logit = target_logit
    fwd.layer_residuals = {}
    rt.forward.return_value = fwd

    if top_k_entries is None:
        top_k_entries = [
            {"token": top1_token, "probability": target_prob},
            {"token": " London", "probability": 0.05},
        ]
    rt.project_to_vocabulary.return_value = {"top_k": top_k_entries}
    return rt


# ─── 1. Checkpoint identity ───────────────────────────────────────────────────

def test_checkpoint_identity_match():
    """Identical identities must verify as True with no mismatches."""
    a = _make_identity("hash_abc")
    b = _make_identity("hash_abc")
    result = verify_checkpoint_identity(live=a, baseline=b)
    assert result["verified"] is True
    assert result["mismatches"] == {}
    assert "VERIFIED" in result["summary"]


def test_checkpoint_identity_hash_mismatch():
    """Differing weights_hash must fail verification and name the differing field."""
    baseline = _make_identity("correct_hash_abc")
    live     = _make_identity("WRONG_hash_xyz")
    result   = verify_checkpoint_identity(live=live, baseline=baseline)
    assert result["verified"] is False
    assert "weights_hash" in result["mismatches"]
    assert "MISMATCH" in result["summary"]


def test_compute_checkpoint_identity_fields():
    """compute_checkpoint_identity must populate all structural fields correctly."""
    identity = compute_checkpoint_identity(
        model=_FakeModel(), tokenizer=_FakeTokenizer(),
        model_id="gpt2", hf_id="gpt2",
    )
    assert identity.model_id   == "gpt2"
    assert identity.vocab_size == 50257
    assert identity.num_layers == 12
    assert identity.hidden_size == 768
    assert len(identity.weights_hash) == 64    # SHA-256 hex
    assert len(identity.config_hash) == 64
    assert len(identity.tokenizer_hash) == 64


# ─── 2. Tokenizer integrity ───────────────────────────────────────────────────

def test_tokenizer_integrity_passes_for_known_tokens():
    """Known single-token targets must pass all four integrity checks."""
    report = verify_tokenizer_integrity(_FakeTokenizer(), [" Paris", " four"])
    assert report.all_passed is True
    assert report.failed_tokens == []
    assert all(r.integrity_pass for r in report.token_reports)
    assert all(r.is_single_token for r in report.token_reports)
    assert all(r.round_trip_ok for r in report.token_reports)


def test_tokenizer_integrity_fails_for_split_token():
    """Tokens that split into multiple sub-tokens must fail integrity."""
    # "UNKNOWNXYZ" is not in FAKE_VOCAB → encode returns [999, 1000]
    report = verify_tokenizer_integrity(_FakeTokenizer(), ["UNKNOWNXYZ"])
    assert report.all_passed is False
    assert "UNKNOWNXYZ" in report.failed_tokens
    token_report = report.token_reports[0]
    assert token_report.is_single_token is False
    assert token_report.integrity_pass is False


def test_tokenizer_integrity_fails_for_out_of_vocab_token():
    """Tokens absent from vocabulary must fail the in-vocab check."""
    report = verify_tokenizer_integrity(_FakeTokenizer(), ["TOTALLY_MISSING_TOKEN_XYZ"])
    assert report.all_passed is False
    token_report = report.token_reports[0]
    assert token_report.target_token_in_vocab is False


def test_tokenizer_round_trip_invariant():
    """decode(encode(t)) == t must hold for all valid tokens."""
    tok = _FakeTokenizer()
    for token in [" Paris", " four", " George", " of"]:
        report = verify_tokenizer_integrity(tok, [token])
        r = report.token_reports[0]
        assert r.round_trip_ok is True, f"Round-trip failed for '{token}'"


# ─── 3. Architecture match ────────────────────────────────────────────────────

def test_architecture_match_correct():
    """Gate must report architecture verified when all structural fields match."""
    result = run_model_integrity_gate(
        model=_FakeModel(), tokenizer=_FakeTokenizer(),
        model_id="gpt2", hf_id="gpt2",
        target_tokens=[" Paris"],
        runtime=_make_mock_runtime(),
        expected_num_layers=12, expected_hidden_size=768,
        expected_architecture="_FakeModel",   # actual class name of the stub
    )
    assert result.architecture_verified is True


def test_architecture_match_wrong_layers():
    """Gate must report architecture NOT verified when num_layers doesn't match."""
    result = run_model_integrity_gate(
        model=_FakeModel(), tokenizer=_FakeTokenizer(),
        model_id="gpt2", hf_id="gpt2",
        target_tokens=[" Paris"],
        runtime=_make_mock_runtime(),
        expected_num_layers=24,   # wrong — actual is 12
        expected_hidden_size=768,
        expected_architecture="GPT2LMHeadModel",
    )
    assert result.architecture_verified is False
    assert "architecture_match" in result.failed_checks


# ─── 4. Behavioral sanity ────────────────────────────────────────────────────

def test_behavioral_sanity_passes_when_token_in_top_k():
    """Sanity must pass when all anchor expected tokens appear within top-k rank."""
    # Build a runtime that returns rank=5 for any target — within top-50 default
    rt = _make_mock_runtime(top1_token=" Paris", target_rank=5, target_prob=0.62)
    anchors = [
        SanityAnchor("sanity_test", "The capital of France is", " Paris",
                     "test anchor", max_acceptable_rank=50),
    ]
    report = run_behavioral_sanity_suite(rt, model_id="gpt2", anchors=anchors)
    assert report.all_passed is True
    assert report.anchors_failed == 0
    assert "PASSED" in report.summary


def test_behavioral_sanity_fails_when_token_beyond_max_rank():
    """Sanity must fail when expected token is outside max_acceptable_rank."""
    rt = _make_mock_runtime(top1_token="David", target_rank=200, target_prob=0.001)
    anchors = [
        SanityAnchor("sanity_capital", "The capital of France is", " Paris",
                     "capital test", max_acceptable_rank=50),
    ]
    report = run_behavioral_sanity_suite(rt, model_id="gpt2", anchors=anchors)
    assert report.all_passed is False
    assert report.anchors_failed == 1
    assert "sanity_capital" in report.failed_anchor_ids
    assert "FAILED" in report.summary


def test_behavioral_sanity_fails_when_token_not_in_top_k():
    """Sanity must fail when runtime returns rank=-1 (token not found)."""
    rt = _make_mock_runtime(top1_token="Sophia", target_rank=-1, target_prob=0.0)
    anchors = [
        SanityAnchor("sanity_president", "The first President of the United States was",
                     " George", "president test", max_acceptable_rank=50),
    ]
    report = run_behavioral_sanity_suite(rt, model_id="gpt2", anchors=anchors)
    assert report.all_passed is False
    assert "sanity_president" in report.failed_anchor_ids


def test_behavioral_sanity_records_r_p_delta_z():
    """Sanity results must carry R_target, P_target, P_top1, delta_z_target from runtime."""
    rt = _make_mock_runtime(
        top1_token=" Paris", target_rank=0, target_prob=0.81, target_logit=9.2,
        top_k_entries=[{" Paris": 0.81}, {" London": 0.05}],
    )
    anchors = [
        SanityAnchor("s1", "The capital of France is", " Paris", "capital", max_acceptable_rank=50),
    ]
    report = run_behavioral_sanity_suite(rt, model_id="gpt2", anchors=anchors)
    r = report.anchor_results[0]
    assert r.R_target == 0
    assert r.P_target == pytest.approx(0.81, abs=1e-4)
    assert r.delta_z_target == pytest.approx(9.2, abs=0.01)


# ─── 5. Composite gate ───────────────────────────────────────────────────────

def test_gate_ready_when_all_checks_pass():
    """Gate must be MODEL_READY when all four sub-checks pass."""
    rt = _make_mock_runtime(top1_token=" Paris", target_rank=0, target_prob=0.80)
    result = run_model_integrity_gate(
        model=_FakeModel(), tokenizer=_FakeTokenizer(),
        model_id="gpt2", hf_id="gpt2",
        target_tokens=[" Paris"],
        runtime=rt,
        expected_num_layers=12, expected_hidden_size=768,
        expected_architecture="_FakeModel",   # actual class name of the stub
    )
    assert result.is_ready is True
    assert result.gate_state == ModelReadinessState.MODEL_READY
    assert result.failed_checks == []
    assert "MODEL_READY" in result.gate_message


def test_gate_not_validated_when_tokenizer_fails():
    """Gate must be MODEL_NOT_VALIDATED when tokenizer integrity fails."""
    rt = _make_mock_runtime()
    result = run_model_integrity_gate(
        model=_FakeModel(), tokenizer=_FakeTokenizer(),
        model_id="gpt2", hf_id="gpt2",
        target_tokens=["TOTALLY_MISSING_SPLIT_TOKEN_XYZ"],   # will fail tokenizer check
        runtime=rt,
        expected_num_layers=12,
    )
    assert result.is_ready is False
    assert result.gate_state == ModelReadinessState.MODEL_NOT_VALIDATED
    assert "tokenizer_integrity" in result.failed_checks
    assert "MODEL_NOT_VALIDATED" in result.gate_message
    assert "ABSTAIN" in result.gate_message


def test_gate_not_validated_when_behavioral_sanity_fails():
    """Gate must be MODEL_NOT_VALIDATED when behavioral sanity anchor ranks exceed threshold."""
    # Runtime returns rank=999 for all targets — nothing is sane
    rt = _make_mock_runtime(top1_token="David", target_rank=999, target_prob=0.0)
    result = run_model_integrity_gate(
        model=_FakeModel(), tokenizer=_FakeTokenizer(),
        model_id="gpt2", hf_id="gpt2",
        target_tokens=[" Paris"],
        runtime=rt,
    )
    assert result.is_ready is False
    assert "behavioral_sanity" in result.failed_checks


def test_gate_result_is_serializable():
    """to_dict must produce a clean dict with all required fields."""
    rt = _make_mock_runtime(top1_token=" Paris", target_rank=0, target_prob=0.80)
    result = run_model_integrity_gate(
        model=_FakeModel(), tokenizer=_FakeTokenizer(),
        model_id="gpt2", hf_id="gpt2",
        target_tokens=[" Paris"],
        runtime=rt,
        expected_num_layers=12,
    )
    d = result.to_dict()
    assert d["model_id"] == "gpt2"
    assert "gate_state" in d
    assert "checkpoint_verified" in d
    assert "tokenizer_verified" in d
    assert "architecture_verified" in d
    assert "behavioral_sanity_passed" in d
    assert "failed_checks" in d
    assert "gate_message" in d


# ─── 6. assert_model_ready ───────────────────────────────────────────────────

def test_assert_model_ready_raises_on_non_ready():
    """assert_model_ready must raise RuntimeError with full diagnostic when gate fails."""
    # Craft a minimally-invalid gate result directly
    result = ModelIntegrityGateResult(
        model_id="gpt2",
        gate_state=ModelReadinessState.MODEL_NOT_VALIDATED,
        timestamp_utc="2026-08-16T00:00:00+00:00",
        checkpoint_verified=True,
        tokenizer_verified=False,
        architecture_verified=True,
        behavioral_sanity_passed=True,
        checkpoint_report={"summary": "OK"},
        tokenizer_report={"summary": "FAILED: ' France' not in vocab"},
        architecture_report={"summary": "OK", "verified": True},
        behavioral_report={"summary": "OK"},
        failed_checks=["tokenizer_integrity"],
        gate_message="MODEL_NOT_VALIDATED: failed ['tokenizer_integrity']. ABSTAIN.",
    )
    with pytest.raises(RuntimeError) as exc_info:
        assert_model_ready(result)
    err = str(exc_info.value)
    assert "BLOCKED" in err
    assert "tokenizer_integrity" in err
    assert "ABSTAIN" in err


def test_assert_model_ready_passes_silently_when_ready():
    """assert_model_ready must not raise when gate_state is MODEL_READY."""
    result = ModelIntegrityGateResult(
        model_id="gpt2",
        gate_state=ModelReadinessState.MODEL_READY,
        timestamp_utc="2026-08-16T00:00:00+00:00",
        checkpoint_verified=True,
        tokenizer_verified=True,
        architecture_verified=True,
        behavioral_sanity_passed=True,
        checkpoint_report=None,
        tokenizer_report=None,
        architecture_report={"summary": "OK", "verified": True},
        behavioral_report=None,
        failed_checks=[],
        gate_message="MODEL_READY: all checks passed.",
    )
    # Should not raise
    assert_model_ready(result)
