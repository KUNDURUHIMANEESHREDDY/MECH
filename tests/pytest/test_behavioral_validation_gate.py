"""Tests for BehavioralValidation gate and its integration into zoo_probe_suite.

Validates the key invariant:

    execution_pass = True  AND  behavioral_pass = True
    ─────────────────────────────────────────────────
    required BEFORE any probe result enters the causal
    discovery pipeline.
"""

from __future__ import annotations

import pytest

from backend.runtime.behavioral_validation import (
    BehavioralGateDecision,
    BehavioralValidation,
    assert_promotion_gate,
    build_behavioral_validation,
    check_promotion_gate,
)
from backend.runtime.zoo_probe_suite import StandardProbeExecutionResult


# ─── helpers ─────────────────────────────────────────────────────────────────

def _make_validation(
    expected_token: str,
    observed_token: str,
    expected_rank: int = 1,
    expected_probability: float = 0.62,
    top_k=None,
    execution_pass: bool = True,
) -> BehavioralValidation:
    """Build a BehavioralValidation via the public constructor."""
    if top_k is None:
        top_k = [
            (observed_token, 0.70),
            (expected_token, expected_probability),
            (" London", 0.04),
        ]
    return build_behavioral_validation(
        probe_id="probe_test",
        expected_token=expected_token,
        observed_token=observed_token,
        expected_rank=expected_rank,
        expected_probability=expected_probability,
        top_k=top_k,
        execution_pass=execution_pass,
    )


# ─── test 1: gate PROMOTES when model produces the expected token ─────────────

def test_behavioral_gate_promotes_on_correct_token():
    """When the model's top-1 == expected_token the gate must PROMOTE."""
    val = _make_validation(expected_token=" Paris", observed_token=" Paris")

    assert val.execution_pass is True
    assert val.behavioral_pass is True
    assert val.gate_decision == BehavioralGateDecision.PROMOTE
    assert val.block_reason == ""
    assert check_promotion_gate(val) is True


# ─── test 2: gate BLOCKS when model produces a different token ────────────────

def test_behavioral_gate_blocks_on_wrong_token():
    """When observed != expected the gate must BLOCK and surface diagnostics.

    This is the case illustrated in the user report:
        probe_capital_france  expected=" France"  observed="David"
    The execution itself is fine, but the model is not producing the
    expected token, so mechanistic investigation would explain the
    wrong behavior.
    """
    val = _make_validation(
        expected_token=" France",
        observed_token="David",
        expected_rank=47,
        expected_probability=0.003,
        top_k=[
            ("David",  0.38),
            ("Bob",    0.21),
            ("Mary",   0.15),
            (" France", 0.003),
        ],
    )

    assert val.execution_pass is True
    assert val.behavioral_pass is False
    assert val.gate_decision == BehavioralGateDecision.BLOCK
    assert check_promotion_gate(val) is False

    # The block_reason must name both the observed and expected tokens
    assert "David" in val.block_reason
    assert " France" in val.block_reason

    # assert_promotion_gate must raise with a diagnostic message
    with pytest.raises(ValueError) as exc_info:
        assert_promotion_gate(val)
    err = str(exc_info.value)
    assert "BLOCKED" in err
    assert "David" in err
    assert " France" in err


# ─── test 3: gate sets ERROR when execution itself failed ────────────────────

def test_behavioral_gate_errors_on_execution_failure():
    """When execution_pass=False the gate must be ERROR regardless of tokens."""
    val = _make_validation(
        expected_token=" Paris",
        observed_token="",
        execution_pass=False,
    )

    assert val.execution_pass is False
    assert val.behavioral_pass is False
    assert val.gate_decision == BehavioralGateDecision.ERROR
    assert check_promotion_gate(val) is False

    with pytest.raises(ValueError) as exc_info:
        assert_promotion_gate(val)
    assert "ERROR" in str(exc_info.value)


# ─── test 4: BehavioralValidation round-trips through to_dict ────────────────

def test_behavioral_validation_serialization():
    """to_dict must expose all fields needed for logging and downstream ingestion."""
    top_k = [(" Paris", 0.70), (" Lyon", 0.12), (" Nice", 0.08)]
    val = _make_validation(
        expected_token=" Paris",
        observed_token=" Paris",
        expected_rank=0,
        expected_probability=0.70,
        top_k=top_k,
    )

    d = val.to_dict()

    assert d["probe_id"] == "probe_test"
    assert d["expected_token"] == " Paris"
    assert d["observed_token"] == " Paris"
    assert d["expected_rank"] == 0
    assert d["behavioral_pass"] is True
    assert d["execution_pass"] is True
    assert d["gate_decision"] == "PROMOTE"
    assert isinstance(d["top_k"], list)
    assert len(d["top_k"]) == 3
    assert d["top_k"][0]["token"] == " Paris"
    assert d["top_k"][0]["probability"] == pytest.approx(0.70, abs=1e-4)


# ─── test 5: StandardProbeExecutionResult exposes convenience properties ─────

def test_probe_result_is_promotable_property():
    """is_promotable must be True iff both execution_pass AND behavioral_pass are True."""
    val_promote = _make_validation(" Paris", " Paris")
    val_block   = _make_validation(" Paris", "David")
    val_error   = _make_validation(" Paris", "", execution_pass=False)

    def _make_result(v: BehavioralValidation) -> StandardProbeExecutionResult:
        return StandardProbeExecutionResult(
            probe_id="probe_test",
            model_id="test-model",
            architecture="gpt2",
            precision="float32",
            runtime_type="in_memory",
            clean_target_logit=5.2,
            clean_target_probability=v.expected_probability,
            clean_target_rank=v.expected_rank,
            corrupted_target_logit=1.1,
            corrupted_target_rank=40,
            intervene_layer=8,
            causal_delta_z=-3.1,
            indirect_effect=0.82,
            mediation_rescue_fraction=0.91,
            logit_lens_trajectory=[0.1, 0.5, 1.2, 3.8, 5.2],
            execution_duration_ms=120.4,
            behavioral_validation=v,
        )

    assert _make_result(val_promote).is_promotable is True
    assert _make_result(val_block).is_promotable is False
    assert _make_result(val_error).is_promotable is False


# ─── test 6: realistic IOI-style BLOCK scenario ──────────────────────────────

def test_ioi_behavioral_block_scenario():
    """Simulates the IOI probe case where expected='Mary' but model returns 'Bob'.

    The probe *executed* successfully.  But behavioral_pass is False.
    MECH must not attempt to explain the IOI circuit under these conditions —
    it would be explaining 'Bob' completion, not the intended Mary/John behaviour.
    """
    val = build_behavioral_validation(
        probe_id="probe_ioi",
        expected_token="Mary",
        observed_token="Bob",
        expected_rank=6,
        expected_probability=0.031,
        top_k=[
            ("Bob",   0.44),
            ("John",  0.22),
            ("David", 0.10),
            ("Emma",  0.09),
            ("Mary",  0.031),
        ],
        execution_pass=True,
    )

    assert val.behavioral_pass is False
    assert val.gate_decision == BehavioralGateDecision.BLOCK
    assert "Bob" in val.block_reason
    assert "Mary" in val.block_reason

    with pytest.raises(ValueError) as exc_info:
        assert_promotion_gate(val)
    err = str(exc_info.value)
    # The error message must surface the full diagnostic
    assert "probe_ioi" in err
    assert "Bob" in err
    assert "Mary" in err


# ─── test 7: induction-head BLOCK scenario ───────────────────────────────────

def test_induction_head_behavioral_block_scenario():
    """Simulates the induction-head probe where expected='d' but model returns 'Emma'."""
    val = build_behavioral_validation(
        probe_id="probe_induction_head",
        expected_token="d",
        observed_token="Emma",
        expected_rank=19,
        expected_probability=0.008,
        top_k=[
            ("Emma",   0.51),
            ("Anna",   0.19),
            ("Bob",    0.11),
            ("d",      0.008),
        ],
        execution_pass=True,
    )

    assert val.execution_pass is True
    assert val.behavioral_pass is False
    assert val.gate_decision == BehavioralGateDecision.BLOCK
    assert check_promotion_gate(val) is False
