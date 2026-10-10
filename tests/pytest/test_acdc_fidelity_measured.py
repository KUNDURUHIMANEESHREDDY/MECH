"""ACDC whole-circuit fidelity is measured on the standard path.

The gap-based `circuit_fidelity` needs io_id/subject_id, which the bundled
datasets omit -- so fidelity was always unmeasured on a standard run. The
target-logit variant needs only the target token the sweep already scored,
so a live run now reports a measured recovery fraction under its own metric
name. The two metrics are never substituted: the record says which one ran.
"""

from __future__ import annotations

import pytest

from _weight_guard import skip_reason, weights_available

needs_weights = pytest.mark.skipif(
    not weights_available(), reason=skip_reason()
)


def _canonical_ioi_pair():
    """Return (subject, io) from the API's deterministic default names.

    The API at /api/gpt2/ioi defaults to (NAMES[0], NAMES[1]) when the caller
    omits names. This reads the same source so tests and API stay in sync.
    """
    from backend.api.dispatcher import NAMES

    return NAMES[0], NAMES[1]


def _clean_corrupted_prompts(subject: str, io_name: str) -> tuple[str, str]:
    """Build the canonical clean/corrupted IOI prompts via the live measurement layer.

    Uses the same prompt templates that `circuit_fidelity_target` and the
    ACDC algorithm rely on, so the test exercises the exact code path the
    platform uses for IOI.
    """
    from backend.interpretability.discovery.live_measure import (
        clean_prompt,
        corrupted_prompt,
    )
    return clean_prompt(subject, io_name), corrupted_prompt(subject, io_name)


def _io_target_token_id(io_name: str) -> int | None:
    """Resolve the IO name's single leading-space token id via the live tokenizer.

    Returns None if the name is not a single token -- callers must fail closed
    rather than guessing token splits.
    """
    from backend.interpretability.discovery.live_measure import single_token_names

    return single_token_names([io_name])[io_name]


# Canonical IOI pair (subject, io) and derived prompts/token for live tests.
_SUBJECT, _IO = _canonical_ioi_pair()
_CLEAN, _CORRUPTED = _clean_corrupted_prompts(_SUBJECT, _IO)
_TARGET_ID = _io_target_token_id(_IO)


def test_target_fidelity_without_a_circuit_is_undefined():
    """No retained heads means nothing to inject -- no weights needed."""
    from backend.interpretability.discovery.live_measure import (
        circuit_fidelity_target,
    )

    # Use a nonsense target_id; the empty circuit triggers the unmeasured path
    # before any forward pass, so the value is irrelevant.
    out = circuit_fidelity_target(_CLEAN, _CORRUPTED, 1234, set())
    assert out["measured"] is False
    assert out["fidelity"] is None
    assert out["reason"]


@needs_weights
def test_target_fidelity_without_a_gap_is_undefined():
    """Identical prompts give the target identical logits: no gap to explain."""
    from backend.interpretability.discovery.live_measure import (
        circuit_fidelity_target,
    )

    assert _TARGET_ID is not None, "IO name must tokenize to a single token"
    out = circuit_fidelity_target(_CLEAN, _CLEAN, _TARGET_ID, {(9, 9)})
    assert out["measured"] is False
    assert out["fidelity"] is None


@needs_weights
def test_target_fidelity_measures_a_single_head_injection():
    from backend.interpretability.discovery.live_measure import (
        circuit_fidelity_target,
    )

    assert _TARGET_ID is not None, "IO name must tokenize to a single token"
    out = circuit_fidelity_target(_CLEAN, _CORRUPTED, _TARGET_ID, {(9, 9)})
    assert out["measured"] is True
    assert out["metric"] == "target-logit recovery"
    assert isinstance(out["fidelity"], float)
    # Internal consistency: the fraction recomputed from the recorded
    # logits must equal the reported fidelity.
    recomputed = ((out["recovered_target_logit"] - out["corrupted_target_logit"])
                  / (out["clean_target_logit"] - out["corrupted_target_logit"]))
    assert recomputed == pytest.approx(out["fidelity"], abs=1e-3)
    assert out["retained_heads"] == ["L9H9"]


def test_mock_run_reports_fidelity_unmeasured_under_its_own_metric():
    from backend.science.models.gpt2_adapter import GPT2Adapter
    from backend.interpretability.discovery.algorithms import get_algorithm

    engine = get_algorithm("acdc", GPT2Adapter(variant="small", mock_mode=True))
    result = engine.run(dataset={
        "id": "ioi_0001", "clean": _CLEAN, "corrupted": _CORRUPTED,
        "target_token": f" {_IO}",
    }).to_dict()

    stats = result["statistics"]
    assert stats["logit_recovery_fidelity"] is None
    assert stats["logit_recovery_fidelity_measured"] is False
    assert stats["logit_recovery_fidelity_metric"] == "unmeasured"


@needs_weights
def test_live_run_reports_measured_target_logit_fidelity():
    from backend.science.models.gpt2_adapter import GPT2Adapter
    from backend.interpretability.discovery.algorithms import get_algorithm

    engine = get_algorithm(
        "acdc", GPT2Adapter(variant="small", mock_mode=False))
    result = engine.run(dataset={
        "id": "ioi_0001", "clean": _CLEAN, "corrupted": _CORRUPTED,
        "target_token": f" {_IO}",
    }).to_dict()

    stats = result["statistics"]
    assert stats["total_evaluations"] > 0
    assert stats["logit_recovery_fidelity_measured"] is True, (
        f"fidelity unmeasured: {stats['logit_recovery_fidelity_detail']}")
    assert stats["logit_recovery_fidelity_metric"] == "target-logit recovery"
    assert isinstance(stats["logit_recovery_fidelity"], float)
    detail = stats["logit_recovery_fidelity_detail"]
    recomputed = ((detail["recovered_target_logit"] - detail["corrupted_target_logit"])
                  / (detail["clean_target_logit"] - detail["corrupted_target_logit"]))
    assert recomputed == pytest.approx(stats["logit_recovery_fidelity"], abs=1e-3)