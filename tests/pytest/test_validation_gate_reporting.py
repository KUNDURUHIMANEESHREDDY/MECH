"""Every conjunct of the validation verdict must appear in ``checks``.

Regression test for a silent gate/evidence disagreement in
``LiveValidationExecutor``.

``validated`` is computed from four predicates::

    all(three named checks pass)  AND  confidence >= 0.85

but only the three named checks were placed in the ``checks`` array. The
confidence predicate -- which is the *binding* one at the published floors --
was never reported. So a payload could carry three passing checks and
``validated: False`` with nothing in ``checks`` naming the rejection::

    replication  0.75 >= 0.75   PASS
    recovery     0.50 >= 0.50   PASS
    minimality   0.75 >= 0.50   PASS
    confidence = 0.5 + 0.5 * mean([0.75, 0.50, 0.75]) = 0.8333 < 0.85

That is the exact state driven below. ``live_measure`` is stubbed with
deterministic values so this runs without GPT-2 weights, and the panel
arithmetic is asserted explicitly so the stub cannot silently drift away
from the floor values it claims to reproduce.

The invariant under test: if ``validated`` is False, at least one entry in
``checks`` must report ``passed: False``. A verdict has to be
reconstructible from the evidence printed beside it -- otherwise the
contract ``_fidelity_shortfall`` documents ("a reader is told which number
fell short rather than being told a field is missing") does not hold for
the one predicate that most often decides the outcome.

The fix reports the confidence predicate as a fourth entry in ``checks`` and
derives ``validated`` from that same list, so verdict and evidence share one
definition and cannot drift apart again.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from backend.interpretability.discovery.live_discovery import (
    MIN_PROMPTS_FOR_VALIDATION,
    prompt_panel,
)
from backend.validation import live_validation as lv

# ── The floor-exact scenario ─────────────────────────────────────────────
#
# Three heads replicate and stay individually necessary; one does neither.
# That yields replication 3/4 and minimality 3/4, i.e. both exactly at the
# boundary the checks compare against.

STRONG_HEADS: List[str] = ["L5H1", "L5H5", "L7H3"]
WEAK_HEAD = "L8H6"
HEADS: List[str] = STRONG_HEADS + [WEAK_HEAD]

CLEAN_DIFF = 2.0
CORRUPTED_DIFF = 0.0

#: (inject_diff - corrupted) / (clean - corrupted)
INJECT_DIFF = 1.0  # -> recovery exactly 0.50

#: replication floor is max(0.10, 0.5 * |mean_delta|) = 0.25 for mean_delta 0.5;
#: the necessity bar is 0.10 * |clean_diff| = 0.20.
ABLATE_STRONG = 0.60  # clears both
ABLATE_WEAK = 0.05  # clears neither


def _discovery() -> Dict[str, Any]:
    """A discovery whose panels are disjoint, so validate() is hermetic."""
    return {
        "discovery_id": "d_gate_reporting",
        "status": "completed",
        "provenance": "live",
        "attested": True,
        "heads": list(HEADS),
        "head_effects": [{"head": h, "mean_delta": 0.5} for h in HEADS],
        "n_prompts": MIN_PROMPTS_FOR_VALIDATION,
        "discovery_panel": prompt_panel(MIN_PROMPTS_FOR_VALIDATION, offset=0),
        "validation_panel": prompt_panel(3, offset=MIN_PROMPTS_FOR_VALIDATION),
    }


@pytest.fixture
def stub_measure(monkeypatch):
    """Install deterministic stand-ins for the forward-pass primitives."""

    def _install(
        strong_heads: List[str] = STRONG_HEADS,
        inject_diff: float = INJECT_DIFF,
    ) -> None:
        def clean_prompt(subject: str, io_name: str) -> str:
            return f"clean::{subject}::{io_name}"

        def corrupted_prompt(subject: str, io_name: str) -> str:
            return f"corrupted::{subject}::{io_name}"

        def baseline(prompt: str, io_id: int, subj_id: int) -> Dict[str, float]:
            if prompt.startswith("clean::"):
                return {"logit_diff": CLEAN_DIFF}
            return {"logit_diff": CORRUPTED_DIFF}

        def parse_head(label: str):
            return label if label in HEADS else None

        def ablate(prompt: str, io_id: int, subj_id: int, head_set: set) -> float:
            label = next(iter(head_set))
            delta = ABLATE_STRONG if label in strong_heads else ABLATE_WEAK
            return CLEAN_DIFF + delta

        def capture(prompt: str):
            return None, {}

        def inject(prompt: str, io_id: int, subj_id: int, caps, circuit) -> Dict[str, float]:
            return {"logit_diff": inject_diff}

        monkeypatch.setattr(lv.lm, "single_token_names", lambda names: {n: 1 for n in names})
        monkeypatch.setattr(lv.lm, "clean_prompt", clean_prompt)
        monkeypatch.setattr(lv.lm, "corrupted_prompt", corrupted_prompt)
        monkeypatch.setattr(lv.lm, "baseline", baseline)
        monkeypatch.setattr(lv.lm, "parse_head", parse_head)
        monkeypatch.setattr(lv.lm, "ablate", ablate)
        monkeypatch.setattr(lv.lm, "capture", capture)
        monkeypatch.setattr(lv.lm, "inject", inject)
        # Attestation reads the engine's weight fingerprint. Identity keeps the
        # payload inspectable; it is unrelated to the gate under test.
        monkeypatch.setattr(lv, "attest_measurement", lambda result, **kw: result)

    return _install


def test_stub_reproduces_the_floor_values_it_claims(stub_measure):
    """Guard the guard: the stubbed run must land where the docstring says.

    If this drifts, the regression below stops measuring the floor-exact
    case and starts measuring nothing in particular.
    """
    stub_measure()
    result = lv.LiveValidationExecutor().validate(_discovery())

    assert result["status"] == "completed", result.get("reason")
    assert result["replication_rate"] == 0.75
    assert result["held_out_recovery"] == 0.50
    assert result["minimality"] == 0.75
    assert result["confidence"]["confidence_score"] == 0.8333


def test_floors_pass_while_the_verdict_fails(stub_measure):
    """The scenario: every rate floor passes and confidence does not.

    This is the state that used to be reported as three green checks beside a
    rejecting verdict. The three rate checks still pass -- that is the point of
    the scenario -- and the confidence check is now what names the rejection.
    """
    stub_measure()
    result = lv.LiveValidationExecutor().validate(_discovery())

    by_name = {c["check"]: c for c in result["peer_review"]["checks"]}
    assert by_name["replication_rate>=0.75"]["passed"] is True
    assert by_name["held_out_recovery>=0.50"]["passed"] is True
    assert by_name["minimality>=0.50"]["passed"] is True
    assert by_name["confidence>=0.85"]["passed"] is False
    assert result["validated"] is False


def test_a_failing_verdict_names_a_failing_check(stub_measure):
    """THE REGRESSION.

    Before the fix this failed: ``validated`` was False while every entry in
    ``checks`` reported ``passed: True``, so the confidence shortfall that
    actually decided the verdict appeared nowhere in the payload.
    """
    stub_measure()
    result = lv.LiveValidationExecutor().validate(_discovery())

    assert result["validated"] is False
    failing = [c for c in result["peer_review"]["checks"] if not c["passed"]]
    assert failing, (
        "validated is False but every entry in `checks` reports passed=True, "
        "so the verdict cannot be reconstructed from the evidence beside it. "
        f"checks={result['peer_review']['checks']}"
    )


def test_a_passing_verdict_reports_no_failing_check(stub_measure):
    """The fix must not manufacture a failure on a genuinely passing run.

    All four heads replicate, stay necessary, and recovery reaches 1.0, so
    confidence is 1.0 and the verdict passes -- every check must agree.
    """
    stub_measure(strong_heads=list(HEADS), inject_diff=CLEAN_DIFF)
    result = lv.LiveValidationExecutor().validate(_discovery())

    assert result["confidence"]["confidence_score"] == 1.0
    assert result["validated"] is True
    failing = [c for c in result["peer_review"]["checks"] if not c["passed"]]
    assert not failing, failing
