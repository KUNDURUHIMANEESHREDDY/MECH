"""Statistical confidence and peer-review transparency for live validation.

Regression tests for the finding that LiveValidationExecutor computed
confidence from three pass rates over three held-out prompts without
reporting sample sizes, uncertainty, per-prompt results, or the scope
of the claim — and labelled its rule-based verdict `peer_review`
with decision "Accept", which downstream readers can mistake for
human peer review.

What this pins
--------------
* the validation payload reports sample sizes (held-out prompts, heads);
* it reports uncertainty (Wilson 95% intervals for the head-level rates,
  range/sd for recovery);
* it reports per-prompt recovery values;
* it states its scope and limitations (single model, deterministic
  forward passes, small N — not broad replication);
* the automated rule check is explicitly tagged as not human peer
  review (`kind`, `is_human_peer_review: False`);
* a distinct `human_peer_review` slot exists and is empty by default,
  so publication interfaces can tell the two apart.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.validation.live_validation import (  # noqa: E402
    LiveValidationExecutor,
    _wilson_interval,
)


# ── pure helpers (no weights needed) ──────────────────────────────────────

def test_wilson_interval_covers_small_n_honestly():
    """With 3/3 heads replicating, the point estimate is 1.0 but the
    95% interval must stay wide — a deterministic threshold pass on
    three prompts is weak evidence of generalization."""
    interval = _wilson_interval(3, 3)
    assert interval["lo"] < 0.65, interval
    assert interval["hi"] == pytest.approx(1.0)
    assert interval["n"] == 3


def test_wilson_interval_marks_empty_denominator():
    interval = _wilson_interval(0, 0)
    assert interval["lo"] == 0.0
    assert interval["hi"] == 1.0


def test_wilson_interval_is_monotone_in_evidence():
    narrow = _wilson_interval(30, 30)
    wide = _wilson_interval(3, 3)
    assert (narrow["hi"] - narrow["lo"]) < (wide["hi"] - wide["lo"])


# ── payload structure (needs live weights) ────────────────────────────────

def _needs_weights():
    from backend.interpretability.discovery.live_discovery import (
        LiveIOIDiscovery,
    )
    return LiveIOIDiscovery.available()


def _run_validation():
    from backend.interpretability.discovery.live_discovery import (
        LiveIOIDiscovery,
        MIN_PROMPTS_FOR_VALIDATION,
        MIN_PROMPTS_FOR_INTERACTION,
    )
    discovery = LiveIOIDiscovery().run(
        hypothesis_statement="Identify the minimal set of heads sufficient for IOI",
        n_prompts=MIN_PROMPTS_FOR_VALIDATION,
        n_interaction_prompts=MIN_PROMPTS_FOR_INTERACTION,
    )
    assert discovery.get("status") == "completed"
    result = LiveValidationExecutor().validate(discovery)
    assert result.get("status") == "completed", result.get("reason")
    return result


@pytest.mark.skipif(not _needs_weights(), reason="live GPT-2 weights unavailable")
def test_validation_reports_sample_sizes():
    result = _run_validation()
    assert result["n_held_out_prompts"] == len(result["validation_panel"])
    assert result["sample_sizes"]["held_out_prompts"] == result["n_held_out_prompts"]
    assert result["sample_sizes"]["heads"] == len(result["head_checks"])
    assert result["confidence"]["n_held_out_prompts"] == result["n_held_out_prompts"]


@pytest.mark.skipif(not _needs_weights(), reason="live GPT-2 weights unavailable")
def test_validation_reports_uncertainty():
    result = _run_validation()
    uncertainty = result["confidence"]["uncertainty"]
    assert "replication_rate_wilson95" in uncertainty
    assert "minimality_wilson95" in uncertainty
    lo, hi = uncertainty["replication_rate_wilson95"]
    assert 0.0 <= lo <= hi <= 1.0
    assert "recovery_range" in uncertainty
    assert "recovery_sd" in uncertainty


@pytest.mark.skipif(not _needs_weights(), reason="live GPT-2 weights unavailable")
def test_validation_reports_per_prompt_recovery():
    result = _run_validation()
    per_prompt = result["per_prompt_recovery"]
    assert len(per_prompt) == result["n_held_out_prompts"]
    for entry in per_prompt:
        assert {"subject", "io", "pool_index", "recovery"} <= set(entry)


@pytest.mark.skipif(not _needs_weights(), reason="live GPT-2 weights unavailable")
def test_validation_states_scope_and_limitations():
    result = _run_validation()
    assert "gpt2-small" in result["scope"]
    assert str(result["n_held_out_prompts"]) in result["scope"]
    assert isinstance(result["limitations"], list) and result["limitations"]
    joined = " ".join(result["limitations"]).lower()
    assert "human" in joined or "peer" in joined
    assert "seed" in joined or "deterministic" in joined


@pytest.mark.skipif(not _needs_weights(), reason="live GPT-2 weights unavailable")
def test_automated_check_is_not_human_peer_review():
    result = _run_validation()
    review = result["peer_review"]
    assert review.get("kind") == "automated-rule-check"
    assert review.get("is_human_peer_review") is False
    # A distinct human-review slot exists and is empty by default.
    assert "human_peer_review" in result
    assert result["human_peer_review"] is None


@pytest.mark.skipif(not _needs_weights(), reason="live GPT-2 weights unavailable")
def test_confidence_is_flagged_heuristic():
    result = _run_validation()
    assert result["confidence"].get("is_heuristic") is True


def test_human_peer_acceptance_helper_defaults_false():
    """Publication interfaces must be able to tell automated acceptance
    apart from human acceptance."""
    from backend.agents.critic import Critic

    assert Critic.has_human_peer_acceptance({}) is False
    assert Critic.has_human_peer_acceptance(
        {"human_peer_review": None}) is False
    assert Critic.has_human_peer_acceptance(
        {"human_peer_review": {"decision": "Accept"}}) is True
    assert Critic.has_human_peer_acceptance(
        {"human_peer_review": {"decision": "Revise"}}) is False
    # An automated Accept alone is not human acceptance.
    assert Critic.has_human_peer_acceptance(
        {"peer_review": {"decision": "Accept",
                         "is_human_peer_review": False}}) is False
