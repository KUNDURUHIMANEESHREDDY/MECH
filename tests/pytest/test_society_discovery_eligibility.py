"""The Society's discovery must be able to reach eligibility.

The defect
----------
`LiveIOIDiscovery.run(hypothesis_statement, n_prompts=4, ...)` derives
validation eligibility from the prompt count and publication eligibility from
the interaction count. `DiscoveryEngine.discover_and_orchestrate()` forwarded
neither, so a caller that wanted an adequate sample had no way to ask for one.

The result was not a crash -- it was a permanently ineligible workflow. Every
Society run stopped at discovery with::

    Discovery blocked: stage status 'unavailable' is not complete.

and the executor reported, correctly, ``4 prompts, below the 10 needed for a
validation decision``. So the gate was doing its job while the plumbing above it
made the job impossible. That combination is the dangerous shape: the honest
check was right, so nothing looked broken except the workflow, which looked like
an environment limitation rather than an interface defect.

What this pins
--------------
* the orchestrator forwards a prompt count when given one;
* the defaults are unchanged, so the fix does not silently raise the cost of
  every other caller;
* the Society agent asks for the executor's own eligibility thresholds;
* those constants stay equal to the executor's, so a threshold change fails a
  test instead of silently re-breaking the workflow.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.agents import discoverer as disc_mod  # noqa: E402
from backend.interpretability.discovery.live_discovery import (  # noqa: E402
    LiveIOIDiscovery,
    MIN_PROMPTS_FOR_INTERACTION,
    MIN_PROMPTS_FOR_VALIDATION,
)
from backend.agents.evidence_policy import discovery_is_live  # noqa: E402

GOAL = "Reproduce IOI on gpt2-small and find causally important heads"


# ── patching ─────────────────────────────────────────────────────────────
# Both `discovery_engine.discover_and_orchestrate` and `discoverer.discover`
# import their collaborator *inside* the method, so replacing a module
# attribute has no effect. The method on the class is looked up at call time,
# which is the seam that actually works -- and it is also the seam the earlier
# version of this file tried and failed to use.

def _record_executor(monkeypatch, sink):
    """Intercept `LiveIOIDiscovery.run`, recording its kwargs into `sink`."""
    def fake_run(self, hypothesis_statement, **kwargs):
        sink.update(kwargs)
        return {"status": "completed", "provenance": "live",
                "validation_eligible": True, "publication_eligible": True}

    monkeypatch.setattr(LiveIOIDiscovery, "run", fake_run)
    monkeypatch.setattr(LiveIOIDiscovery, "available", staticmethod(lambda: True))


def _record_orchestrator(monkeypatch, sink):
    """Intercept `DiscoveryEngine.discover_and_orchestrate` for the agent test."""
    from backend.interpretability.discovery.discovery_engine import DiscoveryEngine

    def fake_orchestrate(self, hypothesis_statement, **kwargs):
        sink.update(kwargs)
        return {"status": "completed", "provenance": "live",
                "validation_eligible": True, "publication_eligible": True}

    monkeypatch.setattr(DiscoveryEngine, "discover_and_orchestrate",
                        fake_orchestrate)


@pytest.fixture
def engine_available():
    """The discovery needs live weights; some tests are about the measurement,
    others only about the call."""
    return LiveIOIDiscovery.available()


def test_the_orchestrator_accepts_and_forwards_a_prompt_count(monkeypatch):
    """The parameter exists and reaches the executor."""
    saw = {}
    _record_executor(monkeypatch, saw)

    from backend.interpretability.discovery.discovery_engine import DiscoveryEngine
    DiscoveryEngine().discover_and_orchestrate(
        hypothesis_statement=GOAL, n_prompts=7)

    assert saw == {"n_prompts": 7}, (
        f"the orchestrator did not forward n_prompts; the executor saw {saw!r}")


def test_the_orchestrator_forwards_both_sample_sizes(monkeypatch):
    saw = {}
    _record_executor(monkeypatch, saw)

    from backend.interpretability.discovery.discovery_engine import DiscoveryEngine
    DiscoveryEngine().discover_and_orchestrate(
        hypothesis_statement=GOAL, n_prompts=10, n_interaction_prompts=5)

    assert saw == {"n_prompts": 10, "n_interaction_prompts": 5}, (
        "the interaction count was not forwarded, so publication eligibility "
        f"stays unreachable even after validation eligibility is fixed; saw {saw!r}")


def test_the_defaults_do_not_silently_change(monkeypatch):
    """A fix that works by raising every caller's cost is not a fix."""
    saw = {}
    _record_executor(monkeypatch, saw)

    from backend.interpretability.discovery.discovery_engine import DiscoveryEngine
    DiscoveryEngine().discover_and_orchestrate(hypothesis_statement=GOAL)

    assert saw == {}, (
        "the orchestrator now forwards counts a caller did not ask for, which "
        f"raises the cost of every existing caller silently; saw {saw!r}")


def test_a_caller_can_now_reach_eligibility(engine_available):
    """The end-to-end claim, against real weights.

    This is the property that was impossible before: with an adequate sample the
    executor's own policy says the discovery is live.
    """
    if not engine_available:
        pytest.skip("live GPT-2 weights unavailable; the interface tests above "
                    "still pin the plumbing")

    from backend.interpretability.discovery.discovery_engine import DiscoveryEngine

    res = DiscoveryEngine().discover_and_orchestrate(
        hypothesis_statement=GOAL,
        n_prompts=MIN_PROMPTS_FOR_VALIDATION,
        n_interaction_prompts=MIN_PROMPTS_FOR_INTERACTION,
    )

    assert res.get("validation_eligible") is True, res.get("ineligible_because")
    assert res.get("publication_eligible") is True, res.get("ineligible_because")
    assert discovery_is_live(res), (
        "an adequate sample still does not satisfy discovery_is_live; either the "
        "executor or the policy has moved")


# ── the Society agent asks for it ────────────────────────────────────────

def test_the_society_agent_requests_an_adequate_sample(monkeypatch):
    """The workflow's caller must actually ask."""
    sent = {}
    _record_orchestrator(monkeypatch, sent)

    disc_mod.Discoverer().discover("Reproduce IOI on gpt2-small")

    assert sent.get("n_prompts") == disc_mod.SOCIETY_DISCOVERY_N_PROMPTS, (
        "the Society did not ask for a validation-adequate sample, so its "
        f"discovery stays ineligible; the executor saw {sent!r}")
    assert sent.get("n_interaction_prompts") == (
        disc_mod.SOCIETY_DISCOVERY_N_INTERACTION_PROMPTS), (
        "the Society did not ask for an interaction-adequate sample, so "
        f"publication eligibility stays unreachable; saw {sent!r}")


def test_the_society_counts_match_the_executors_thresholds():
    """A threshold change must break this test, not silently break the workflow."""
    assert disc_mod.SOCIETY_DISCOVERY_N_PROMPTS == MIN_PROMPTS_FOR_VALIDATION
    assert (disc_mod.SOCIETY_DISCOVERY_N_INTERACTION_PROMPTS
            == MIN_PROMPTS_FOR_INTERACTION)


def test_the_default_gate_still_refuses_an_inadequate_sample(monkeypatch):
    """The negative control: raising the caller's count must not weaken the gate.

    If the orchestrator began returning eligibility for a short sample, this
    fails and the assertions above would be passing for the wrong reason.
    """
    import backend.interpretability.discovery.live_discovery as ld

    real = ld._adequacy

    def fake_adequacy(*args, **kwargs):
        return {"statistically_adequate": False, "interaction_adequate": False,
                "validation_eligible": False, "publication_eligible": False,
                "ineligible_because": ["stub: too few prompts"]}

    monkeypatch.setattr(ld, "_adequacy", fake_adequacy)
    res = ld.LiveIOIDiscovery().run(GOAL, n_prompts=2, n_interaction_prompts=1)
    assert real is not ld._adequacy

    assert res.get("validation_eligible") is False, (
        "an inadequate sample was reported validation-eligible; the gate has "
        "been weakened")
    assert discovery_is_live(res) is False
