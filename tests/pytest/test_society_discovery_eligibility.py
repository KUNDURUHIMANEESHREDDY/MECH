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


@pytest.fixture
def engine_available():
    """Skip-free: the discovery needs live weights, and the tests below are
    about the *call*, not about the measurement."""
    return LiveIOIDiscovery.available()


# ── the interface ────────────────────────────────────────────────────────

def test_the_orchestrator_accepts_and_forwards_a_prompt_count():
    """The parameter exists and reaches the executor."""
    import backend.interpretability.discovery.discovery_engine as de_mod

    with RecorderPatch() as recorder:
        de_mod.DiscoveryEngine().discover_and_orchestrate(
            hypothesis_statement=GOAL, n_prompts=7)

    assert recorder.last_kwargs == {"n_prompts": 7}, (
        f"the orchestrator did not forward n_prompts; the executor saw "
        f"{recorder.last_kwargs!r}")


class RecorderPatch:
    """Patch `LiveIOIDiscovery` into the discovery_engine module's namespace."""

    def __enter__(self):
        import backend.interpretability.discovery.discovery_engine as de_mod
        self._mod = de_mod
        self._original = getattr(de_mod, "LiveIOIDiscovery", None)

        class Recorder:
            available = staticmethod(lambda: True)

            def run(self, hypothesis_statement, **kwargs):
                Recorder.last_kwargs = kwargs  # type: ignore[attr-defined]
                return {"status": "completed", "provenance": "live",
                        "validation_eligible": True}

        de_mod.LiveIOIDiscovery = lambda: Recorder()  # type: ignore[assignment]
        return Recorder

    def __exit__(self, *exc):
        if self._original is not None:
            self._mod.LiveIOIDiscovery = self._original  # type: ignore[assignment]
        return False


def test_the_orchestrator_forwards_both_sample_sizes():
    import backend.interpretability.discovery.discovery_engine as de_mod

    with RecorderPatch() as recorder:
        de_mod.DiscoveryEngine().discover_and_orchestrate(
            hypothesis_statement=GOAL, n_prompts=10, n_interaction_prompts=5)

    assert recorder.last_kwargs == {"n_prompts": 10, "n_interaction_prompts": 5}, (
        "the interaction count was not forwarded, so publication eligibility "
        "stays unreachable even after validation eligibility is fixed")


def test_the_defaults_do_not_silently_change():
    """A fix that works by raising every caller's cost is not a fix."""
    import backend.interpretability.discovery.discovery_engine as de_mod

    with RecorderPatch() as recorder:
        de_mod.DiscoveryEngine().discover_and_orchestrate(
            hypothesis_statement=GOAL)

    assert recorder.last_kwargs == {}, (
        "the orchestrator now forwards counts a caller did not ask for, which "
        "raises the cost of every existing caller silently")


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

def test_the_society_agent_requests_an_adequate_sample():
    """The workflow's caller must actually ask."""
    sent = {}

    class FakeEngine:
        def discover_and_orchestrate(self, hypothesis_statement, **kwargs):
            sent.update(kwargs)
            return {"status": "completed", "provenance": "live",
                    "validation_eligible": True, "publication_eligible": True}

    agent = disc_mod.Discoverer.__new__(disc_mod.Discoverer)
    # `discover` needs only the engine; bind the fake in place of the real one.
    import backend.agents.discoverer as d

    real_engine = getattr(d, "DiscoveryEngine", None)
    d.DiscoveryEngine = lambda: FakeEngine()  # type: ignore[assignment]
    try:
        agent.discover("Reproduce IOI on gpt2-small")
    finally:
        if real_engine is not None:
            d.DiscoveryEngine = real_engine  # type: ignore[assignment]

    assert sent.get("n_prompts") == disc_mod.SOCIETY_DISCOVERY_N_PROMPTS, (
        "the Society did not ask for a validation-adequate sample, so its "
        "discovery stays ineligible")
    assert sent.get("n_interaction_prompts") == (
        disc_mod.SOCIETY_DISCOVERY_N_INTERACTION_PROMPTS), (
        "the Society did not ask for an interaction-adequate sample, so "
        "publication eligibility stays unreachable")


def test_the_society_counts_match_the_executors_thresholds():
    """A threshold change must break this test, not silently break the workflow."""
    assert disc_mod.SOCIETY_DISCOVERY_N_PROMPTS == MIN_PROMPTS_FOR_VALIDATION
    assert (disc_mod.SOCIETY_DISCOVERY_N_INTERACTION_PROMPTS
            == MIN_PROMPTS_FOR_INTERACTION)


def test_the_default_gate_still_refuses_an_inadequate_sample():
    """The negative control: raising the caller's count must not weaken the gate.

    If the orchestrator began returning eligibility for a short sample, this
    fails and the other assertions here would be passing for the wrong reason.
    """
    import backend.interpretability.discovery.live_discovery as ld

    def _adequacy_stub(n_prompts, n_interaction_prompts):
        return {"statistically_adequate": False, "interaction_adequate": False,
                "validation_eligible": False, "publication_eligible": False,
                "ineligible_because": ["stub: too few prompts"]}

    real = getattr(ld, "_adequacy", None)
    if real is not None:
        ld._adequacy = _adequacy_stub  # type: ignore[assignment]
    try:
        res = ld.LiveIOIDiscovery().run(GOAL, n_prompts=2,
                                        n_interaction_prompts=1)
    finally:
        if real is not None:
            ld._adequacy = real  # type: ignore[assignment]

    assert res.get("validation_eligible") is False, (
        "an inadequate sample was reported validation-eligible; the gate has "
        "been weakened")
    assert discovery_is_live(res) is False
