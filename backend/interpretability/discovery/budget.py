"""The cost budget for one discovery, and whether it should stop.

A Society run is a CPU-bound measurement, not a request handler. The live
discovery sweeps every attention head of GPT-2 on every prompt, so the cost of
a run is decided by the caller: a run that asks for ten prompts and five
interaction prompts does hundreds of forward passes, and on a CPU that is
minutes. Two facts therefore have to travel with the run:

* **the effort** -- how many prompts, how many interaction prompts, how many
  candidate heads. A caller that wants a cheap run says so, and the result
  reports the effort it actually used, so a quick run can never be mistaken
  for a thorough one;
* **a stop check** -- so a cancelled or timed-out run stops between
  measurements instead of burning CPU until the pipeline happens to return.

Why a context variable rather than a parameter threaded through every call:
the sweep is several layers deep (society -> discoverer -> discovery engine ->
live discovery -> live measurement), and a stop that cannot be observed where
the work actually happens is not a stop. The context travels with the call
chain, including into worker threads (``asyncio.to_thread`` copies the
context), and it is scoped to one run: nothing is left set afterwards.

The budget is optional. With no budget set, every layer keeps its own
defaults, exactly as before.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import Callable, Optional

#: The budget for the current run, if any.
_BUDGET: contextvars.ContextVar[Optional["DiscoveryBudget"]] = contextvars.ContextVar(
    "mech_discovery_budget", default=None
)


@dataclass(frozen=True)
class DiscoveryBudget:
    """What one discovery is allowed to spend, and when to stop.

    ``should_stop`` is called between measurements. Returning True means the
    run has been cancelled or has hit its deadline; the caller must stop
    promptly and report that, not carry on and report a result as if nothing
    had happened.
    """

    n_prompts: int
    n_interaction_prompts: int
    max_heads: int
    should_stop: Callable[[], bool] = lambda: False

    def stopped(self) -> bool:
        """True when the run has been asked to stop."""
        try:
            return bool(self.should_stop())
        except Exception:
            # A broken stop check must not crash the measurement; the run
            # continues and is bounded by its own timeout.
            return False


def set_budget(budget: Optional[DiscoveryBudget]) -> contextvars.Token:
    """Install ``budget`` for the current context. Returns a reset token."""
    return _BUDGET.set(budget)


def reset_budget(token: contextvars.Token) -> None:
    """Remove the budget installed by :func:`set_budget`."""
    try:
        _BUDGET.reset(token)
    except ValueError:
        # The token was created in another context (a different thread); the
        # budget is then simply not set here.
        _BUDGET.set(None)


def current_budget() -> Optional[DiscoveryBudget]:
    """The budget for the current context, or None."""
    return _BUDGET.get()


def stop_requested() -> bool:
    """True when the current run has been asked to stop."""
    budget = _BUDGET.get()
    return budget.stopped() if budget is not None else False


def effective_effort(n_prompts: Optional[int],
                     n_interaction_prompts: Optional[int],
                     max_heads: Optional[int]) -> tuple:
    """Resolve the effort for a discovery: the budget wins, else the caller's.

    The budget wins because it is the *run owner's* decision. The Society
    decides how much a run may cost; the layers below carry their own
    defaults, which exist for a direct call, not for a run someone asked to
    be cheap. The resolved triple is reported with the result, so the cost
    that was paid is always visible next to the result it produced, and the
    eligibility flags are derived from that same sample.
    """
    budget = _BUDGET.get()
    if budget is None:
        return (n_prompts, n_interaction_prompts, max_heads)
    return (
        int(budget.n_prompts),
        int(budget.n_interaction_prompts),
        int(budget.max_heads),
    )


def _self_check() -> None:
    assert current_budget() is None
    assert stop_requested() is False
    token = set_budget(DiscoveryBudget(n_prompts=2, n_interaction_prompts=1,
                                       max_heads=4))
    try:
        assert current_budget() is not None
        assert effective_effort(None, None, None) == (2, 1, 4)
        # The budget wins over the caller's own numbers: it is the run
        # owner's decision.
        assert effective_effort(10, 5, 12) == (2, 1, 4)
        assert stop_requested() is False
    finally:
        reset_budget(token)
    assert current_budget() is None

    state = {"stop": False}
    token = set_budget(DiscoveryBudget(n_prompts=1, n_interaction_prompts=1,
                                       max_heads=1,
                                       should_stop=lambda: state["stop"]))
    try:
        assert stop_requested() is False
        state["stop"] = True
        assert stop_requested() is True
    finally:
        reset_budget(token)


if __name__ == "__main__":
    _self_check()
    print("discovery budget self-check passed")
