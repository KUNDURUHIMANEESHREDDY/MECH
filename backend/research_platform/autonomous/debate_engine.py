"""Scientific Debate & Consensus Engine.

What this used to be
--------------------
A function that returned a debate it had not had:

    {
        "hypothesis_a": hypothesis_a,
        "hypothesis_b": hypothesis_b,
        "debate_rounds": 3,
        "counter_evidence_evaluated": True,
        "consensus_winner": hypothesis_a,     # always A
        "confidence": 0.93,
    }

`consensus_winner` was unconditionally `hypothesis_a`, so the engine's verdict did
not depend on its inputs -- swapping the two hypotheses returned the same winner.
`debate_rounds: 3` and `counter_evidence_evaluated: True` asserted that rounds
happened and that counter-evidence was weighed; nothing was exchanged. And the
`confidence: 0.93` was a constant attached to that predetermined outcome.

A debate whose winner is fixed by argument order is not a debate, and reporting
it as one puts a number in a field a reader will treat as a finding.

What it does now
----------------
It takes the arguments that a consensus needs -- which hypothesis has more
supporting evidence, how much, and from where -- and derives the verdict from
them. Without them it reports that no consensus was reached, which is the honest
state for a debate where nobody presented anything.

`counter_evidence` is counted rather than asserted: a hypothesis supported only
by supporting evidence and met by none is reported as untested, not as unrefuted.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional


class ScientificDebateEngine:
    """Derives a consensus verdict from presented evidence.

    Does not generate the arguments. Callers present them; this decides between
    what was presented and states how weak that basis is.
    """

    #: How many arguments the leading side must present before "consensus" is
    #: claimed. A count, deliberately: summing per-argument support and comparing
    #: that against a threshold makes the bar depend on how the strengths happen
    #: to be scaled, so two arguments at 0.8 each would fail a bar that two
    #: arguments at 0.6 each passed for no reason other than the scaling.
    MIN_ARGUMENTS_FOR_CONSENSUS = 2

    def debate_hypotheses(
        self,
        hypothesis_a: str,
        hypothesis_b: str,
        evidence_a: Optional[List[Dict[str, Any]]] = None,
        evidence_b: Optional[List[Dict[str, Any]]] = None,
        counter_evidence_a: Optional[List[Dict[str, Any]]] = None,
        counter_evidence_b: Optional[List[Dict[str, Any]]] = None,
        rounds: int = 0,
    ) -> Dict[str, Any]:
        """Compare two hypotheses on the evidence supplied for each.

        `evidence_a` / `evidence_b` are the supporting arguments, each
        ``{"claim": str, "support": float}`` where `support` is in [0, 1] on the
        strength of that single argument. `rounds` is the number of exchange
        rounds that actually happened -- a number the caller knows, not one this
        engine invents.

        With no evidence the result is `verdict: "no_consensus"` with the reason,
        rather than a winner picked by argument order.
        """
        support_a = self._total_support(evidence_a)
        support_b = self._total_support(evidence_b)
        n_args_a = self._count_arguments(evidence_a)
        n_args_b = self._count_arguments(evidence_b)
        counter_a = len(counter_evidence_a or [])
        counter_b = len(counter_evidence_b or [])

        evaluated = bool(evidence_a or evidence_b
                         or counter_evidence_a or counter_evidence_b)

        if not evaluated:
            return {
                "hypothesis_a": hypothesis_a,
                "hypothesis_b": hypothesis_b,
                "debate_rounds": rounds,
                "rounds_reported_by_caller": True,
                "counter_evidence_evaluated": False,
                "verdict": "no_consensus",
                "consensus_winner": None,
                "winning_side": None,
                "confidence": None,
                "reason": (
                    "No evidence or counter-evidence was supplied for either "
                    "hypothesis, so no comparison is possible. This previously "
                    "returned hypothesis_a as the winner with confidence 0.93 "
                    "regardless of input."
                ),
                "evaluated_at": _now(),
            }

        # The winning side is the *slot*, decided by support. Resolving it to a
        # string afterwards keeps `hypothesis_a`/`hypothesis_b` labels out of the
        # comparison, so two calls that pass the same evidence under swapped
        # argument names produce the same verdict.
        winning_side: Optional[str]
        if support_a > support_b:
            winning_side = "a"
        elif support_b > support_a:
            winning_side = "b"
        else:
            winning_side = None

        winner = (hypothesis_a if winning_side == "a"
                  else hypothesis_b if winning_side == "b" else None)
        losing = (hypothesis_b if winning_side == "a"
                  else hypothesis_a if winning_side == "b" else None)

        margin = abs(support_a - support_b)
        leading_args = n_args_a if winning_side == "a" else (
            n_args_b if winning_side == "b" else 0)

        if winning_side is None:
            verdict = "tie"
            reason = (
                f"Both hypotheses carry equal support ({support_a:.4f}), so no "
                f"winner can be named."
            )
        elif leading_args < self.MIN_ARGUMENTS_FOR_CONSENSUS:
            verdict = "insufficient_evidence"
            reason = (
                f"The leading side presents {leading_args} argument(s) against "
                f"{self.MIN_ARGUMENTS_FOR_CONSENSUS} required for a consensus "
                f"claim, so this is a preference rather than a result."
            )
        else:
            verdict = "consensus"
            reason = (
                f"{winning_side.upper()} has support {max(support_a, support_b):.4f} "
                f"from {leading_args} argument(s) against "
                f"{min(support_a, support_b):.4f} from "
                f"{(n_args_b if winning_side == 'a' else n_args_a)}."
            )

        winner_counter_count = (
            counter_a if winning_side == "a"
            else counter_b if winning_side == "b" else None)

        return {
            "hypothesis_a": hypothesis_a,
            "hypothesis_b": hypothesis_b,
            "debate_rounds": rounds,
            "rounds_reported_by_caller": True,
            "counter_evidence_evaluated": (counter_a + counter_b) > 0,
            "counter_evidence_counts": {"a": counter_a, "b": counter_b},
            "n_arguments": {"a": n_args_a, "b": n_args_b},
            "support": {"a": round(support_a, 4), "b": round(support_b, 4)},
            "support_margin": round(margin, 4),
            "verdict": verdict,
            "consensus_winner": winner,
            "winning_side": winning_side,
            "loser": losing,
            # Derived from the support margin and how much counter-evidence the
            # winner had to survive, rather than a constant. Not a probability:
            # this is a summary of the arguments presented, not a probability
            # that the winner is correct.
            "confidence": self._confidence(verdict, margin,
                                          winner_counter_count),
            "confidence_is": (
                "summary of the arguments presented; not a probability that the "
                "winner is correct"
            ),
            "winner_was_countered": bool(winner_counter_count),
            "winner_is_untested": winner is not None and not winner_counter_count,
            "reason": reason,
            "evaluated_at": _now(),
        }

    def _count_arguments(self, evidence: Optional[List[Dict[str, Any]]]) -> int:
        """How many arguments carry a numeric support value.

        Counted separately from the support sum so the consensus bar does not
        depend on how the strengths are scaled.
        """
        if not evidence:
            return 0
        count = 0
        for item in evidence:
            value = item.get("support") if isinstance(item, dict) else item
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                count += 1
        return count

    def _total_support(self, evidence: Optional[List[Dict[str, Any]]]) -> float:
        """Sum of the per-argument support values.

        An argument with no numeric `support` contributes nothing rather than a
        default, since a missing strength is not a weak argument.
        """
        if not evidence:
            return 0.0
        total = 0.0
        for item in evidence:
            if isinstance(item, dict):
                value = item.get("support")
            else:
                value = item
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                total += float(value)
        return total

    def _confidence(
        self,
        verdict: str,
        margin: float,
        winner_counter_count: Optional[int],
    ) -> Optional[float]:
        """A bounded summary of the arguments, or None when there is no verdict.

        Scaled by the margin, then discounted if the winner met no
        counter-evidence -- an untested winner is weaker evidence than a tested
        one, and the discount says so.
        """
        if verdict in {"tie", "insufficient_evidence"}:
            return None

        # Margin of 1.0 (all support on one side) maps to 1.0.
        base = min(1.0, margin)
        if winner_counter_count == 0:
            base *= 0.5
        return round(base, 4)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()