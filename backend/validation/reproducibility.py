"""Discovery Reproduction & Falsification Engine.

Intended to re-run a mechanistic discovery and report whether it reproduces or
can be falsified.

Status: NOT IMPLEMENTED
-----------------------
``reproduce_discovery`` previously returned, for every input::

    return {
        "discovery_id": discovery_id,
        "reproducibility_score": 0.96,
        "falsification_attempts": 3,
        "falsified": False,
        "reproduced_cleanly": True,
    }

That is the most consequential fabrication found in this package, because of
what it claims rather than how plausible it looks. An engine whose stated job is
to *attempt to falsify* a discovery returned `falsified: False` and
`reproduced_cleanly: True` for every id, having attempted nothing. The
`falsification_attempts: 3` was the only trace of work and was itself a literal.
A caller polling this engine would conclude that every discovery it was asked
about had survived scrutiny.

`reproducibility_score: 0.96` compounds it: a single scalar, no denominator, no
task definition, no seed, no model, no prompt set. There is nothing it could have
been computed from.

The method now raises `LiveUnavailable`. It had no callers, so nothing depends on
the old shape.

What implementing this requires, and why it is not a one-liner:
  * Resolve `discovery_id` to an actual claim -- its circuit, its prompts, its
    model, its seed. A reproduction score is meaningless without the thing being
    reproduced pinned down.
  * Re-run the measurement, not a re-read of the stored result. Reproducing your
    own stored number is not reproduction.
  * Define the falsification criterion in advance. "Three attempts" is not a
    criterion; deciding after seeing the result whether it counts is not either.
  * Report per-attempt outcomes so a score can be recomputed from them.
"""

from __future__ import annotations

from typing import Any, Dict

from backend.science.models.adapter_base import LiveUnavailable


class DiscoveryReproductionEngine:
    """Not implemented. Raises rather than confirming every discovery."""

    def reproduce_discovery(self, discovery_id: str) -> Dict[str, Any]:
        raise LiveUnavailable(
            "DiscoveryReproductionEngine.reproduce_discovery is not implemented. "
            "It previously returned reproducibility_score=0.96, "
            "falsification_attempts=3, falsified=False and reproduced_cleanly=True "
            "for every discovery_id, having attempted nothing -- so an engine "
            "whose purpose is to try to falsify a discovery reported that none "
            "of them were falsified. See the module docstring for what a real "
            "implementation requires."
        )
