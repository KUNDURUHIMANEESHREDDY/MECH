"""Live discovery must not be eligible for publication just because it ran.

Two defects in `live_discovery.py`, both about a result describing itself rather
than about the measurement.

1. Eligibility was asserted, not derived.
   `LiveIOIDiscovery.run` returned `validation_eligible: True` and
   `publication_eligible: True` unconditionally, as literals next to real
   measurements. The defaults were four prompts, and the pairwise interaction
   stage ran on `prompts[:2]` -- so a four-prompt run whose edges were averaged
   from two observations declared itself fit for publication.

   The flags were not a judgement about the result. They were constants, and they
   sat in the most load-bearing position in the record: `evidence_policy` gates on
   exactly these fields.

2. `discovery_id` was different on every interpreter start.
   The id was `abs(hash((hypothesis_statement, len(prompts), tuple(clean...))))`.
   Python salts `hash` of strings per process unless `PYTHONHASHSEED` is pinned,
   so byte-identical inputs produced 0c49e6ce, then 78e1c9a0, then 1cca0669.

   That defeats the purpose of an identifier. `recall(discovery_id)` cannot find
   a previous run, two runs of the same hypothesis cannot be deduplicated, and a
   discovery cannot be cited by the ID it was assigned.

   It is now SHA-256 over a canonical JSON encoding, stable across processes,
   platforms and Python versions, and still discriminating on model and
   hypothesis.

The eligibility ladder, and why `replicated` is always False
-----------------------------------------------------------
    live                  -- weights loaded, forward passes ran
    measured              -- a faithfulness figure was computed
    statistically_adequate -- enough prompts behind the screening stage
    interaction_adequate   -- enough prompts behind the pairwise edges
    validation_eligible    -- measured and statistically_adequate
    publication_eligible  -- both of the above

`replicated` is hardcoded False, on purpose: this module runs one model at one
seed and performs no replication, so the flag could only ever be asserted. An
asserted rung is worse than a missing one, because a missing rung is visible.
"""

from __future__ import annotations

import os
import re

import pytest

from backend.interpretability.discovery.live_discovery import (
    MIN_PROMPTS_FOR_INTERACTION,
    MIN_PROMPTS_FOR_VALIDATION,
    _adequacy,
    _discovery_id,
)


PROMPTS = [
    {"clean": "When John and Mary went to the store, John gave a drink to",
     "corrupted": "When Mary and John went to the store, Mary gave a drink to",
     "io": "Mary", "subject": "John"},
]

#: A second panel entry. Needed because `PROMPTS[:1] == PROMPTS` -- with a single
#: prompt, "a different prompt count" is not a different input, and a test
#: asserting the id changes with prompt count would fail for the right reason
#: while looking like a defect in the id.
TWO_PROMPTS = PROMPTS + [
    {"clean": "When Alice and Bob went to the library, Bob handed a book to",
     "corrupted": "When Bob and Alice went to the library, Alice handed a book to",
     "io": "Alice", "subject": "Bob"},
]


# ── Eligibility is derived ──────────────────────────────────────────────────

def test_four_prompts_is_not_publication_eligible():
    """The defaults this module shipped with."""
    a = _adequacy(n_prompts=4, n_interaction_prompts=2, faithfulness=0.72)

    assert a["live"] is True
    assert a["measured"] is True
    assert a["statistically_adequate"] is False
    assert a["validation_eligible"] is False
    assert a["publication_eligible"] is False
    assert any("4 prompts" in r for r in a["ineligible_because"])


def test_enough_prompts_but_thin_interactions_is_not_publication_eligible():
    """The distinction the unconditional flags erased.

    With ten prompts the screening stage is adequately sampled, but the pairwise
    edges are still a mean of two observations. Validation is supported;
    publication is not.
    """
    a = _adequacy(n_prompts=10, n_interaction_prompts=2, faithfulness=0.8755)

    assert a["statistically_adequate"] is True
    assert a["validation_eligible"] is True
    assert a["interaction_adequate"] is False
    assert a["publication_eligible"] is False, (
        "publication eligibility must also require the interaction stage to be "
        "adequately sampled"
    )
    assert any("pairwise edges" in r for r in a["ineligible_because"])


def test_full_ladder_reaches_publication_eligible():
    a = _adequacy(n_prompts=10, n_interaction_prompts=5, faithfulness=0.8755)

    assert a["validation_eligible"] is True
    assert a["publication_eligible"] is True
    assert a["ineligible_because"] is None


def test_no_faithfulness_blocks_everything():
    a = _adequacy(n_prompts=20, n_interaction_prompts=20, faithfulness=None)

    assert a["measured"] is False
    assert a["validation_eligible"] is False
    assert a["publication_eligible"] is False
    assert any("faithfulness" in r for r in a["ineligible_because"])


def test_replicated_is_never_asserted():
    """One model, one seed, no replication -- so the rung stays False."""
    for n, ni, f in ((4, 2, 0.7), (10, 5, 0.9), (20, 20, 0.9)):
        a = _adequacy(n, ni, f)
        assert a["replicated"] is False
        assert "not measured" in a["replicated_reason"]


def test_eligibility_ladder_is_monotone_in_prompt_count():
    """More prompts must never reduce eligibility."""
    previous = None
    for n in (1, 4, 9, 10, 12, 20):
        a = _adequacy(n_prompts=n, n_interaction_prompts=20, faithfulness=0.8)
        current = (a["validation_eligible"], a["publication_eligible"])
        if previous is not None:
            assert current[0] >= previous[0], f"validation regressed at n={n}"
            assert current[1] >= previous[1], f"publication regressed at n={n}"
        previous = current


def test_thresholds_are_reported_so_the_bar_is_visible():
    a = _adequacy(n_prompts=4, n_interaction_prompts=2, faithfulness=0.7)
    assert a["min_prompts_for_validation"] == MIN_PROMPTS_FOR_VALIDATION
    assert a["min_prompts_for_interaction"] == MIN_PROMPTS_FOR_INTERACTION
    assert a["n_prompts"] == 4
    assert a["n_interaction_prompts"] == 2


def test_no_weights_path_is_still_fail_closed(monkeypatch):
    """The unavailable branch reports the whole ladder, not just two flags.

    It previously emitted only `validation_eligible` / `publication_eligible`, so
    a consumer reading `measured` found the key absent -- indistinguishable from a
    result where the field had simply not been computed.
    """
    from backend.interpretability.discovery import live_discovery as ld

    def _no_weights():
        raise ld.lm.LiveUnavailable("no weights")

    monkeypatch.setattr(ld.lm, "_engine", _no_weights)
    result = ld.LiveIOIDiscovery().run(hypothesis_statement="h")

    assert result["status"] == "unavailable"
    assert result["provenance"] == "unavailable"
    assert result["measured"] is False
    assert result["replicated"] is False
    assert result["validation_eligible"] is False
    assert result["publication_eligible"] is False
    assert "no weights" in result["reason"]


# ── discovery_id is stable and discriminating ───────────────────────────────

def test_discovery_id_is_deterministic_within_a_process():
    a = _discovery_id("hypothesis text", PROMPTS, "gpt2-small")
    b = _discovery_id("hypothesis text", PROMPTS, "gpt2-small")
    assert a == b
    assert a.startswith("disc_live_")


def test_discovery_id_varies_with_hypothesis_model_and_prompts():
    base = _discovery_id("h", PROMPTS, "gpt2-small")
    assert _discovery_id("h2", PROMPTS, "gpt2-small") != base, "hypothesis ignored"
    assert _discovery_id("h", PROMPTS, "gpt2-medium") != base, "model ignored"
    assert _discovery_id("h", TWO_PROMPTS, "gpt2-small") != base, "prompt count ignored"

    other = dict(PROMPTS[0], io="Alice")
    assert _discovery_id("h", [other], "gpt2-small") != base, "prompt content ignored"


def test_discovery_id_is_stable_across_processes():
    """The regression: `hash()` is salted per interpreter.

    Runs a subprocess twice with no PYTHONHASHSEED override and requires the same
    id, which is the property `hash()` cannot provide.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, "backend")
        from backend.interpretability.discovery.live_discovery import _discovery_id
        prompts = {PROMPTS!r}
        print(_discovery_id("cross-process check", prompts, "gpt2-small"))
    """)
    script_path = "_discovery_id_stability_probe.py"
    with open(script_path, "w", encoding="utf-8") as handle:
        handle.write(script)

    try:
        # Inherit the environment: importing torch in the child needs PATH and
        # the rest. Deliberately NOT setting PYTHONHASHSEED -- pinning it would
        # make hash() stable and hide the very thing this test exists to catch.
        env = dict(os.environ)
        env["PYTHONPATH"] = "backend;."
        env["PYTHONIOENCODING"] = "utf-8"
        env.pop("PYTHONHASHSEED", None)

        outs = []
        for _ in range(3):
            proc = subprocess.run(
                [sys.executable, script_path],
                capture_output=True, text=True, env=env, timeout=300,
            )
            assert proc.returncode == 0, proc.stderr[-800:]
            outs.append(proc.stdout.strip().splitlines()[-1])
    finally:
        os.remove(script_path)

    assert len(set(outs)) == 1, (
        f"discovery_id differs across processes: {outs}. Python salts str hashing "
        f"per interpreter, so a hash()-based id cannot be stable."
    )


def test_discovery_id_source_does_not_use_hash():
    from source_assert import executable_source

    import backend.interpretability.discovery.live_discovery as ld

    src = executable_source(ld)

    assert not re.search(r"abs\(hash\(", src), (
        "discovery_id is hash()-based again; that is unstable across processes"
    )
    assert "hashlib.sha256" in src