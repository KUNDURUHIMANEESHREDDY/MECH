"""Confidence defaults must be absent, not plausible.

MECH's contract is that a value is either measured or marked as not-measured,
with nothing in between. Confidence scores violated that repeatedly, because a
default is the easiest number to invent and the hardest to notice: every caller
that omits the argument silently inherits it.

The four fixed instances, all of which *manufactured* confidence:

  GraphNode.confidence_score            1.0   -- total certainty, in the
                                                 evidence store
  MechanismRegistry seed entry           0.96  -- with replication_score 0.985
                                                 and status "Validated"
  register_mechanism confidence default  0.95  -- plus 0.90 and 0.85 for the
                                                 two quality scores
  seeded knowledge-graph SUPPORTS edge  0.95  -- a confidence on evidence that
                                                 was transcribed from a paper

`register_mechanism` was the most damaging, because it did not merely default a
score: it set `status: "Validated"` unconditionally, turning an unevidenced claim
into a validated one in a single call. `status` is now "Registered", which is
the only statement the function can actually support.

Deliberately NOT changed: `critic.is_confident` and `society._confidence_of`
default a *missing* confidence to 0.0. That direction is conservative -- 0.0
fails critic's 0.85 threshold and returns False -- so it cannot manufacture a
positive verdict, which is what the 0.95/0.96/1.0 defaults did. Changing them
would be churn without a defect. They are pinned by a test here so that nobody
"fixes" them into an optimistic default later.

That reasoning covered the confidence score only. The *peer review* default in
the same function was a separate defect and is now fixed: `decision` defaulted
to "" at two levels (an absent `peer_review` dict, and an absent `decision`
key), and "" was an accepted value, so a validation with a 0.90 score and no
peer review at all passed. A missing review is now a failed review. See
`test_missing_peer_review_is_not_an_accept` below.
"""

from __future__ import annotations

import re

import pytest


# ── The evidence store ──────────────────────────────────────────────────────

def test_knowledge_graph_nodes_do_not_default_to_full_confidence():
    from backend.science.explorer.knowledge_graph import GraphNode

    node = GraphNode(id="n", type="Circuit", label="x")
    assert node.confidence_score is None, (
        f"GraphNode defaults to confidence {node.confidence_score!r}; a node "
        f"added without a measured score must be absent, not certain"
    )


def test_add_node_does_not_default_to_full_confidence():
    import inspect

    from backend.science.explorer.knowledge_graph import MechanisticKnowledgeGraph

    default = inspect.signature(MechanisticKnowledgeGraph.add_node).parameters[
        "confidence_score"
    ].default
    assert default is None, f"add_node defaults confidence_score to {default!r}"

    node = MechanisticKnowledgeGraph().add_node("Circuit", "some circuit")
    assert node.confidence_score is None


def test_seeded_knowledge_graph_carries_no_invented_confidence():
    """The seeded graph is a transcription of Wang et al., so it is labelled.

    `_seed_mock_data` has no callers, so the old `confidence_score: 0.95` was
    latent rather than active -- but a dev-seeded graph is exactly the thing
    someone screenshots into a slide, so it must not carry a number nothing
    measured.
    """
    from backend.science.explorer.knowledge_graph import MechanisticKnowledgeGraph

    graph = MechanisticKnowledgeGraph()
    # The graph is a process-wide singleton (`__new__` caches `_instance`), and
    # `_seed_mock_data` returns early when nodes already exist. Earlier tests in
    # this file add nodes, so without clearing, seeding silently does nothing and
    # every assertion below would pass against an unrelated graph.
    graph.clear()
    graph._seed_mock_data()

    assert graph.nodes and graph.edges, (
        "seeding produced no nodes or edges; this test would be vacuous"
    )
    for edge in graph.edges:
        assert "confidence_score" not in edge.metadata, (
            f"{edge.relationship} edge carries an unmeasured confidence"
        )
    for node in graph.nodes.values():
        assert node.confidence_score is None
        assert node.metadata.get("provenance") == "seeded", (
            f"seeded node {node.id} is not marked as seeded"
        )
        assert node.metadata.get("validation_eligible") is False
        assert node.metadata.get("publication_eligible") is False


# ── The mechanism registry ──────────────────────────────────────────────────

def test_registering_a_mechanism_does_not_validate_it():
    """Registration is bookkeeping; it is not evidence.

    Previously `register_mechanism` set `evidence_count: 1`,
    `replication_score: 0.95` and `status: "Validated"` for every call. So one
    line turned an unevidenced claim into a validated mechanism.
    """
    from backend.interpretability.discovery.mechanism_registry import MechanismRegistry

    entry = MechanismRegistry().register_mechanism("m1", "Some Mechanism")

    assert entry["status"] != "Validated"
    assert entry["status"] == "Registered"
    assert entry["replication_score"] is None, (
        "a replication score was fabricated; no replication was run"
    )
    assert entry["confidence_score"] is None
    assert entry["scientific_quality_score"] is None
    assert entry["expected_impact_score"] is None
    assert entry["evidence_count"] == 0, (
        "registering a name attaches no evidence"
    )


def test_register_mechanism_accepts_a_measured_score():
    """The fix is absence-by-default, not an inability to record one."""
    from backend.interpretability.discovery.mechanism_registry import MechanismRegistry

    entry = MechanismRegistry().register_mechanism(
        "m2", "Measured Mechanism",
        confidence_score=0.62, evidence_count=3,
    )
    assert entry["confidence_score"] == 0.62
    assert entry["evidence_count"] == 3
    # Still not validated by supplying a score -- validation is a separate,
    # evidence-backed transition that this method does not perform.
    assert entry["status"] == "Registered"


def test_mechanism_registry_seed_entry_is_not_a_validated_result():
    from backend.interpretability.discovery.mechanism_registry import MechanismRegistry

    seed = MechanismRegistry().list_mechanisms()[0]
    assert seed["status"] != "Validated"
    assert seed["replication_score"] is None
    assert seed["confidence_score"] is None
    assert seed["evidence_count"] == 0
    assert seed["provenance"] == "seeded"
    assert seed["validation_eligible"] is False


# ── The claim registry ──────────────────────────────────────────────────────

def test_a_new_claim_is_not_pre_validated():
    """Constructing a claim must not manufacture evidence for it.

    `RegisteredMechanismClaim` defaulted to `status="Validated"`,
    `confidence=0.95`, `replications=1` and `supporting_experiments=1`, so
    passing only an id, title and description produced a claim that was already
    validated, at 0.95 confidence, with one replication and one supporting
    experiment in hand. `supporting_experiments=1` was the sharpest edge: it
    asserted an experiment had been run and had come out in favour, when the only
    thing that had happened was the constructor.
    """
    from backend.interpretability.discovery.mechanism_claim_registry import (
        RegisteredMechanismClaim,
    )

    claim = RegisteredMechanismClaim(claim_id="c", title="t", description="d")

    assert claim.status == "Hypothesized"
    assert claim.confidence is None
    assert claim.replications == 0
    assert claim.supporting_experiments == 0
    # `contradicting_experiments=0` is left alone: zero is a true statement about
    # a claim that has just been written down.
    assert claim.contradicting_experiments == 0


def test_deserialising_a_claim_does_not_promote_it_to_validated():
    """The load path was worse than the constructor.

    `from_dict` defaulted `status` to "Validated", `confidence` to 0.95,
    `replications` to 1 and `supporting_experiments` to 1, so every stored record
    written before those fields existed was silently promoted the moment it was
    read back.
    """
    from backend.interpretability.discovery.mechanism_claim_registry import (
        RegisteredMechanismClaim,
    )

    claim = RegisteredMechanismClaim.from_dict(
        {"claim_id": "c", "title": "t", "description": "d"}
    )
    assert claim.status == "Hypothesized"
    assert claim.confidence is None
    assert claim.replications == 0
    assert claim.supporting_experiments == 0


def test_seeded_claims_are_reported_not_validated(tmp_path):
    """The seeds are literature transcriptions, not MECH evidence.

    They previously claimed `confidence=0.962/0.941`, `replications=14/22` and
    `supporting_experiments=103/145`. There is no record of 103 experiments and
    the IOI circuit has been discovered once on this platform.
    """
    from backend.interpretability.discovery.mechanism_claim_registry import (
        MechanismClaimRegistry,
    )

    registry = MechanismClaimRegistry(storage_dir=str(tmp_path))
    registry._seed_defaults()

    claims = registry.list_all()
    assert claims, "seeding produced no claims; this test would be vacuous"
    for claim in claims:
        assert claim.status == "Reported", claim.claim_id
        assert claim.confidence is None
        assert claim.replications == 0
        assert claim.supporting_experiments == 0
        assert claim.literature, "the citation is the provenance and must survive"


def test_claim_confidence_is_derived_from_recorded_experiments(tmp_path):
    """Confidence must come from the evidence counts, not a fixed step.

    It was `min(0.99, confidence + (1 - confidence) * 0.05)` per success and
    `-0.05` per failure, applied to a field starting at 0.95 and capped at 0.99 --
    so the value moved by a constant per event regardless of what the event was,
    and twenty consecutive successes could only reach 0.99.
    """
    from backend.interpretability.discovery.mechanism_claim_registry import (
        MechanismClaimRegistry,
    )

    registry = MechanismClaimRegistry(storage_dir=str(tmp_path))
    registry._seed_defaults()
    cid = "claim_ioi_name_mover"

    registry.record_replication(cid, model_id="GPT2-S", success=True)
    claim = registry.get(cid)
    assert claim.confidence == 1.0            # 1 supporting / 1 total
    assert claim.supporting_experiments == 1

    registry.record_replication(cid, model_id="GPT2-S", success=False)
    claim = registry.get(cid)
    assert claim.confidence == 0.5            # 1 supporting / 2 total
    assert claim.contradicting_experiments == 1

    # And it must move in both directions, which the old step model also did --
    # but from a base that started at 0.95 and could not fall.
    assert claim.confidence < 0.95


def test_replicator_does_not_multiply_component_count_into_experiments():
    """`supporting_experiments = len(components) * 10` was a visual proxy.

    Three circuit components became "30 supporting experiments". It was a number
    in a field meaning an experiment had been run, derived from arithmetic on a
    list of strings.
    """
    from source_assert import executable_source

    import backend.interpretability.discovery.autonomous_paper_replicator as repl

    src = executable_source(repl)

    assert not re.search(r"supporting_experiments\s*=\s*len\(", src), (
        "an evidence count is being derived from a component count again"
    )
    assert not re.search(r"supporting_experiments\s*=\s*\w+\s*\*\s*\d+", src)


# ── Conservative defaults, pinned so they are not "improved" ───────────────

def test_missing_confidence_defaults_to_zero_and_stays_conservative():
    """0.0 is the safe direction; pin it so it is not raised to something else.

    These two are deliberately left as-is. A missing confidence becomes 0.0,
    which fails critic's 0.85 threshold and cannot produce a positive verdict.
    The fabricated defaults this file is about went the other way.

    A payload only reaches the confidence check once `validation_is_live` passes,
    which requires status completed, provenance "live", `attested is True`,
    both eligibility flags set, and `validated: True`. Asserting against a
    payload that fails that gate would prove nothing -- every such case returns
    False regardless of the confidence value -- so the positive case is built
    to actually reach it.
    """
    from backend.agents.critic import CONFIDENCE_THRESHOLD, Critic

    assert CONFIDENCE_THRESHOLD == 0.85
    critic = Critic()

    def live(confidence, decision="Accept"):
        return {
            "status": "completed",
            "provenance": "live",
            "attested": True,
            "validation_eligible": True,
            "publication_eligible": True,
            "validated": True,
            "confidence": confidence,
            "peer_review": {"decision": decision},
        }

    # Reaches the threshold check and clears it.
    assert critic.is_confident(live({"confidence_score": CONFIDENCE_THRESHOLD})) is True

    # Absent confidence -> 0.0 -> below threshold. This is the case being pinned:
    # if the default were ever raised toward 0.95, this would flip to True.
    assert critic.is_confident(live({})) is False
    assert critic.is_confident(live({"confidence_score": 0.0})) is False

    # And the gate itself still holds: a high score on a non-live payload is
    # rejected, so raising the default could not smuggle one through anyway.
    not_live = live({"confidence_score": 1.0})
    not_live["provenance"] = "unavailable"
    assert critic.is_confident(not_live) is False


class _Absent:
    """Sentinel: distinguishes 'no peer_review key' from 'an empty one'."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<absent>"


_ABSENT = _Absent()


def test_missing_peer_review_is_not_an_accept():
    """A review that did not happen is not a review that succeeded.

    `is_confident` accepted `decision in ("", "Accept")`, and `decision` was
    "" whenever `peer_review` was absent, or present without a `decision` key.
    So confidence 0.90 with no peer review whatsoever passed the gate, while
    the documentation for the same function said peer review must be Accept.
    """
    from backend.agents.critic import CONFIDENCE_THRESHOLD, Critic

    critic = Critic()

    def payload(peer_review):
        record = {
            "status": "completed",
            "provenance": "live",
            "attested": True,
            "validation_eligible": True,
            "publication_eligible": True,
            "validated": True,
            "confidence": {"confidence_score": CONFIDENCE_THRESHOLD},
        }
        if peer_review is not _ABSENT:
            record["peer_review"] = peer_review
        return record

    # The cases that used to pass.
    assert critic.is_confident(payload(_ABSENT)) is False, (
        "an absent peer_review satisfied the gate")
    assert critic.is_confident(payload({})) is False
    assert critic.is_confident(payload({"decision": ""})) is False
    assert critic.is_confident(payload({"decision": None})) is False

    # Decisions that are not an accept.
    for decision in ("Reject", "Revise", "pending", "accept", "ACCEPT",
                     "Accepted"):
        assert critic.is_confident(payload({"decision": decision})) is False, decision

    # Surrounding whitespace is tolerated, so a real reviewer's stray space does
    # not silently fail a legitimate accept. The comparison is exact otherwise.
    assert critic.is_confident(payload({"decision": "Accept"})) is True
    assert critic.is_confident(payload({"decision": " Accept "})) is True


# ── Source-level guard against reintroduction ───────────────────────────────

@pytest.mark.parametrize(
    "module_path",
    [
        "backend.science.explorer.knowledge_graph",
        "backend.interpretability.discovery.mechanism_registry",
        "backend.interpretability.discovery.mechanism_claim_registry",
    ],
)
def test_no_confidence_default_is_a_plausible_float(module_path):
    """Catch any new `confidence_score: float = 0.x` default in these modules."""
    import importlib

    from source_assert import executable_source

    module = importlib.import_module(module_path)
    src = executable_source(module)

    offenders = re.findall(
        r"confidence(?:_score)?\s*:\s*float\s*=\s*(0\.\d+|1\.0)", src
    )
    assert not offenders, (
        f"{module_path} defaults a confidence to {offenders}; an unmeasured "
        f"confidence must default to None"
    )
