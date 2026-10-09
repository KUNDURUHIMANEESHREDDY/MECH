"""Attested evidence: live labels without attestation are not evidence.

Seams: measurement source (backend.core.provenance attest/withhold),
wrapper boundary (dispatcher._mark), policy predicates
(backend.agents.evidence_policy), graph extraction
(backend.core.evidence_graph._step_allows_evidence / from_run).
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.agents import evidence_policy as pol  # noqa: E402
from backend.core import evidence_graph as eg  # noqa: E402
from backend.core.provenance import (  # noqa: E402
    attest_measurement,
    pass_through,
    withhold,
)


def _eligible(**overrides):
    payload = {"status": "completed", "provenance": "live",
               "attested": True, "validation_eligible": True,
               "publication_eligible": True}
    payload.update(overrides)
    return payload


# ── measurement source ──────────────────────────────────────────────────

def test_attest_marks_attested():
    out = attest_measurement({"status": "ok", "value": 1.0})
    assert out["provenance"] == "live" and out["attested"] is True
    assert "attested" not in out["field_provenance"]


def test_attest_refuses_to_upgrade_a_bare_live_label():
    """This used to assert the opposite, and it was wrong.

    It read:

        def test_attest_repairs_a_bare_live_label():
            out = attest_measurement({"provenance": "live", "value": 1.0})
            assert out["attested"] is True

    Calling that a "repair" assumed the only way a `live` label can exist is
    that the measurement layer wrote it. But `provenance` is an ordinary dict
    key, so any component that builds a result can set it -- and the function
    cannot tell the two apart. Honouring the label meant anyone could mint an
    attestation by typing two words into a dictionary.

    `attested` is what `evidence_policy._is_attested` and the evidence graph
    gate on, so that is a scientific claim, and it has to be issued by the
    measurement boundary rather than inherited from a label.
    """
    out = attest_measurement({"provenance": "live", "value": 1.0})
    assert out["attested"] is not True, (
        "attest_measurement issued an attestation from a pre-existing "
        "'live' label alone")
    assert out["provenance"] != "live"
    assert out["reason"], "the refusal must record why"


def test_attest_still_withholds_failures():
    out = attest_measurement({"status": "unavailable", "value": 1.0})
    assert out["provenance"] != "live" and out["attested"] is False


def test_withhold_is_explicitly_unattested():
    assert withhold({"status": "ok"}, reason="r")["attested"] is False
    seeded = withhold({"provenance": "seeded"}, reason="r")
    assert seeded["provenance"] == "seeded" and seeded["attested"] is False


# ── wrapper boundary ────────────────────────────────────────────────────

def test_mark_never_invents_live():
    from backend.api import dispatcher
    # The audit's case: availability is not attestation.
    out = dispatcher._mark({"status": "ok", "value": 1.0}, "live")
    assert out["provenance"] != "live"
    assert out["provenance"] == "unavailable"
    # An attested engine label passes through untouched.
    engine = {"status": "ok", "provenance": "live", "attested": True,
              "field_provenance": {"status": "live"}}
    assert dispatcher._mark(dict(engine), "live") == engine
    # Unlabeled failures stay failures, not upgrades.
    out = dispatcher._mark({"status": "unavailable"}, "live")
    assert out["provenance"] == "unavailable"


# ── policy predicates ───────────────────────────────────────────────────

def test_predicates_require_attestation():
    assert pol.discovery_is_live(_eligible()) is True
    assert pol.discovery_is_live(_eligible(attested=False)) is False
    assert pol.discovery_is_live(_eligible(attested=None)) is False
    del_attested = _eligible()
    del del_attested["attested"]
    assert pol.discovery_is_live(del_attested) is False


def test_validation_reproduction_gate_require_attestation():
    valid = _eligible(validated=True)
    assert pol.validation_is_live(valid) is True
    assert pol.validation_is_live(_eligible(validated=True, attested=False)) is False
    assert pol.reproduction_is_live(_eligible(mock_mode=False)) == \
        _eligible(mock_mode=False)
    assert pol.reproduction_is_live(_eligible(mock_mode=False, attested=False)) == {}
    assert pol.gate_is_live(_eligible(passed=True)) is True
    assert pol.gate_is_live(_eligible(passed=True, attested=False)) is False
    reason = pol.blocked_reason(_eligible(attested=False), "Discovery")
    assert "attest" in reason


# ── graph extraction ────────────────────────────────────────────────────

def _executor_step():
    return {"node": "execute", "agent": "Executor", "status": "ok",
            "op": "patch_head", "provenance": "live",
            "result": {"status": "ok", "delta": 0.0146, "confidence_score": 0.9}}


def _discover_step(attested=True):
    result = {"status": "completed", "provenance": "live",
              "validation_eligible": True, "publication_eligible": True,
              "discovery_id": "d1", "result": {"delta": 0.0146}}
    if attested:
        result["attested"] = True
        step_attested = True
    else:
        step_attested = False
    return {"node": "discover", "agent": "Discoverer", "status": "completed",
            "provenance": "live", "attested": step_attested, "result": result}


def test_executor_numerics_never_become_evidence():
    assert eg._step_allows_evidence(_executor_step()) is False
    graph = eg.TraceableEvidenceGraph.from_run("rab12cd34ef56", "goal",
                                               [_executor_step()])
    assert [n for n in graph.nodes if n["type"] == "Evidence"] == []


def test_attested_discover_step_yields_evidence():
    step = _discover_step(attested=True)
    assert eg._step_allows_evidence(step) is True
    graph = eg.TraceableEvidenceGraph.from_run("rab12cd34ef56", "goal", [step])
    evidence = [n for n in graph.nodes if n["type"] == "Evidence"]
    assert evidence and all(n["payload"]["provenance"] == "live"
                            for n in evidence)


def test_unattested_discover_step_yields_nothing():
    assert eg._step_allows_evidence(_discover_step(attested=False)) is False
    graph = eg.TraceableEvidenceGraph.from_run("rab12cd34ef56", "goal",
                                               [_discover_step(attested=False)])
    assert [n for n in graph.nodes if n["type"] == "Evidence"] == []


def test_attested_non_gated_node_yields_nothing():
    step = dict(_executor_step(), attested=True)
    assert eg._step_allows_evidence(step) is False
