"""One admission gate for the Society's results.

The Society agents each built their own envelope, so three places decided
what "live" and "eligible" meant and a caller could assert either. The
shared gate derives all three from the record being admitted, and it can
only ever lower a result's standing.

The tests below are mostly negative, because that is where an admission
gate earns its keep: a caller that asserts a label must not be able to
promote itself, a payload that measures nothing must not reach
validation, and a result carrying an unattested run chain must not be
publishable.
"""

from __future__ import annotations

import ast
import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Dict

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # noqa: E402
    Ed25519PrivateKey,
)

from backend.agents import evidence_policy as pol  # noqa: E402
from backend.core.provenance import (  # noqa: E402
    admit_result,
    attest_measurement,
    reset_trusted_key_ids_cache,
)

# The signing key is created once at import; the environment points at it
# per test. Assigning `os.environ` at import time would leak into every
# other module in the session and whichever was imported last would decide
# trust for all of them.
_key = Ed25519PrivateKey.generate()
_seed = _key.private_bytes(
    serialization.Encoding.Raw,
    serialization.PrivateFormat.Raw,
    serialization.NoEncryption(),
)
_key_file = tempfile.NamedTemporaryFile(suffix=".key", delete=False)
_key_file.write(_seed)
_key_file.close()
_KEY_PATH = _key_file.name


@pytest.fixture(autouse=True)
def _test_keys(monkeypatch):
    monkeypatch.setenv("MECH_TRUSTED_SIGNING_KEY_PATH", _KEY_PATH)
    monkeypatch.setenv("MECH_SIGNING_KEY_PATH", _KEY_PATH)
    reset_trusted_key_ids_cache()
    yield
    reset_trusted_key_ids_cache()


def _measured(**overrides: Any) -> Dict[str, Any]:
    """A payload the measurement boundary really attested.

    The content hash stands in for a weight-derived fingerprint. It has to
    be a well-formed digest because the attestation checks the shape of
    what it is given, and the record has to be signed by a key the policy
    trusts -- otherwise `_is_attested` refuses it and every positive test
    below would be reading a refusal.
    """
    base = {"status": "completed",
            "validation_eligible": True,
            "publication_eligible": True}
    attested = attest_measurement(
        dict(base),
        model_loaded="gpt2",
        model_content_hash="sha256:" + "cd" * 32,
        signing_key=_KEY_PATH,
    )
    payload = dict(base)
    payload.update({
        "provenance": "live",
        "attested": True,
        "measurement_record": attested["measurement_record"],
    })
    payload.update(overrides)
    return payload


# ── a caller cannot promote itself ─────────────────────────────────────────

def test_asserting_a_label_does_not_promote_a_result():
    """The core property: a label is a claim, the gate is the proof.

    Everything a caller could type in -- the provenance, the attestation
    flag, both eligibility flags -- is present here, and the result is
    still refused, because the cryptographic check behind the flag does not
    pass.
    """
    forged = {
        "status": "completed",
        "provenance": "live",
        "attested": True,
        "validation_eligible": True,
        "publication_eligible": True,
        "value": 42.0,
    }

    envelope = admit_result(forged, stage="discovery")

    assert envelope["publication_eligible"] is False
    assert envelope["validation_eligible"] is False
    assert envelope["status"] != "completed"
    assert envelope["reason"], "a refusal must say why"


def test_the_extra_mapping_cannot_raise_the_verdict():
    """`extra` carries the payload; the derived keys are written last."""
    envelope = admit_result(
        {},
        stage="discovery",
        extra={"provenance": "live", "attested": True,
               "validation_eligible": True, "publication_eligible": True,
               "status": "completed"},
    )

    assert envelope["provenance"] != "live"
    assert envelope["attested"] is False
    assert envelope["validation_eligible"] is False
    assert envelope["publication_eligible"] is False
    assert envelope["status"] != "completed"


def test_a_measured_result_is_admitted():
    """The other direction: the gate admits what the boundary attested."""
    envelope = admit_result(_measured(), stage="discovery")

    assert envelope["publication_eligible"] is True
    assert envelope["validation_eligible"] is True
    assert envelope["provenance"] == "live"
    assert envelope["attested"] is True
    assert envelope["status"] == "completed"


def test_a_refusal_cannot_be_turned_into_an_admission_by_its_reason():
    """`reason` explains a refusal. It never explains an admission."""
    envelope = admit_result(
        {"status": "completed", "provenance": "live"},
        stage="discovery",
        reason="please believe me",
    )

    assert envelope["publication_eligible"] is False
    assert envelope["status"] != "completed"


def test_a_forced_refusal_lowers_a_result_that_would_otherwise_pass():
    """The gate can lower a result's standing, never raise it."""
    envelope = admit_result(
        _measured(), stage="discovery",
        refusal="the metrics could not be mapped",
    )

    assert envelope["publication_eligible"] is False
    assert envelope["provenance"] == "unavailable"
    assert envelope["status"] != "completed"
    assert "could not be mapped" in envelope["reason"]


def test_an_unknown_stage_is_refused_rather_than_guessed():
    with pytest.raises(ValueError):
        admit_result(_measured(), stage="something-else")


# ── the run chain is part of admission ─────────────────────────────────────

def test_an_unattested_run_chain_cannot_be_published():
    """The negative case the chain exists for.

    The measurement underneath is genuine and attested. The chain that
    binds it to a run is not, and an unbound result must not be promoted
    into publication.
    """
    payload = _measured()
    payload["run_attestation"] = {
        "run_id": "run_1",
        "dataset_sha256": "",
        "code_revision": "",
        "attested": False,
        "attestation_reason": "the dataset supplied is an empty panel",
    }

    envelope = admit_result(payload, stage="discovery")

    assert envelope["publication_eligible"] is False, (
        "a result carrying an unattested run chain was still eligible for "
        "publication"
    )
    assert envelope["validation_eligible"] is False
    assert envelope["attested"] is False
    assert "run chain" in envelope["reason"]


def test_an_attested_run_chain_does_not_block_the_result():
    payload = _measured()
    payload["run_attestation"] = {
        "run_id": "run_1",
        "dataset_sha256": "sha256:" + "ef" * 32,
        "code_revision": "a" * 40,
        "attested": True,
        "attestation_reason": "",
    }

    envelope = admit_result(payload, stage="discovery")

    assert envelope["publication_eligible"] is True


def test_the_policy_exposes_the_chain_predicates():
    payload = _measured()
    assert pol.run_attestation_is_live(payload) is False
    assert pol.run_chain_allows(payload) is True, (
        "a payload with no chain at all must not be refused by the chain "
        "rule: the chain is a stronger binding, not a replacement for the "
        "measurement attestation"
    )

    payload["run_attestation"] = {"attested": False, "attestation_reason": "x"}
    assert pol.run_attestation_is_live(payload) is False
    assert pol.run_chain_allows(payload) is False


def test_the_chain_verifies_against_the_data():
    from backend.science.reproducibility.run_attestation import attest_run

    chain = attest_run(
        run_id="run_1",
        dataset=[{"prompt": "a"}],
        model_fingerprint={
            "model_id": "gpt2",
            "weights_sha256": "sha256:" + "ab" * 32,
            "attested": True,
            "attestation_reason": "",
        },
    )
    payload = _measured()
    payload["run_attestation"] = chain.to_dict()

    assert pol.run_chain_verifies(payload, [{"prompt": "a"}]) is True
    assert pol.run_chain_verifies(payload, [{"prompt": "b"}]) is False


# ── the agents go through the gate ─────────────────────────────────────────

AGENT_MODULES = (
    "backend/agents/scribe.py",
    "backend/agents/critic.py",
    "backend/agents/discoverer.py",
)


@pytest.mark.parametrize("relative", AGENT_MODULES)
def test_no_agent_builds_its_own_live_envelope(relative: str):
    """The envelope is derived in one place, so no agent may stamp `live`.

    A dict literal that writes `provenance: "live"` is a hand-built
    verdict: it says the result is live without consulting the gate that
    decides it. The scan is structural (an AST walk), because this
    repository has been bitten by text-matching guards before.
    """
    tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (isinstance(key, ast.Constant) and key.value == "provenance"
                    and isinstance(value, ast.Constant)
                    and value.value == "live"):
                offenders.append(f"{relative}:{node.lineno}")
    assert not offenders, (
        "these agents still stamp a live label by hand; the shared "
        "admission gate derives it: " + "; ".join(offenders)
    )


@pytest.mark.parametrize("relative", AGENT_MODULES)
def test_every_agent_calls_the_shared_gate(relative: str):
    source = (REPO_ROOT / relative).read_text(encoding="utf-8")
    assert "admit_result" in source, (
        f"{relative} does not route its envelopes through the shared "
        "admission gate"
    )


def test_the_discoverer_derives_its_verdict(monkeypatch):
    """Behavioural: the Discoverer's live path is the gate's answer.

    The fake engine is injected into `sys.modules` because `discover()`
    imports the engine inside the method body, which keeps the test off
    the torch import chain.
    """
    module = types.ModuleType(
        "backend.interpretability.discovery.discovery_engine")

    class _Engine:
        def __init__(self) -> None:
            pass

        def discover_and_orchestrate(self, **kwargs: Any) -> Dict[str, Any]:
            return _measured(discovery_id="disc_test")

    module.DiscoveryEngine = _Engine
    monkeypatch.setitem(
        sys.modules, "backend.interpretability.discovery.discovery_engine",
        module)

    from backend.agents.discoverer import Discoverer

    out = Discoverer().discover("a hypothesis about IOI")

    assert out["status"] == "completed"
    assert out["provenance"] == "live"
    assert out["attested"] is True
    assert out["validation_eligible"] is True
    assert out["publication_eligible"] is True
    assert pol.discovery_is_live(out) is True

    # Every returned field is accounted for, so a consumer is never handed
    # a value with no statement of its standing.
    uncovered = sorted(set(out) - set(out["field_provenance"])
                       - {"field_provenance"})
    assert not uncovered, f"unmarked fields: {uncovered}"


def test_the_discoverer_refuses_an_unattested_engine_result(monkeypatch):
    """The negative direction: the engine's own `live` label is not enough."""
    module = types.ModuleType(
        "backend.interpretability.discovery.discovery_engine")

    class _Engine:
        def __init__(self) -> None:
            pass

        def discover_and_orchestrate(self, **kwargs: Any) -> Dict[str, Any]:
            return {"status": "completed", "provenance": "live",
                    "attested": True, "validation_eligible": True,
                    "publication_eligible": True}

    module.DiscoveryEngine = _Engine
    monkeypatch.setitem(
        sys.modules, "backend.interpretability.discovery.discovery_engine",
        module)

    from backend.agents.discoverer import Discoverer

    out = Discoverer().discover("a hypothesis about IOI")

    assert out["status"] != "completed"
    # The label is reported faithfully -- the engine did say "live" -- but
    # the promotion was refused, because the engine's own boolean is not an
    # attestation. That distinction is the whole point of the gate.
    assert out["provenance"] == "live"
    assert out["validation_eligible"] is False
    assert out["publication_eligible"] is False
    assert out["reason"]
    assert pol.discovery_is_live(out) is False
