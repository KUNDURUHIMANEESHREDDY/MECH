"""P0 #1 -- the LLM engines must be fail-closed and must never emit canned text.

Claim under test: every failure/unreachable path in
``backend/agents/llm/ollama_engine.py`` and ``openai_engine.py`` returns a
``status``/``reason`` and NEVER canned text.

What this module tries to break:

* Reach a hard-coded hypothesis/evaluation/critique string without a live
  server.  The engines' ``_mock_*`` methods are checked directly *and* through
  every public entry point with the transport monkeypatched to fail.
* Get an exception converted into a successful result -- a fabricated
  ``type: supportive``, a non-empty ``critique``, a non-empty ``prompt``.
* Get ``provenance`` reported as ``live`` on a path that never reached a model.

The transport is patched, not the engine, so the failure paths are exercised as
written.  Nothing here requires a server, a token, or a network.
"""

from __future__ import annotations

import sys
import urllib.error
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.agents.llm.engine_base import LLMEngine  # noqa: E402
from backend.agents.llm.ollama_engine import OllamaEngine  # noqa: E402
from backend.agents.llm.openai_engine import OpenAIEngine  # noqa: E402

#: Statuses that mean "this did not happen". Anything else on a refusal path is
#: a claim that something ran.
NON_LIVE_STATUSES = {"unavailable", "not_executed", "error", "failed", "refused"}

#: The marker the project uses for "not measured".
NON_LIVE_PROVENANCE = {"unavailable", "not_executed", "reference", "error"}


def _fail_transport(monkeypatch, exc: BaseException):
    """Make every urlopen raise `exc` -- i.e. no server is reachable."""
    def _boom(*_args, **_kwargs):
        raise exc

    monkeypatch.setattr("urllib.request.urlopen", _boom)


def _both_engines():
    yield "ollama", OllamaEngine()
    yield "openai", OpenAIEngine()


# --------------------------------------------------------------------------- #
# the public entry points, with no server reachable
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("engine_name,engine", list(_both_engines()),
                         ids=["ollama", "openai"])
def test_unreachable_hypotheses_are_refused_not_invented(monkeypatch, engine_name,
                                                         engine):
    """No server -> a refusal object. Never a fabricated description."""
    _fail_transport(monkeypatch, urllib.error.URLError("connection refused"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    out = engine.generate_hypotheses({"n_grams": [["the", 3]]}, max_candidates=3)

    assert isinstance(out, list) and out, "must return a stated refusal"
    # Every returned candidate must be a refusal, not a hypothesis.
    for candidate in out:
        assert isinstance(candidate, dict)
        assert candidate.get("status") in NON_LIVE_STATUSES, candidate
        assert str(candidate.get("reason") or "").strip(), (
            f"{engine_name} returned a refusal with no reason: {candidate}"
        )
        assert candidate.get("provenance") in NON_LIVE_PROVENANCE, candidate
        # The two fields a consumer would read as a hypothesis must be absent.
        assert not candidate.get("description"), (
            f"{engine_name} fabricated a hypothesis description: {candidate}"
        )
        assert "initial_confidence" not in candidate or \
            candidate["initial_confidence"] is None, (
            f"{engine_name} attached a confidence to a refusal: {candidate}"
        )


@pytest.mark.parametrize("engine_name,engine", list(_both_engines()),
                         ids=["ollama", "openai"])
def test_unreachable_evaluation_is_not_supportive(monkeypatch, engine_name,
                                                  engine):
    """A failed evaluation must not come back as `supportive`."""
    _fail_transport(monkeypatch, urllib.error.URLError("connection refused"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    out = engine.evaluate_experiment("h", {"type": "supportive"})

    assert isinstance(out, dict)
    assert out.get("status") in NON_LIVE_STATUSES, out
    assert out.get("type") != "supportive", (
        f"{engine_name} reported a supportive evaluation without a model: {out}"
    )
    assert out.get("type") in (None, "inconclusive", "unavailable", "error"), out
    assert str(out.get("reason") or "").strip(), out


@pytest.mark.parametrize("engine_name,engine", list(_both_engines()),
                         ids=["ollama", "openai"])
def test_unreachable_critique_has_no_text(monkeypatch, engine_name, engine):
    """A critique that never reached a model must carry no critique text."""
    _fail_transport(monkeypatch, urllib.error.URLError("connection refused"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    out = engine.generate_critique("Fires on French cities")

    assert out.get("status") in NON_LIVE_STATUSES, out
    assert not str(out.get("critique") or "").strip(), (
        f"{engine_name} fabricated critique text: {out!r}"
    )
    assert not str(out.get("alternative_explanation") or "").strip(), out


@pytest.mark.parametrize("engine_name,engine", list(_both_engines()),
                         ids=["ollama", "openai"])
def test_unreachable_counterexample_has_no_prompt(monkeypatch, engine_name,
                                                  engine):
    """A counterexample prompt that was never generated must be empty."""
    _fail_transport(monkeypatch, urllib.error.URLError("connection refused"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    out = engine.generate_counterexample("h", "c")

    assert out.get("status") in NON_LIVE_STATUSES, out
    assert not str(out.get("prompt") or "").strip(), (
        f"{engine_name} fabricated a counterexample prompt: {out!r}"
    )
    assert out.get("expected_firing") is False, (
        f"{engine_name} asserted an expected firing without an experiment: {out}"
    )


@pytest.mark.parametrize("engine_name,engine", list(_both_engines()),
                         ids=["ollama", "openai"])
def test_unreachable_review_verdict_is_inconclusive(monkeypatch, engine_name,
                                                    engine):
    """The peer-review path must not return `accepted` from a failed call."""
    _fail_transport(monkeypatch, urllib.error.URLError("connection refused"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    out = engine.review_evidence("h", "c", {"prompt": "x",
                                            "expected_firing": True},
                                 {"type": "supportive"})

    assert out.get("status") in NON_LIVE_STATUSES, out
    assert out.get("verdict") != "accepted", (
        f"{engine_name} accepted a hypothesis with no reviewer: {out}"
    )
    assert str(out.get("reason") or "").strip(), out


# --------------------------------------------------------------------------- #
# the legacy `_mock_*` hooks: reachable without a live server
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("engine_name,engine", list(_both_engines()),
                         ids=["ollama", "openai"])
def test_mock_hooks_are_themselves_fail_closed(engine_name, engine):
    """The `_mock_*` methods are the classic reachable-without-a-server path.

    They must be refusals in their own right -- if any later code path calls
    them, it must not receive fabricated science.
    """
    assert isinstance(engine, LLMEngine), "engines must implement the interface"

    hyp = engine._mock_generate({"n_grams": []}, 3)
    assert isinstance(hyp, list) and hyp
    for candidate in hyp:
        assert candidate.get("status") in NON_LIVE_STATUSES, candidate
        assert str(candidate.get("reason") or "").strip(), candidate
        assert not candidate.get("description"), candidate

    ev = engine._mock_evaluate("h", {})
    assert ev.get("status") in NON_LIVE_STATUSES, ev
    assert ev.get("type") != "supportive", ev
    assert not str(ev.get("rationale") or "").strip() or \
        "disabled" in str(ev.get("rationale")).lower(), ev

    ce = engine._mock_counterexample("h")
    assert ce.get("status") in NON_LIVE_STATUSES, ce
    assert not str(ce.get("prompt") or "").strip(), ce


# --------------------------------------------------------------------------- #
# the openai engine without a key is a refusal before any network call
# --------------------------------------------------------------------------- #

def test_openai_without_key_refuses_without_network(monkeypatch):
    """No API key -> refusal, and the transport must not even be touched."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    calls = []

    def _record(*args, **kwargs):  # pragma: no cover - must not run
        calls.append((args, kwargs))
        raise AssertionError("network accessed without an API key")

    monkeypatch.setattr("urllib.request.urlopen", _record)

    engine = OpenAIEngine()
    hyp = engine.generate_hypotheses({}, 3)
    assert hyp[0]["status"] in NON_LIVE_STATUSES, hyp
    assert "OPENAI_API_KEY" in str(hyp[0].get("reason")), hyp
    assert calls == [], f"the engine hit the network with no key: {calls}"


# --------------------------------------------------------------------------- #
# a live-but-unparseable response is still a refusal
# --------------------------------------------------------------------------- #

class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize("payload", [
    b"not json at all",
    b"{}",
    b'{"choices": []}',
    b'{"choices": [{"message": {"content": "I think it fires on French cities"}}]}',
], ids=["garbage", "empty-object", "no-choices", "prose-not-json"])
def test_malformed_or_prose_response_is_not_turned_into_a_hypothesis(
        monkeypatch, payload):
    """A 200 that cannot be parsed must degrade to a refusal, not pass through.

    The last case is the dangerous one: a plausible sentence, but not the JSON
    contract the prompt demands. Returning it as a hypothesis would be exactly
    the "plausible-looking payload" the project forbids.
    """
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: _FakeResponse(payload))

    out = OllamaEngine().generate_hypotheses({"n_grams": []}, 3)

    assert isinstance(out, list) and out
    for candidate in out:
        assert candidate.get("status") in NON_LIVE_STATUSES, (
            f"unparseable response was accepted as a hypothesis: {candidate}"
        )
        assert not candidate.get("description"), candidate


def test_parse_json_response_strips_markdown_fences_only():
    """The parser is a transport detail; check it does not invent structure."""
    engine = OllamaEngine()
    assert engine._parse_json_response('```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert engine._parse_json_response('```\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(Exception):
        engine._parse_json_response("plain prose, no JSON")


# --------------------------------------------------------------------------- #
# the transport wrapper names the endpoint it failed on
# --------------------------------------------------------------------------- #

def test_ollama_call_api_error_names_its_endpoint(monkeypatch):
    """The raised error must be actionable rather than a bare URLError."""
    engine = OllamaEngine(endpoint="http://127.0.0.1:1/v1/chat/completions")
    _fail_transport(monkeypatch, urllib.error.URLError("refused"))

    with pytest.raises(RuntimeError) as excinfo:
        engine._call_api("hi")
    assert "Ollama" in str(excinfo.value)
    assert str(engine.endpoint) in str(excinfo.value), (
        "the failure message must name the endpoint it could not reach"
    )


def test_no_canned_hypothesis_literals_in_the_engines():
    """Source guard: the removed canned strings must not come back.

    Scanned with the module docstrings retained, because these are prose
    literals a future edit would paste straight back in.
    """
    banned = (
        "Fires on French cities",
        "firing on the token",
        "This feature appears to represent",
    )
    for path in (REPO_ROOT / "backend" / "agents" / "llm" / "ollama_engine.py",
                 REPO_ROOT / "backend" / "agents" / "llm" / "openai_engine.py"):
        text = path.read_text(encoding="utf-8")
        for literal in banned:
            assert literal not in text, f"{path.name} reintroduced: {literal!r}"
