"""The LLM engines fail closed: a missing key or an unreachable backend is never
converted into scientific-sounding content.

Why this file exists
--------------------
`openai_engine.py` and `ollama_engine.py` used to answer every failure with a
`_mock_*` payload. With `OPENAI_API_KEY` unset -- the default on a developer
machine -- `generate_hypotheses` silently returned two invented hypotheses at
confidences `0.65` / `0.45`, and `_mock_evaluate` returned
`{"type": "supportive"}`: a machine-readable *positive experimental verdict*
manufactured from nothing, carrying no provenance field, so the platform's own
evidence gate could not tell it from a real LLM answer.

That is the exact failure AGENTS.md names: "An exception is never converted into
a successful result." The same shape appeared in `_OfflineLLMEngine`, which
`AutoHypothesisTester.test_hypothesis` used to run a "falsification loop" against
hardcoded literals and then report a `confidence` derived from them.

This module pins the replacement contract: unavailable input produces an
unavailable envelope that names its reason and carries `provenance`, and no code
path can reach a supportive/confirmed verdict without a measurement.
"""

from __future__ import annotations

import ast
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from _weight_guard import skip_reason, weights_available  # noqa: E402

from backend.agents.llm.ollama_engine import OllamaEngine  # noqa: E402
from backend.agents.llm.openai_engine import OpenAIEngine  # noqa: E402

needs_weights = pytest.mark.skipif(not weights_available(), reason=skip_reason())

#: Every method that must fail closed, mapped to the engine it belongs to.
FAIL_CLOSED_METHODS = (
    "generate_hypotheses",
    "evaluate_experiment",
    "generate_critique",
    "generate_counterexample",
    "review_evidence",
)

#: The literals the old mocks returned. None may appear in a returned payload.
FABRICATED_LITERALS = (
    "French locations",
    "Python syntax",
    "Paris Hilton",
    "Mock evaluation",
    "Might be spurious",
    "Mock reviewer",
)


def _dead_ollama() -> OllamaEngine:
    """An engine pointed at a port nothing is listening on."""
    return OllamaEngine(endpoint="http://127.0.0.1:9/v1/chat/completions")


def _call(engine, method: str):
    """Invoke a fail-closed method with arguments valid for every signature."""
    if method == "generate_hypotheses":
        return engine.generate_hypotheses({"feature_index": 1402})
    if method == "evaluate_experiment":
        return engine.evaluate_experiment("a hypothesis", {"type": "supportive"})
    if method == "generate_critique":
        return engine.generate_critique("a hypothesis")
    if method == "generate_counterexample":
        return engine.generate_counterexample("a hypothesis", "a critique")
    if method == "review_evidence":
        return engine.review_evidence(
            "a hypothesis", "a critique", {"prompt": "p", "expected_firing": True},
            {"type": "supportive"},
        )
    raise AssertionError(method)


def _assert_unavailable(payload, method: str):
    """A returned envelope that cannot be mistaken for a measurement."""
    assert payload is not None, f"{method} returned None"
    items = payload if isinstance(payload, list) else [payload]
    assert items, f"{method} returned an empty sequence"
    for item in items:
        assert isinstance(item, dict), f"{method} returned {type(item)}"
        assert item.get("provenance") == "unavailable", (
            f"{method} returned no unavailable provenance: {item!r}")
        assert item.get("reason"), f"{method} returned no reason: {item!r}"
        rendered = repr(item)
        for literal in FABRICATED_LITERALS:
            assert literal not in rendered, (
                f"{method} still returns the fabricated literal {literal!r}")
        # A fabricated confidence is the specific thing that made the old
        # payloads look like results.
        for key in ("initial_confidence", "confidence"):
            assert key not in item, (
                f"{method} invented a {key}: {item!r}")


@pytest.mark.parametrize("method", FAIL_CLOSED_METHODS)
def test_openai_engine_fails_closed_without_a_key(method, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    engine = OpenAIEngine()
    _assert_unavailable(_call(engine, method), method)


@pytest.mark.parametrize("method", FAIL_CLOSED_METHODS)
def test_ollama_engine_fails_closed_when_unreachable(method):
    _assert_unavailable(_call(_dead_ollama(), method), method)


def test_unavailable_evaluation_is_not_a_supportive_verdict(monkeypatch):
    """The specific regression: a failed call must not read as a positive result."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = OpenAIEngine().evaluate_experiment("h", {"type": "supportive"})
    assert result.get("type") != "supportive", result
    assert result.get("type") == "inconclusive", result

    ollama = _dead_ollama().evaluate_experiment("h", {"type": "supportive"})
    assert ollama.get("type") != "supportive", ollama
    assert ollama.get("type") == "inconclusive", ollama


def test_unavailable_review_is_not_an_acceptance(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    verdict = OpenAIEngine().review_evidence(
        "h", "c", {"prompt": "p", "expected_firing": True}, {"type": "supportive"})
    assert verdict.get("verdict") != "accepted", verdict


def test_no_mock_method_is_reachable_from_the_public_api(monkeypatch):
    """The `_mock_*` stubs may exist, but nothing may call them.

    They are retained only as documented fail-closed stubs for any external
    caller that imported them; the engines must not route to them.
    """
    for rel in ("backend/agents/llm/openai_engine.py",
                "backend/agents/llm/ollama_engine.py"):
        src = open(os.path.join(REPO_ROOT, rel), encoding="utf-8").read()
        tree = ast.parse(src)
        offenders = [
            node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr.startswith("_mock_")
        ]
        assert not offenders, f"{rel} still calls a _mock_* method at {offenders}"


def test_no_silent_fallback_print_survives():
    """The old path logged a warning and returned fabrication; it is gone."""
    for rel in ("backend/agents/llm/openai_engine.py",
                "backend/agents/llm/ollama_engine.py"):
        src = open(os.path.join(REPO_ROOT, rel), encoding="utf-8").read()
        assert "Falling back to mock" not in src, rel


def test_hypothesis_tester_does_not_confirm_from_hardcoded_hypotheses(monkeypatch):
    """`test_hypothesis` must not mint a verdict out of literals.

    Previously it ran its loop on `_OfflineLLMEngine`, whose hypotheses were two
    fixed strings, and returned `confidence` from them. With no usable LLM the
    engine must report unavailable rather than deriving a confidence.
    """
    from backend.interpretability.discovery import auto_hypothesis_tester as aht

    source = open(aht.__file__, encoding="utf-8").read()
    assert "_OfflineLLMEngine()" not in source, (
        "test_hypothesis still constructs the hardcoded offline engine")

    tester = aht.AutoHypothesisTester()
    # No LLM engine configured: generate() must not invent candidates.
    candidates = tester.generate("Feature_1", {"feature_index": 1}, aht.ReasoningTrace(
        observation_id="Feature_1", llm_model="none", timestamp=0.0))
    assert candidates == [], candidates


def test_skeptic_critique_is_not_labelled_live_when_nothing_measured_it():
    """A critique is a judgement, not a measurement, so it is not `live`.

    The trace entry used to default the provenance to 'live' whenever the
    LLM engine answered without a provenance of its own -- an answer nobody
    measured, labelled as though it were.
    """
    from backend.interpretability.discovery import auto_hypothesis_tester as aht

    class _Engine:
        model = "stub"

        def generate_critique(self, hypothesis):
            return {"critique": "the claim may be vacuous",
                    "alternative_explanation": "none"}

    tester = aht.AutoHypothesisTester(engine=_Engine())
    trace = aht.ReasoningTrace(observation_id="Feature_1", llm_model="stub",
                                timestamp=0.0)
    hypothesis = aht.Hypothesis(id="H_1", target="Feature_1", description="d")
    tester.critique(hypothesis, trace)

    entry = trace.experiments[-1]
    assert entry["type"] == "skeptic_critique"
    assert entry["provenance"] != "live", entry
    assert entry["provenance"] == "reference", entry


def test_skeptic_critique_keeps_the_provenance_the_engine_supplied():
    """A label the engine did supply is the engine's to give."""
    from backend.interpretability.discovery import auto_hypothesis_tester as aht

    class _Engine:
        model = "stub"

        def generate_critique(self, hypothesis):
            return {"critique": "", "status": "unavailable",
                    "provenance": "unavailable",
                    "reason": "no LLM configured"}

    tester = aht.AutoHypothesisTester(engine=_Engine())
    trace = aht.ReasoningTrace(observation_id="Feature_1", llm_model="stub",
                                timestamp=0.0)
    hypothesis = aht.Hypothesis(id="H_1", target="Feature_1", description="d")
    tester.critique(hypothesis, trace)

    assert trace.experiments[-1]["provenance"] == "unavailable"


@needs_weights
def test_hypothesis_verdict_is_measured_against_a_real_forward_pass():
    """With weights present the verdict carries live provenance and evidence.

    A true statement about the model's own prediction is confirmed; a false one
    is refuted. Both are measured, and the confidence is a probability the model
    actually assigned -- not a constant.
    """
    from backend.interpretability.discovery.auto_hypothesis_tester import (
        AutoHypothesisTester,
    )

    tester = AutoHypothesisTester()

    confirmed = tester.test_hypothesis(
        "When given 'The capital of France is', the model predicts ' the'")
    assert confirmed["provenance"] == "live", confirmed
    assert confirmed["measured"] is True, confirmed
    assert confirmed["outcome_state"] == "Confirmed", confirmed
    assert confirmed["evidence"], confirmed
    assert confirmed["evidence"][0]["prompt"] == "The capital of France is", confirmed

    refuted = tester.test_hypothesis(
        "When given 'The capital of France is', the model predicts ' Tokyo'")
    assert refuted["provenance"] == "live", refuted
    assert refuted["measured"] is True, refuted
    assert refuted["outcome_state"] == "Refuted", refuted
    assert refuted["passed"] is False, refuted

    # The two verdicts must not share a hardcoded confidence.
    assert confirmed["confidence"] != refuted["confidence"] or (
        confirmed["outcome_state"] != refuted["outcome_state"])
