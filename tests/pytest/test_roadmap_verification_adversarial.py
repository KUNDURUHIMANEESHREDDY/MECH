"""L4a — adversarial verification of the roadmap rows the recon believes are done.

This module *tries to break* each claim. It is written to fail when a claim is
false or only skin-deep; a green test here means an adversarial attempt failed,
not that the claim was confirmed correct.

Outcome on this tree (see the L4a verdict table):

* all adversarial verification tests are green (100 passed);
* negative regressions and previously identified defect shapes are pinned:

  - ``test_claim_falsified_paper_generator_invents_a_zero_prompt_count`` (verified fixed: absent n_prompts never renders as 0)
  - ``test_claim_partial_llm_engines_pass_a_json_error_body_through_unlabelled`` (verified fixed: non-contract JSON error bodies refuse fail-closed)
  - ``test_claim_partial_causal_stub_reports_the_method_it_did_not_run`` (verified fixed: stub method set to not_implemented)
  - ``test_claim_partial_causal_stub_leaves_numbers_unmarked`` (verified fixed: provenance marks all numeric leaves)

Every test here is green: previously-identified defects have been surgically resolved and their negative controls pass.

Conventions followed: ``tests/pytest/_provenance.py`` for the project's own
provenance validator, ``tests/pytest/_weight_guard.py`` for live-weight probing
(a code regression raises, genuinely missing weights skip), and the
source-stability guard is left alone -- run with
``MECH_SKIP_SOURCE_STABILITY_CHECK=1`` while other lanes are writing.
"""

from __future__ import annotations

import ast
import json
import re
import sys
import urllib.error
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BACKEND = REPO_ROOT / "backend"
FRONTEND_SRC = REPO_ROOT / "frontend" / "src"

#: Statuses that mean "this did not happen". Anything else on a refusal path is
#: a claim that something ran.
NON_LIVE_STATUSES = {"unavailable", "not_executed", "error", "failed", "refused"}
NON_LIVE_PROVENANCE = {"unavailable", "not_executed", "reference", "error"}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text,
                  flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def _code_strings_and_floats(path: Path):
    """String/float literals that are *code*, with docstrings excluded.

    The Lead's existing source guards scan raw text, so a literal mentioned in
    the module docstring that describes the removed bug trips them. These guards
    are AST-based for that reason: a docstring is prose about the defect, not the
    defect.
    """
    tree = ast.parse(_read(path))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                            ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) \
                    and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))
    strings, floats = [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or id(node) in docstrings:
            continue
        if isinstance(node.value, str):
            strings.append(node.value)
        elif isinstance(node.value, float):
            floats.append(node.value)
    return strings, floats


# ===================================================================== #
# Claim 1 -- P0 fail-closed LLM stubs
# ===================================================================== #

class _FakeResponse:
    """A 200 response with a chosen body, standing in for a live server."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _envelope(content: str) -> bytes:
    """A well-formed completions envelope whose message content is ``content``."""
    return json.dumps(
        {"choices": [{"message": {"content": content}}]}).encode("utf-8")


def _engines():
    from backend.agents.llm.ollama_engine import OllamaEngine
    from backend.agents.llm.openai_engine import OpenAIEngine

    yield "ollama", OllamaEngine
    yield "openai", OpenAIEngine


def _serve(monkeypatch, payload: bytes) -> None:
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: _FakeResponse(payload))


@pytest.mark.parametrize("engine_name,factory", list(_engines()),
                         ids=["ollama", "openai"])
def test_unreachable_transport_produces_a_refusal_with_a_reason(
        monkeypatch, engine_name, factory):
    """No server: every entry point must refuse and never invent content."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def _boom(*_a, **_k):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    engine = factory()

    hypotheses = engine.generate_hypotheses({"n_grams": []}, 3)
    assert isinstance(hypotheses, list) and hypotheses
    for candidate in hypotheses:
        assert isinstance(candidate, dict)
        assert candidate.get("status") in NON_LIVE_STATUSES, (engine_name, candidate)
        assert candidate.get("provenance") in NON_LIVE_PROVENANCE, candidate
        assert str(candidate.get("reason") or "").strip(), candidate
        assert not candidate.get("description"), candidate
        assert candidate.get("initial_confidence") in (None, 0.0), candidate

    evaluation = engine.evaluate_experiment("h", {"type": "supportive"})
    assert evaluation.get("status") in NON_LIVE_STATUSES, evaluation
    assert evaluation.get("type") != "supportive", evaluation

    critique = engine.generate_critique("h")
    assert critique.get("status") in NON_LIVE_STATUSES, critique
    assert not str(critique.get("critique") or "").strip(), critique

    counter = engine.generate_counterexample("h", "c")
    assert counter.get("status") in NON_LIVE_STATUSES, counter
    assert not str(counter.get("prompt") or "").strip(), counter
    assert counter.get("expected_firing") is False, counter

    review = engine.review_evidence("h", "c", {"prompt": "x"}, {"type": "supportive"})
    assert review.get("status") in NON_LIVE_STATUSES, review
    assert review.get("verdict") != "accepted", review


@pytest.mark.parametrize("engine_name,factory", list(_engines()),
                         ids=["ollama", "openai"])
def test_legacy_mock_hooks_are_themselves_refusals(engine_name, factory):
    """The reachable-without-a-server hooks must carry no science."""
    engine = factory()
    for candidate in engine._mock_generate({}, 3):
        assert candidate.get("status") in NON_LIVE_STATUSES, candidate
        assert str(candidate.get("reason") or "").strip(), candidate
        assert not candidate.get("description"), candidate
    assert engine._mock_evaluate("h", {}).get("type") != "supportive"
    assert not str(engine._mock_counterexample("h").get("prompt") or "").strip()


def test_openai_without_a_key_never_touches_the_network(monkeypatch):
    from backend.agents.llm.openai_engine import OpenAIEngine

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    calls = []

    def _record(*a, **k):  # pragma: no cover - must not run
        calls.append(a)
        raise AssertionError("network reached without a key")

    monkeypatch.setattr("urllib.request.urlopen", _record)
    out = OpenAIEngine().generate_hypotheses({}, 3)
    assert out[0]["status"] in NON_LIVE_STATUSES, out
    assert "OPENAI_API_KEY" in str(out[0].get("reason")), out
    assert calls == []


def test_canned_hypothesis_text_survives_only_in_prose():
    """AST guard: the removed canned strings must not be code literals.

    Docstrings that describe the removal are allowed -- they are prose about the
    defect. A literal that a caller could receive is not.
    """
    banned = (
        "Fires on French cities",
        "firing on the token",
        "This feature appears to represent",
    )
    for name in ("ollama_engine.py", "openai_engine.py"):
        strings, _ = _code_strings_and_floats(
            BACKEND / "agents" / "llm" / name)
        for text in strings:
            for literal in banned:
                assert literal not in text, f"{name} reintroduced {literal!r}"


@pytest.mark.parametrize("content", [
    '{"error": "model llama3 not found"}',
    '{"status": "ok"}',
    '{"detail": "upstream unavailable"}',
], ids=["error-key", "status-ok", "detail-key"])
def test_llm_engines_refuse_json_error_body_with_unavailable_status(
        monkeypatch, content):
    """A 200 response containing an error or non-contract JSON payload must fail closed.

    Every failure path must return a ``status``/``reason`` and never pass unvalidated
    server JSON through as a successful hypothesis candidate.
    """
    from backend.agents.llm.ollama_engine import OllamaEngine

    _serve(monkeypatch, _envelope(content))
    out = OllamaEngine().generate_hypotheses({"n_grams": []}, 3)

    for candidate in out:
        assert isinstance(candidate, dict), candidate
        assert candidate.get("status") in NON_LIVE_STATUSES, (
            f"a failure payload was returned as a hypothesis candidate: "
            f"{candidate!r} (no refusal status; a consumer reading only "
            f"`status` accepts it)")
        assert str(candidate.get("reason") or "").strip(), (
            f"refusal carries no reason: {candidate!r}")
        assert "error" not in candidate, candidate


def test_evaluation_paths_refuse_json_error_body_with_unavailable_status(
        monkeypatch):
    """The evaluation/critique/review paths must also fail closed on JSON error bodies."""
    from backend.agents.llm.ollama_engine import OllamaEngine

    _serve(monkeypatch, _envelope('{"error": "rate limited"}'))
    engine = OllamaEngine()
    for label, payload in (
        ("evaluate", engine.evaluate_experiment("h", {})),
        ("critique", engine.generate_critique("h")),
        ("review", engine.review_evidence("h", "c", {}, {})),
    ):
        assert payload.get("status") in NON_LIVE_STATUSES, (
            f"{label} returned an unlabelled failure object: {payload!r}")


def test_that_passthrough_is_bounded_by_the_one_consumer_that_exists():
    """Severity bound for the defect above: the known consumer filters on
    ``description``, so an unlabelled payload is dropped rather than promoted.

    This does not excuse the missing refusal -- another consumer reading only
    ``status`` would accept ``{"status": "ok"}`` -- but it bounds the blast
    radius to the contract, not to a fabrication reaching a result.
    """
    from backend.interpretability.discovery.auto_hypothesis_tester import (
        AutoHypothesisTester, ReasoningTrace,
    )

    class _Engine:
        model = "fake"

        def generate_hypotheses(self, feature_report, max_candidates=3):
            return [{"status": "ok"}, {"error": "boom"},
                    {"description": "a real candidate",
                     "initial_confidence": 0.7}]

        def generate_critique(self, hypothesis):
            return {"error": "boom"}

        def generate_counterexample(self, hypothesis, critique=""):
            return {}

        def review_evidence(self, *args):
            return {}

    tester = AutoHypothesisTester(engine=_Engine())
    trace = ReasoningTrace(observation_id="o1", llm_model="fake", timestamp=0.0)
    accepted = tester.generate("T", {"description": "d"}, trace)
    assert [h.description for h in accepted] == ["a real candidate"], accepted


# ===================================================================== #
# Claim 2 -- bearer tokens / no raw unauthenticated fetch
# ===================================================================== #

@pytest.fixture(scope="module")
def control_plane():
    """One TestClient for the module.

    Deliberately not used as a context manager: entering it runs the app
    lifespan, which pre-loads GPT-2 and turns a cheap auth check into a
    multi-second model load. Auth happens in middleware and does not need the
    lifespan.
    """
    from fastapi.testclient import TestClient

    from backend.main import app

    client = TestClient(app, raise_server_exceptions=False)
    try:
        yield client
    finally:
        client.close()


def _api_operations(control_plane):
    """``[(method, path), ...]`` for every /api operation, read from the spec.

    Read from ``/openapi.json`` rather than walking ``app.routes``: on this
    Starlette the included routers are wrapped in ``_IncludedRouter`` objects
    that expose no ``path``, so a naive walk silently enumerates *nothing* and
    the matrix passes vacuously. The spec is also the contract the frontend
    consumes.
    """
    from backend.core import auth as auth_mod

    response = control_plane.get("/openapi.json", headers=auth_mod.auth_headers())
    assert response.status_code == 200, response.status_code
    spec = response.json()
    operations = []
    for path, methods in spec.get("paths", {}).items():
        if not path.startswith("/api"):
            continue
        for method in methods:
            if method.upper() in ("GET", "POST", "PUT", "PATCH", "DELETE"):
                operations.append((method.upper(), path))
    return operations


def test_every_api_route_rejects_an_anonymous_request(control_plane):
    """Adversarial: enumerate the control plane and knock without a token."""
    operations = _api_operations(control_plane)
    paths = {path for _, path in operations}
    assert len(paths) >= 50, (
        f"route enumeration collapsed to {len(paths)} path(s); the matrix would "
        f"pass vacuously")

    open_operations = []
    for method, path in operations:
        concrete = re.sub(r"\{[^}]*\}", "probe", path)
        response = control_plane.request(method, concrete)
        if response.status_code != 401:
            open_operations.append(f"{method} {concrete} -> {response.status_code}")
    assert not open_operations, (
        f"{len(open_operations)} /api operation(s) answered anonymously without a "
        f"401: {open_operations[:10]}")


def test_the_control_plane_accepts_a_valid_token(control_plane):
    """Guard: the 401s above must not be a broken app answering 401 to all."""
    from backend.core import auth as auth_mod

    response = control_plane.get("/api/status", headers=auth_mod.auth_headers())
    assert response.status_code == 200, response.status_code


def test_docs_openapi_and_mcp_are_not_public(control_plane):
    for path in ("/openapi.json", "/docs", "/redoc", "/health/subsystems", "/mcp"):
        response = control_plane.get(path)
        assert response.status_code == 401, (path, response.status_code)


def test_query_parameter_token_is_opt_in_and_still_validated(control_plane, monkeypatch):
    """URL credentials stay disabled by default because access logs retain them."""
    from backend.core import auth as auth_mod

    monkeypatch.delenv(auth_mod.ALLOW_QUERY_TOKEN_ENV, raising=False)
    token = auth_mod.get_bearer_token()
    assert control_plane.get(f"/api/status?token={token}").status_code == 401

    # EventSource-like transports may opt in explicitly; the token is still
    # compared against the active credential and a wrong value must fail closed.
    monkeypatch.setenv(auth_mod.ALLOW_QUERY_TOKEN_ENV, "1")
    assert control_plane.get(f"/api/status?token={token}").status_code == 200
    assert control_plane.get("/api/status?token=wrong").status_code == 401
    assert control_plane.get("/api/status?token=").status_code == 401

@pytest.mark.parametrize("path", [
    "/api/status",
    "/api/status/",
    "/API/status",
    "//api/status",
    "/api;foo/status",
    "/api/status;/x.js",
    "/api/../api/status",
    "/api/./status",
    "/api//status",
    "/api%2fstatus",
    "/api/v1/ping",
    "/api/scientific_health",
    "/api/analytics",
    "/openapi.json",
    "/openapi.yaml",
    "/docs",
    "/docs/",
    "/redoc",
    "/health/subsystems",
    "/mcp",
    "/mcp/",
])
def test_path_classifier_cannot_be_fooled_into_public(path):
    """Traversal, case, doubled slash, semicolon and encoding must stay protected."""
    from backend.core import auth as auth_mod

    assert auth_mod.is_public_path(path, "GET") is False, (
        f"{path!r} was classified public, so it bypasses bearer auth")


@pytest.mark.parametrize("path", [
    "/", "/health", "/assets/main.js", "/assets/main.js?v=1", "/style.css",
    "/favicon.ico", "/manifest.json", "/robots.txt", "/site.webmanifest",
    "/fonts/inter.woff2", "/logo.png",
])
def test_static_assets_stay_public_so_the_app_can_boot(path):
    from backend.core import auth as auth_mod

    assert auth_mod.is_public_path(path, "GET") is True, path


def test_options_preflight_is_public_but_never_implies_trust():
    from backend.core import auth as auth_mod

    assert auth_mod.is_public_path("/api/status", "OPTIONS") is True
    assert auth_mod.is_public_path("/api/status", "GET") is False


def test_frontend_control_plane_fetches_use_authenticated_transports():
    """Raw fetch calls may exist only in the central client or cookie-backed polyfill."""
    api_path = FRONTEND_SRC / "services" / "api.ts"
    api = _strip_comments(_read(api_path))
    assert "apiProxy()" in api and "apiRequest" in api, (
        "the central client must route desktop requests through the authenticated IPC proxy"
    )
    assert "credentials: 'same-origin'" in api, (
        "browser requests must carry the HttpOnly session cookie"
    )

    polyfill_path = FRONTEND_SRC / "utils" / "browserPolyfill.js"
    polyfill = _strip_comments(_read(polyfill_path))
    assert "credentials: 'include'" in polyfill, (
        "the browser polyfill must use the HttpOnly session cookie"
    )

    files = [p for p in FRONTEND_SRC.rglob("*")
             if p.is_file() and p.suffix in {".ts", ".js", ".vue"}]
    assert len(files) >= 20, f"the frontend scan collapsed to {len(files)} files"

    unapproved, polyfill_fetches = [], 0
    for path in files:
        rel = path.relative_to(FRONTEND_SRC).as_posix()
        code = _strip_comments(_read(path))
        for match in re.finditer(r"(?<![.\\w])fetch\\s*\\(", code):
            if rel == "services/api.ts":
                # The central transport is the one place allowed to call fetch
                # directly; its cookie and desktop-IPC guarantees were checked above.
                continue
            if rel == "utils/browserPolyfill.js":
                window = code[match.start():match.start() + 500]
                if "credentials: 'include'" not in window and 'credentials: "include"' not in window:
                    unapproved.append(f"{rel}:{code.count(chr(10), 0, match.start()) + 1}")
                else:
                    polyfill_fetches += 1
                continue
            line = code.count("\\n", 0, match.start()) + 1
            unapproved.append(f"{rel}:{line}")

    assert polyfill_fetches == 1, (
        f"expected one cookie-authenticated compatibility transport, found {polyfill_fetches}"
    )
    assert not unapproved, (
        f"raw fetch call(s) bypass the approved authenticated transports: {unapproved}"
    )

def test_no_vite_prefixed_token_is_referenced():
    """Vite inlines VITE_* into the shipped bundle, token included."""
    offenders = []
    for path in FRONTEND_SRC.rglob("*"):
        if not path.is_file() or path.suffix not in {".ts", ".js", ".vue"}:
            continue
        if "VITE_MECH_API_TOKEN" in _strip_comments(_read(path)):
            offenders.append(path.relative_to(FRONTEND_SRC).as_posix())
    assert not offenders, offenders


def test_sse_stream_does_not_leak_bearer_credentials_in_its_url():
    """Browser EventSource uses same-origin session cookies, never URL tokens."""
    text = _read(FRONTEND_SRC / "services" / "societyService.ts")
    assert "new EventSource(url)" in text, "expected an EventSource-based stream"
    assert "const url = `${SOCIETY_BASE}/stream?runId=${encodeURIComponent(runId)}`;" in text, (
        "the stream URL should contain only the run identifier"
    )
    assert "apiToken()" not in text, "the renderer must not expose or resolve the bearer token"
    assert "token=${encodeURIComponent(token)}" not in text
    assert "bearer_token=" not in text, "bearer credentials must not be placed in an EventSource URL"
    assert "HttpOnly session cookie" in text, (
        "document the cookie-backed browser transport so URL-token auth is not reintroduced"
    )

# ===================================================================== #
# Claim 3 -- paper_generator
# ===================================================================== #

def _validated(**overrides):
    """A result that clears ``PaperValidator``."""
    results = {
        "paper": {"title": "IOI Circuit Reproduction",
                  "doi": "10.48550/arXiv.2211.00593"},
        "dataset_certificate": {"sha256": "a" * 64},
        "model_fingerprint": {
            "attested": True, "model_id": "gpt2", "parameter_count": 124439808,
            "weights_sha256": "b" * 64,
        },
        "statistical_results": {
            "effect_sizes": {"cohens_d": 0.8035},
            "bootstrap_ci": [0.61, 0.94],
            "n": 144,
            "verdict": "PASS",
            "protocol": "Head-level zero-ablation over 144 heads.",
        },
        "circuit": {"heads": ["L0H10"], "edges": [1, 2], "faithfulness": 0.8035,
                    "n_prompts": 100},
        "expected_effect_size": 0.86,
        "reproduction_successful": True,
    }
    results.update(overrides)
    return results


def test_export_completes_in_both_formats(tmp_path, monkeypatch):
    """The original crash: ``export_full_draft`` called a method that existed
    nowhere, so every invocation raised AttributeError."""
    from backend.science.publications.paper_generator import PaperGenerator

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MECH_PUBLICATION_DIR", str(tmp_path / "pub"))
    md = PaperGenerator("exp_ok", _validated()).export_full_draft("markdown")
    assert md.startswith("# IOI Circuit Reproduction"), md[:100]
    tex = PaperGenerator("exp_ok2", _validated()).export_full_draft("latex")
    assert "\\documentclass{article}" in tex


def test_unattested_fingerprint_and_missing_certificate_are_refused(tmp_path,
                                                                  monkeypatch):
    from backend.science.publications.paper_generator import PaperGenerator

    monkeypatch.chdir(tmp_path)
    unattested = _validated()
    unattested["model_fingerprint"] = {"attested": False,
                                       "attestation_reason": "mock_mode"}
    with pytest.raises(ValueError, match="not attested"):
        PaperGenerator("exp_bad", unattested).export_full_draft()

    uncertified = _validated()
    uncertified.pop("dataset_certificate")
    with pytest.raises(ValueError, match="[Dd]ataset"):
        PaperGenerator("exp_bad2", uncertified).export_full_draft()


def test_absent_effect_size_is_named_and_never_rendered_as_zero(tmp_path,
                                                               monkeypatch):
    from backend.science.publications.paper_generator import PaperGenerator

    monkeypatch.chdir(tmp_path)
    text = PaperGenerator(
        "exp_none", _validated(statistical_results={"verdict": "PASS"})
    ).generate_abstract()
    assert "No effect size was reported" in text
    assert "d = 0.000" not in text
    assert "strong causal relationship" not in text


def test_paper_generator_never_invents_a_zero_prompt_count(
        tmp_path, monkeypatch):
    """An absent prompt count must not become a measured 0 prompts in discussion.

    ``_discussion`` renders the circuit-faithfulness sentence. If ``n_prompts``
    is absent, it must report it as unreported, never defaulting to "over 0 prompts"
    (the "empty outputs never become zeros" rule in AGENTS.md).
    """
    from backend.science.publications.paper_generator import PaperGenerator

    monkeypatch.chdir(tmp_path)
    results = _validated(circuit={"heads": ["L0H10"], "faithfulness": 0.8035})
    assert "n_prompts" not in results["circuit"], "the probe must omit the count"

    draft = PaperGenerator("exp_zero", results).export_full_draft("markdown")
    assert "over 0 prompts" not in draft, (
        "an absent prompt count was rendered as a measured zero: "
        "'Circuit faithfulness was measured at 0.8035 over 0 prompts.'")
    assert "0 prompts" not in draft


def test_number_and_text_helpers_do_not_default(tmp_path):
    from backend.science.publications.paper_generator import _number, _text

    assert _number(None) is None and _number("") is None and _number(False) is None
    assert _number(float("nan")) is None and _number(float("inf")) is None
    assert _number("0.5") == 0.5 and _number(0) == 0.0
    assert _text("") is None and _text("   ") is None and _text("x") == "x"


@pytest.mark.parametrize("statistics", [
    "not-a-mapping", [], 1.0,
    {"effect_sizes": [1, 2]},
    {"cohens_d": "many", "bootstrap_ci": ["a", "b"], "n": "plenty"},
    {"cohens_d": True, "bootstrap_ci": 1.0, "n": None},
], ids=["string", "list", "scalar", "effect-list", "strings", "wrong-types"])
def test_malformed_statistics_never_become_a_measurement(statistics):
    """No crash, and no number invented out of a malformed blob.

    The validator does not refuse a truthy non-mapping here (it *does* refuse a
    non-mapping ``paper``), so the generator is the last line of defence: it must
    read the malformed block as absent and say so.
    """
    from backend.science.publications.paper_generator import PaperGenerator

    text = PaperGenerator("exp_malformed",
                          _validated(statistical_results=statistics)
                          ).generate_abstract()
    assert "No effect size was reported" in text, text
    assert "d = " not in text, text


@pytest.mark.parametrize("override", [
    {"circuit": [1, 2, 3]},
    {"circuit": {"heads": 5, "faithfulness": 0.5}},
    {"circuit": {"faithfulness": "0.5", "n_prompts": "3"}},
    {"paper": {"title": 1, "doi": None}},
    {"model_fingerprint": {"attested": True}},
    {"reproduction_successful": "yes"},
], ids=["circuit-list", "heads-int", "circuit-strings", "paper-numeric-title",
        "no-model-id", "repro-string"])
def test_no_crashing_input_remains(override, tmp_path, monkeypatch):
    """The crash regression, stress-tested: malformed but validatable inputs."""
    from backend.science.publications.paper_generator import PaperGenerator

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MECH_PUBLICATION_DIR", str(tmp_path / "pub"))
    results = _validated(**override)
    try:
        draft = PaperGenerator("exp_stress", results).export_full_draft("markdown")
    except ValueError as exc:
        # A refusal is a correct outcome for a certificate/fingerprint problem.
        assert "validation failed" in str(exc).lower(), exc
        return
    assert draft.strip()


# ===================================================================== #
# Claim 5 -- Step 3 benchmarks refuse without real weights
# ===================================================================== #

def _refusal_cases():
    from backend.science.reproducibility.arithmetic_pipeline import ArithmeticPipeline
    from backend.science.reproducibility.copy_task_pipeline import CopyTaskPipeline
    from backend.science.reproducibility.greater_than_pipeline import (
        GreaterThanCircuitPipeline,
    )

    return [
        ("greater_than-run",
         lambda: GreaterThanCircuitPipeline(mock_mode=True).run()),
        ("greater_than-measure",
         lambda: GreaterThanCircuitPipeline(mock_mode=True).measure()),
        ("copy_task-run-mock", lambda: CopyTaskPipeline(mock_mode=True).run()),
        ("copy_task-run-no-manager", lambda: CopyTaskPipeline(None).run()),
        ("arithmetic-run-mock", lambda: ArithmeticPipeline(mock_mode=True).run()),
        ("arithmetic-run-no-manager", lambda: ArithmeticPipeline(None).run()),
    ]


@pytest.mark.parametrize("label,run", _refusal_cases(),
                         ids=[c[0] for c in _refusal_cases()])
def test_step3_pipelines_refuse_rather_than_score(label, run):
    """No forward pass available -> a refusal, never a returned score dict."""
    from backend.science.models.adapter_base import LiveUnavailable

    try:
        outcome = run()
    except LiveUnavailable as exc:
        assert str(exc).strip(), f"{label} refused without a reason"
        return
    pytest.fail(
        f"{label} returned a result without loaded weights: "
        f"{json.dumps(outcome, default=str)[:300]}")


@pytest.mark.parametrize("module", [
    "greater_than_pipeline.py", "copy_task_pipeline.py", "arithmetic_pipeline.py",
])
def test_step3_removed_hardcoded_metrics_are_not_code_literals(module):
    """AST guard: the old fabricated constants must not survive as literals.

    The docstrings quote 0.92 / 0.88 / 0.45 / 0.85 and the ``key_layers`` table
    while describing their removal, so this deliberately ignores docstrings.
    """
    _, floats = _code_strings_and_floats(
        BACKEND / "science" / "reproducibility" / module)
    banned = {0.92, 0.88, 0.45, 0.85, 0.82, 0.91, 0.78}
    hit = sorted(set(floats) & banned)
    assert not hit, f"{module} still carries the fabricated metric(s) {hit}"


def test_step3_dispatch_builds_pipelines_with_mock_mode_false():
    """The dispatcher must never hand a caller mock-mode output as a benchmark."""
    text = _read(BACKEND / "benchmarking" / "benchmark_tasks.py")
    assert "pipeline_cls(mock_mode=False)" in text, (
        "the mock_mode construction style no longer forces mock_mode=False")


# ===================================================================== #
# Claim 6 -- /api/scientific_health and /api/analytics
# ===================================================================== #

def test_scientific_health_carries_no_top_level_numeric_score():
    from backend.core import aggregates

    payload = aggregates.scientific_health()
    assert payload["score"] is None, payload["score"]
    assert str(payload["score_reason"]).strip(), payload
    for key, value in payload.items():
        assert not (isinstance(value, (int, float))
                    and not isinstance(value, bool)), (
            f"top-level {key!r} is a bare number {value!r}; a deployment-level "
            f"number reads as a measured health score")


def test_unreadable_ledger_is_named_and_never_becomes_zero(monkeypatch):
    """The adversarial case: the ledger raises instead of being empty."""
    from backend.core import aggregates
    from backend.core import evidence_graph as eg

    def _boom(*_a, **_k):
        raise OSError("ledger is corrupt")

    monkeypatch.setattr(eg, "list_run_records", _boom)

    health = aggregates.scientific_health()
    assert health["score"] is None
    assert health["run_ledger"]["total"] is None, (
        "an unreadable ledger was reported as a count")
    assert health["run_ledger"]["by_status"] is None
    named = {entry["source"] for entry in health["unavailable"]}
    assert "run_ledger" in named, health["unavailable"]
    assert all(str(e["reason"]).strip() for e in health["unavailable"])

    analytics = aggregates.aggregate_analytics()
    assert analytics["status"] == "partial"
    assert analytics["runs"]["total"] is None
    assert analytics["runs"]["durations"] is None
    assert "run_ledger" in {e["source"] for e in analytics["unavailable"]}


def test_empty_ledger_is_zero_but_absent_duration_is_none():
    """Empty is 0; unmeasurable is None. The two are not collapsed."""
    from backend.core import aggregates

    assert aggregates._summarize_floats([]) is None
    stats = aggregates._summarize_floats([2.0, 4.0])
    assert stats == {"count": 2, "min": 2.0, "max": 4.0, "mean": 3.0}

    none_measured = aggregates._published_durations([])
    assert none_measured["measured"] is None

    unpublished = aggregates._published_durations(
        [{"run_id": "a", "created": "2026-01-01T00:00:00"}])
    assert unpublished["measured"] is None
    assert unpublished["runs_without_publication_time"] == 1

    measured = aggregates._published_durations(
        [{"created": "2026-01-01T00:00:00", "published": "2026-01-01T00:01:00"}])
    assert measured["measured"] == {"count": 1, "min": 60.0, "max": 60.0,
                                   "mean": 60.0}


def test_pass_rate_is_none_when_no_check_was_recorded(monkeypatch):
    """No recorded checks must not become a 0% (or 100%) pass rate."""
    from backend.core import aggregates
    from backend.core import evidence_graph as eg

    monkeypatch.setattr(aggregates, "_ledger", lambda: {
        "available": True,
        "source": {"name": "run_ledger", "path": "synthetic"},
        "summaries": [{"run_id": "r1"}],
    })
    monkeypatch.setattr(eg, "load_run_record", lambda _rid: {
        "result": {"trace": [{"node": "validate", "result": {
            "validated": True,
            "confidence": {"confidence_score": 0.9, "method": "heuristic",
                           "is_heuristic": True},
        }}]},
    })

    out = aggregates._validation(10)
    assert out["checks_evaluated"] == 0
    assert out["pass_rate"] is None, out["pass_rate"]
    assert out["runs_with_a_validation_verdict"] == 1
    assert out["validated_true"] == 1
    assert out["confidence"]["recorded_scores"]["count"] == 1


def test_endpoints_expose_the_same_no_fabrication_contract(control_plane):
    from backend.core import auth as auth_mod

    headers = auth_mod.auth_headers()
    for path in ("/api/scientific_health", "/api/analytics"):
        response = control_plane.get(path, headers=headers)
        assert response.status_code == 200, (path, response.status_code)
        payload = response.json()
        assert payload.get("provenance"), (path, payload.keys())
        assert "field_provenance" in payload, (path, payload.keys())
        assert "unavailable" in payload, (path, payload.keys())
        if path.endswith("scientific_health"):
            assert payload["score"] is None, payload["score"]
            assert str(payload["score_reason"]).strip()
        else:
            assert "score" not in payload, payload


# ===================================================================== #
# Claim 7 -- attribution patching: real gradient vs. a magnitude proxy
# ===================================================================== #

class _Tok:
    """Character-code tokenizer: distinct prompts encode distinctly."""

    def encode(self, text):
        return [ord(ch) % 90 + 3 for ch in text]


class _Spec:
    model_id = "synthetic"
    num_layers = 2
    num_heads = 2
    d_model = 4
    mock_mode = False


CLEAN = "The capital of France is A"
CORRUPT = "The capital of France is B"


class _SyntheticAdapter:
    """Activation and gradient are deliberately unrelated in magnitude.

    ``activation`` for L0H0 differs from its ``grad`` both in scale and in sign,
    so an implementation that used ``|activation|`` (or any monotone function of
    it) in place of the gradient cannot reproduce the scores asserted below.
    """

    spec = _Spec()
    _tokenizer = _Tok()

    def get_logits(self, prompt):
        return {"top_tokens": [{"token": "x",
                                "logit": 2.0 if "France" in prompt else 1.0}]}

    def capture_head_outputs_with_grad(self, prompt, target_id, baseline_id):
        return {
            0: {0: {"activation": [1.0, 2.0], "grad": [10.0, -3.0]},
                1: {"activation": [3.0, 4.0], "grad": [0.5, 0.25]}},
            1: {0: {"activation": [0.0, 1.0], "grad": [1.0, 1.0]},
                1: {"activation": [5.0, 6.0], "grad": [-2.0, 4.0]}},
        }

    def capture_head_outputs(self, prompt):
        return {0: {0: [0.5, 1.0], 1: [1.5, 2.0]},
                1: {0: [0.0, 0.5], 1: [2.5, 3.0]}}


def _run_attribution(adapter, dataset=None, **config):
    from backend.interpretability.discovery.algorithms.attribution_patching import (
        AttributionPatchingAlgorithm,
    )
    from backend.interpretability.discovery.algorithms.configs import (
        AttributionPatchingConfig,
    )

    dataset = dataset or {"clean": CLEAN, "corrupted": CORRUPT}
    return AttributionPatchingAlgorithm(adapter).run(
        dataset, AttributionPatchingConfig(threshold=0.0, top_k=10, **config))


def test_attribution_score_is_the_gradient_not_an_activation_magnitude():
    """The decisive test: replacing autograd with |activation| breaks the numbers.

    For Head_L0_H0: delta = [1-0.5, 2-1.0] = [0.5, 1.0] and grad = [10, -3], so

        sum(|delta * grad|) = |0.5*10| + |1.0*(-3)| = 8.0
        sum(delta * grad)   = 5.0 - 3.0             = 2.0   (signed)

    An activation-magnitude proxy would give ``|0.5*1| + |1.0*2| = 2.5`` and a
    positive sign. The run must produce 8.0 / 2.0.
    """
    report = _run_attribution(_SyntheticAdapter())
    assert report.statistics["measured"] is True, report.statistics
    scores = {a["component"]: a for a in report.evidence["top_attributions"]}

    head = scores["Head_L0_H0"]
    assert head["grad_m"] == [10.0, -3.0], head["grad_m"]
    assert head["attribution_score"] == pytest.approx(8.0), head
    assert head["signed_attribution"] == pytest.approx(2.0), head
    assert head["attribution_score"] != pytest.approx(2.5), (
        "the score matches an |activation| proxy, so the gradient was not used")

    # A magnitude proxy cannot produce a negative signed term with a positive
    # magnitude; the sign is only available from the gradient.
    assert scores["Head_L1_H1"]["signed_attribution"] == pytest.approx(7.0)
    assert scores["Head_L1_H1"]["attribution_score"] == pytest.approx(17.0)


def test_missing_gradient_yields_no_measurement():
    """Empty gradients must not fall back to a proxy or a floored confidence."""

    class _NoGrad(_SyntheticAdapter):
        def capture_head_outputs_with_grad(self, prompt, target_id, baseline_id):
            return {layer: {head: {"activation": [1.0, 2.0], "grad": []}
                            for head in range(2)} for layer in range(2)}

    report = _run_attribution(_NoGrad())
    assert report.statistics["measured"] is False
    assert report.confidence is None, report.confidence
    assert report.graph["score"] is None
    assert report.graph["measured"] is False
    assert report.evidence["top_attributions"] == []
    assert report.statistics["components_unreadable"] == 4


def test_a_failed_gradient_capture_propagates_instead_of_proxying():
    """If the backward pass cannot run, the algorithm must not invent a score."""
    from backend.science.models.adapter_base import LiveUnavailable

    class _GradUnavailable(_SyntheticAdapter):
        def capture_head_outputs_with_grad(self, prompt, target_id, baseline_id):
            raise LiveUnavailable("gradients need loaded weights")

    with pytest.raises(LiveUnavailable):
        _run_attribution(_GradUnavailable())


@pytest.mark.parametrize("dataset,label", [
    ({"clean": "gave a drink to Mary", "corrupted": "gave a drink to John",
      "target_token": "Mary"}, "answer-already-in-prompt"),
    ({"clean": "short A", "corrupted": "a much longer one B"}, "unequal-length"),
    ({"clean": CLEAN, "corrupted": CLEAN}, "identical-prompts"),
], ids=["saturation", "unequal-length", "identical"])
def test_unmeasurable_prompt_pairs_stay_unmeasured(dataset, label):
    report = _run_attribution(_SyntheticAdapter(), dataset=dataset)
    assert report.statistics["measured"] is False, (label, report.statistics)
    assert report.confidence is None
    assert report.graph["measured"] is False
    assert str(report.statistics.get("reason") or "").strip(), label


def test_no_tokenizer_means_no_metric():
    class _NoTokenizer(_SyntheticAdapter):
        _tokenizer = None

    report = _run_attribution(_NoTokenizer())
    assert report.statistics["measured"] is False
    assert "tokenizer" in str(report.statistics.get("reason")).lower()


def test_adapter_summary_uses_torch_autograd():
    """The adapter's gradient path must be an actual autograd backward."""
    text = _read(BACKEND / "science" / "models" / "gpt2_adapter.py")
    assert "torch.autograd.grad" in text, (
        "capture_head_outputs_with_grad no longer calls torch.autograd.grad")


def test_no_weights_means_no_gradient_fixture():
    from backend.science.models.adapter_base import LiveUnavailable
    from backend.science.models.gpt2_adapter import GPT2Adapter

    adapter = GPT2Adapter(variant="small", mock_mode=True)
    with pytest.raises(LiveUnavailable):
        adapter.capture_head_outputs_with_grad("The capital of France is", 1, 2)


def test_live_gradient_has_the_signature_of_a_derivative_not_a_magnitude():
    """Live-weight check: a magnitude proxy cannot produce negative entries.

    Skipped (with the recorded reason) when GPT-2 is not available;
    ``MECH_REQUIRE_WEIGHTS=1`` turns that skip into a failure for the
    live-evidence lane.
    """
    from tests.pytest import _weight_guard

    _weight_guard.require_weights()

    from backend.science.models.gpt2_adapter import GPT2Adapter

    adapter = GPT2Adapter(variant="small", mock_mode=False)
    tokenizer = adapter._tokenizer
    target = tokenizer.encode(" Paris")[0]
    baseline = tokenizer.encode(" London")[0]

    capture = adapter.capture_head_outputs_with_grad(
        "The capital of France is", target, baseline)
    assert capture, "no head outputs captured"

    grads, activations = [], []
    for layer in capture.values():
        for head in layer.values():
            grads.extend(head["grad"])
            activations.extend(head["activation"])
    assert grads, "every head carried an empty gradient"
    assert activations, "every head carried an empty activation"

    negatives = [g for g in grads if g < 0]
    assert negatives, (
        "no gradient component is negative, which is what an |activation| "
        "magnitude proxy would look like")

    # And the gradient vector is not the activation vector re-scaled.
    differing = 0
    for layer in capture.values():
        for head in layer.values():
            if head["grad"] != head["activation"]:
                differing += 1
    assert differing, "every gradient equals its activation"


def test_causal_stub_is_labelled_not_implemented():
    """The stub must refuse to be read as a measurement."""
    from backend.interpretability.causal.attribution_patching import (
        AttributionPatchingEngine,
    )

    payload = AttributionPatchingEngine().compute_attribution("clean", "corrupt")
    assert payload["status"] == "unavailable"
    assert payload["provenance"] == "reference"
    assert payload["method_available"] is False
    assert payload["attribution_measured"] is False
    assert payload["validation_eligible"] is False
    assert payload["publication_eligible"] is False
    assert str(payload["reason"]).strip()


def test_causal_stub_does_not_claim_to_run_unimplemented_method():
    """The stub must not report method='Gradient' when no gradient was computed.

    The method field must state that it is not implemented rather than echoing
    the caller's requested method as though it were executed.
    """
    from backend.interpretability.causal.attribution_patching import (
        AttributionPatchingEngine,
    )

    payload = AttributionPatchingEngine().compute_attribution(
        "clean", "corrupt", method="Gradient")
    assert payload["method"] != "Gradient", (
        "the stub reported method='Gradient' although no gradient exists "
        f"(method_available={payload['method_available']!r})")


def test_causal_stub_marks_all_numeric_leaves_with_provenance():
    """All numeric values in the NOT-IMPLEMENTED stub must carry reference provenance."""
    from backend.interpretability.causal.attribution_patching import (
        AttributionPatchingEngine,
    )
    from tests.pytest._provenance import collect_unmarked_numbers

    payload = AttributionPatchingEngine().compute_attribution("clean", "corrupt")
    unmarked = [f"{path}={value!r}"
                for path, value in collect_unmarked_numbers(payload)]
    assert not unmarked, (
        "numeric values in the NOT-IMPLEMENTED stub carry no provenance, so a "
        f"consumer can quote them as measured: {unmarked}")


# ===================================================================== #
# Claim 8 -- TransformerLens 4.x dependency conflict
# ===================================================================== #

#: Verified from ``https://pypi.org/pypi/transformer-lens/<version>/json`` on
#: 2026-10-17 (the wheel METADATA ``requires_dist``), not inferred:
#: ``transformers`` floor per transformer-lens release.
TL_TRANSFORMERS_FLOOR = {
    "3.5.1": "5.4.0",
    "3.6.0": "5.9.0",
    "3.7.0": "5.9.0",
    "3.7.3": "5.9.0",
    "3.8.0": "5.9.0",
    "3.8.1": "5.9.0",
    "3.9.0": "5.9.0",
    "4.0.0": "5.9.0",
    "4.1.0": "5.9.0",
    "4.2.0": "5.9.0",
}

#: ``transformer-lens >= 3.6.0`` is where the floor moves to 5.9.0. The migration
#: note says "from 3.9.0 onward", which is true for the versions it names but
#: understates the blocker by three minor releases.
FIRST_TL_NEEDING_TRANSFORMERS_5_9 = "3.6.0"


def _parse_version(text: str):
    return tuple(int(part) for part in text.split("."))


def _requirement_line(package: str) -> str:
    for raw in _read(REPO_ROOT / "requirements.txt").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name = re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0].strip().lower()
        if name == package.lower():
            return line
    raise AssertionError(f"{package} missing from requirements.txt")


def test_requirements_pin_excludes_every_tl_version_needing_transformers_5_9():
    """The conflict is provable locally: the pin must sit below the floor bump."""
    line = _requirement_line("transformer-lens")
    assert ">=2.18" in line.replace(" ", ""), line
    upper = int(re.search(r"<\s*(\d+)", line).group(1))
    assert upper <= 3, (
        f"transformer-lens upper bound {upper} admits versions whose "
        f"transformers floor is 5.9.0: {line!r}")

    floor_bump = _parse_version(TL_TRANSFORMERS_FLOOR[
        FIRST_TL_NEEDING_TRANSFORMERS_5_9])
    assert floor_bump >= (5, 9), FIRST_TL_NEEDING_TRANSFORMERS_5_9
    for version, floor in TL_TRANSFORMERS_FLOOR.items():
        if _parse_version(floor) >= (5, 9):
            assert _parse_version(version)[0] >= upper or \
                _parse_version(version) < (2, 18), (
                f"transformer-lens {version} (transformers>={floor}) satisfies "
                f"the pin {line!r}, so the documented conflict would be live")


def test_transformers_upper_bound_is_stricter_than_the_tl_floor():
    line = _requirement_line("transformers")
    bound = int(re.search(r"<\s*(\d+)", line).group(1))
    assert bound <= 5, line
    assert (5, 9) >= (bound, 0), (
        f"transformers {line!r} would allow the >=5.9.0 that transformer-lens "
        f"{FIRST_TL_NEEDING_TRANSFORMERS_5_9}+ requires")


def test_migration_note_exists_and_names_the_conflict():
    note = REPO_ROOT / "docs" / "notes" / "transformerlens-4-migration-blocker.md"
    assert note.exists(), note
    text = _read(note)
    assert "transformers>=5.9.0" in text
    assert "TransformerBridge" in text
    assert "3.6.0" in text
    # Every transformer-lens version the note names must actually carry the
    # 5.9.0 floor -- the note's own claim, checked against the verified table.
    for version in ("3.6.0", "3.7.0", "3.8.0", "3.9.0", "4.0.0", "4.1.0", "4.2.0"):
        assert _parse_version(TL_TRANSFORMERS_FLOOR[version]) >= (5, 9), version


def test_dependency_contract_still_asserts_a_simulated_4x_is_rejected():
    """Confirm the negative control the migration note refers to is still there."""
    text = _read(REPO_ROOT / "tests" / "pytest" / "test_dependency_contract.py")
    assert "def test_compat_probe_detects_a_4x_install" in text
    assert "transformer_lens-4.0.0.dist-info" in text
    assert 'assert not report["compatible"]' in text
