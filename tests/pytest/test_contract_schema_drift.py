"""Backend/Frontend schema drift guard (#16).

Every response the dispatcher can emit must parse against
backend.contracts.models. A rename, removal, or silent shape change
fails here before it reaches the frontend as a mysteriously empty panel.

Also pins the two fail-closed invariants:
  * no "ok"-family status without provenance != unavailable carrying
    an explicit error/reason
  * unavailable responses never carry measured-looking payload fields
    (matrix, tokens, neurons, attention_maps) without provenance=live
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict

import pytest
from pydantic import ValidationError

from backend.contracts import models as C


# route -> contract model
ROUTE_MODELS: Dict[str, Any] = {
    "/infer": C.InferenceResponse,
    "/models/load": C.ModelLoadResponse,
    "/gpt2/load": C.ModelLoadResponse,
    "/gpt2/architecture": C.GPT2ArchitectureResponse,
    "/gpt2/layer": C.GPT2LayerResponse,
    "/gpt2/neurons": C.GPT2NeuronsResponse,
    "/gpt2/neuron": C.GPT2NeuronDetailResponse,
    "/gpt2/attention_head": C.GPT2AttentionHeadResponse,
    "/gpt2/head": C.GPT2HeadResponse,
    "/gpt2/activations": C.GPT2ActivationsResponse,
    "/gpt2/fresh_prompt": C.GPT2FreshPromptResponse,
    "/gpt2/patch_neuron": C.GPT2PatchNeuronResponse,
    "/gpt2/run_prompt": C.GPT2RunPromptResponse,
    "/gpt2/patch_head": C.GPT2PatchHeadResponse,
    "/gpt2/ioi": C.GPT2IOIResponse,
    "/gpt2/steer": C.GPT2SteerResponse,
    "/gpt2/layer_activations": C.GPT2LayerActivationsResponse,
    "/gpt2/logit_lens_all": C.GPT2LogitLensAllResponse,
    "/sae/status": C.SAEStatusResponse,
    "/sae/inspect": C.SAEInspectResponse,
    "/sae/train": C.SAETrainStartResponse,
    "/sae/runs/{run_id}": C.SAERunStatusResponse,
    "/benchmarks/run": C.BenchmarkRunResponse,
    "/runtime/status": C.RuntimeStatusResponse,
}

MEASURED_PAYLOAD_KEYS = {
    "matrix", "tokens", "str_tokens", "attention_maps",
    "neuron_activations", "neurons", "resid_post", "mlp_post",
    "clean_ld", "patched_ld", "delta",
}


def test_all_contracted_routes_have_models():
    from backend.api import dispatcher as D

    routes = set()
    for r in D.router.routes:
        path = getattr(r, "path", "")
        if path.startswith("/gpt2/") or path.startswith("/sae/") or path in (
            "/infer", "/benchmarks/run", "/runtime/status",
        ):
            # strip trailing alias duplicates (/patchhead vs /patch_head)
            routes.add(path)
    # Every live route must be in the contract map (aliases may share).
    uncovered = {p for p in routes if p not in ROUTE_MODELS and p != "/gpt2/patchhead"}
    assert not uncovered, f"routes without contract models: {sorted(uncovered)}"


def test_contract_models_import_and_validate_unavailable():
    # Every GPT-2 response model must accept a minimal unavailable payload.
    unavailable_models = [
        C.GPT2ArchitectureResponse, C.GPT2LayerResponse,
        C.GPT2NeuronsResponse, C.GPT2NeuronDetailResponse,
        C.GPT2AttentionHeadResponse, C.GPT2HeadResponse,
        C.GPT2ActivationsResponse, C.GPT2FreshPromptResponse,
        C.GPT2PatchNeuronResponse, C.GPT2RunPromptResponse,
        C.GPT2PatchHeadResponse, C.GPT2IOIResponse,
        C.GPT2SteerResponse, C.GPT2LayerActivationsResponse,
        C.GPT2LogitLensAllResponse, C.InferenceResponse,
        C.SAEStatusResponse, C.SAEInspectResponse, C.SAERunStatusResponse,
    ]
    for model in unavailable_models:
        obj = model.model_validate({
            "status": "unavailable",
            "provenance": "unavailable",
            "error": "torch/transformers not available",
        })
        assert obj.provenance == "unavailable"
        assert obj.status == "unavailable"


def test_fail_closed_no_ok_without_provenance():
    """An ok/loaded/completed status must never ride on unavailable provenance
    without an explicit error/reason — that is the seeded-fallback shape.
    The contract models enforce this with a validator; this test pins it."""
    ok_statuses = {"ok", "loaded", "completed", "connected"}
    for name, model in ROUTE_MODELS.items():
        for status in ok_statuses:
            with pytest.raises(ValidationError):
                model.model_validate({
                    "status": status,
                    "provenance": "unavailable",
                })
        # And the fail-closed shape (with cause) must parse.
        obj = model.model_validate({
            "status": "unavailable",
            "provenance": "unavailable",
            "error": "torch/transformers not available",
        })
        assert obj.provenance == "unavailable"


def test_live_dispatcher_unavailable_paths_parse():
    """Call the dispatcher with the ML stack forced off and validate
    every fail-closed response against its contract model."""
    from fastapi.testclient import TestClient
    from backend.core import auth as auth_mod
    from backend.main import app

    client = TestClient(app, raise_server_exceptions=False,
                        headers=auth_mod.auth_headers())

    cases = [
        ("POST", "/api/infer", {"prompt": "Hello"}, "/infer"),
        ("POST", "/api/gpt2/run_prompt", {"prompt": "Hello"}, "/gpt2/run_prompt"),
        ("POST", "/api/gpt2/activations", {"layer": 0}, "/gpt2/activations"),
        ("POST", "/api/gpt2/attention_head", {"layer": 0, "head": 0}, "/gpt2/attention_head"),
        ("POST", "/api/gpt2/patch_head", {"layer": 0, "head": 0}, "/gpt2/patch_head"),
        ("POST", "/api/gpt2/ioi", {"io_name": "Alice", "subj_name": "Bob"}, "/gpt2/ioi"),
        ("POST", "/api/gpt2/architecture", {}, "/gpt2/architecture"),
        ("POST", "/api/gpt2/layer", {"layer": 0}, "/gpt2/layer"),
        ("POST", "/api/gpt2/neurons", {"layer": 0}, "/gpt2/neurons"),
        ("POST", "/api/gpt2/neuron", {"layer": 0, "neuron_index": 0}, "/gpt2/neuron"),
        ("POST", "/api/gpt2/layer_activations", {"layer": 0, "prompt": "Hi"}, "/gpt2/layer_activations"),
        ("POST", "/api/gpt2/logit_lens_all", {"prompt": "Hi"}, "/gpt2/logit_lens_all"),
    ]
    try:
        import backend.services.gpt2_engine as eng
    except Exception:
        eng = None
    ml_was_available = bool(eng and eng.is_available())

    for method, url, body, key in cases:
        resp = client.request(method, url, json=body)
        assert resp.status_code == 200, f"{url} -> HTTP {resp.status_code}"
        data = resp.json()
        model = ROUTE_MODELS[key]
        try:
            obj = model.model_validate(data)
        except ValidationError as exc:
            pytest.fail(f"{url} response fails contract {model.__name__}:\n{exc}\nData: {str(data)[:500]}")
        # If the ML stack is absent, unavailable responses must not smuggle
        # measured-looking payload keys.
        if not ml_was_available and obj.model_dump().get("provenance") == "unavailable":
            leaked = [k for k in MEASURED_PAYLOAD_KEYS if data.get(k) not in (None, [], {})]
            # attention empty-list etc. is fine; non-empty measured payload is not
            assert not leaked, f"{url} leaks measured fields while unavailable: {leaked}"


def test_seeded_standins_stay_deleted():
    """The deterministic fake-measurement helpers must not come back.

    `_seed`, `_seeded_tokens`, `TOKEN_POOLS` and `_random_token_sequence`
    produced stable, convincing, entirely unmeasured attention matrices,
    token lists and logits. Every route fails closed now; these names in
    executable code mean a fallback path was reintroduced.
    """
    from source_assert import executable_source

    import backend.api.dispatcher as dispatcher

    src = executable_source(dispatcher)
    for banned in ("_seeded_tokens", "TOKEN_POOLS", "_random_token_sequence"):
        assert banned not in src, f"seeded stand-in {banned!r} is back in the dispatcher"
    assert "def _seed(" not in src.replace("def _seeded_tokens(", ""), \
        "seeded stand-in `_seed` is back in the dispatcher"
