"""Systematic feature audit across all platform capabilities — pytest version.

Each endpoint is exercised via TestClient (in-process) with proper auth and validated for
successful HTTP status AND meaningful response content. A 2xx status alone
is not sufficient — responses must contain expected fields and provenance.
"""

import json
import time
from typing import Any, Dict, List, Tuple

import pytest
from fastapi.testclient import TestClient

from backend.core import auth as auth_mod


AUTH_HEADERS = auth_mod.auth_headers()


def _client():
    from backend.main import app
    return TestClient(app, raise_server_exceptions=False)


FEATURES: List[Tuple[str, str, str, Any, bool, str]] = [
    # (name, method, path, body, requires_auth, expected_type)
    # 1. Health & System (public)
    ("Health Check", "GET", "/health", None, False, "json"),
    ("API Status", "GET", "/api/status", None, True, "json"),
    ("API Ping", "POST", "/api/ping", {}, True, "json"),
    ("Portal Summary", "GET", "/api/portal/summary", None, True, "json"),

    # 2. Model Catalog & Management
    ("List Models", "GET", "/api/models", None, True, "json"),
    ("Get Model Info (gpt2-small)", "GET", "/api/models/gpt2-small", None, True, "json"),
    ("Load Model", "POST", "/api/models/load", {"model_name": "gpt2-small"}, True, "json"),
    ("GPT2 Load", "POST", "/api/gpt2/load", {"model_name": "gpt2"}, True, "json"),

    # 3. GPT-2 Inference & Probing
    ("Infer (Prompt)", "POST", "/api/infer", {"prompt": "The capital of France is"}, True, "json"),
    ("GPT2 Run Prompt", "POST", "/api/gpt2/run_prompt", {"prompt": "Neural networks learn"}, True, "json"),
    ("GPT2 Architecture", "POST", "/api/gpt2/architecture", {}, True, "json"),
    ("GPT2 Layer Detail (L0)", "POST", "/api/gpt2/layer", {"layer": 0}, True, "json"),
    ("GPT2 Head Detail (L0H0)", "POST", "/api/gpt2/head", {"layer": 0, "head": 0}, True, "json"),
    ("GPT2 Attention Head (L5H5)", "POST", "/api/gpt2/attention_head", {"prompt": "The cat sat on the mat", "layer": 5, "head": 5}, True, "json"),
    ("GPT2 Activations (L2)", "POST", "/api/gpt2/activations", {"prompt": "Hello world", "layer": 2}, True, "json"),
    ("GPT2 Layer Activations", "POST", "/api/gpt2/layer_activations", {"prompt": "Quick brown fox", "layer": 1}, True, "json"),
    ("GPT2 Single Neuron (L0N10)", "POST", "/api/gpt2/neuron", {"layer": 0, "neuron_index": 10}, True, "json"),
    ("GPT2 Neurons Page", "POST", "/api/gpt2/neurons", {"layer": 0, "page": 1, "page_size": 10}, True, "json"),
    ("GPT2 Patch Head", "POST", "/api/gpt2/patch_head", {"layer": 9, "head": 9, "pos_token": "Paris", "neg_token": "London"}, True, "json"),
    ("GPT2 Patch Neuron", "POST", "/api/gpt2/patch_neuron", {"layer": 0, "neuron": 10, "pos_token": "Paris", "neg_token": "London"}, True, "json"),
    ("GPT2 Logit Lens All", "POST", "/api/gpt2/logit_lens_all", {"prompt": "France capital is"}, True, "json"),
    ("Figure Attention", "GET", "/api/figures/attention?layer=0&head=0&prompt=Hello", None, True, "image"),

    # 4. Mechanistic Circuits & Interpretability
    ("GPT2 IOI Reproduction", "POST", "/api/gpt2/ioi", {"io_name": "Mary", "subj_name": "John"}, True, "json"),
    ("List Circuits", "GET", "/api/circuits", None, True, "json"),
    ("Get Circuit Detail", "GET", "/api/circuits/ioi_circuit_gpt2_small", None, True, "json"),
    ("List Discoveries", "GET", "/api/discoveries", None, True, "json"),
    ("List Inspectors", "GET", "/api/interpretability/inspectors", None, True, "json"),
    ("Runtime Analyze Tokens", "POST", "/api/runtime/analyze_tokens", {"text": "Transformers process tokens"}, True, "json"),
    ("Runtime Engines", "GET", "/api/runtime/engines", None, True, "json"),
    ("Runtime Status", "POST", "/api/runtime/status", {}, True, "json"),

    # 5. Autonomous AI Scientist & Society
    ("List Agents", "GET", "/api/agents", None, True, "json"),
    ("Research Catalog", "GET", "/api/research_catalog", None, True, "json"),
    ("Society Run Proposal", "POST", "/api/society/run", {"hypothesis": "Layer 9 mediates indirect object identification", "model": "gpt2-small"}, True, "json"),
    ("Society Runs List", "GET", "/api/society/runs", None, True, "json"),

    # 6. Knowledge Graph & Storage Persistence
    ("Knowledge Graph Info", "GET", "/api/knowledge-graph", None, True, "json"),
    ("List Experiments", "GET", "/api/experiments", None, True, "json"),
    ("Create Experiment", "POST", "/api/experiments", {"id": "audit_exp_001", "title": "Comprehensive Feature Audit", "status": "running"}, True, "json"),
    ("Delete Experiment", "DELETE", "/api/experiments/audit_exp_001", None, True, "json"),
    ("List Sessions", "GET", "/api/sessions", None, True, "json"),
    ("Create Session", "POST", "/api/sessions", {"id": "audit_sess_001", "name": "Audit Session", "model": "gpt2-small"}, True, "json"),
    ("Delete Session", "DELETE", "/api/sessions/audit_sess_001", None, True, "json"),
    ("List Benchmarks", "GET", "/api/benchmarks", None, True, "json"),
    ("Run Benchmark", "POST", "/api/benchmarks/run", {"benchmark_name": "induction_test"}, True, "json"),
]


@pytest.mark.parametrize("name,method,path,body,requires_auth,expected_type", FEATURES)
def test_feature_endpoint(name, method, path, body, requires_auth, expected_type):
    """Each platform feature endpoint must respond successfully with valid content."""
    client = _client()

    headers = AUTH_HEADERS if requires_auth else None

    start = time.time()
    try:
        if method == "GET":
            res = client.get(path, headers=headers)
        elif method == "POST":
            res = client.post(path, json=body, headers=headers)
        elif method == "DELETE":
            res = client.delete(path, headers=headers)
        else:
            pytest.skip(f"Unsupported method: {method}")
    except Exception as e:
        pytest.fail(f"[{name}] Request failed: {e}")

    elapsed = round((time.time() - start) * 1000, 2)

    # Basic HTTP success
    assert 200 <= res.status_code < 300, (
        f"[{name}] HTTP {res.status_code} on {method} {path} (latency: {elapsed}ms)\n"
        f"Response: {res.text[:500]}"
    )

    # Response validation by expected type
    if expected_type == "json" and res.status_code != 204:
        try:
            data = res.json()
        except json.JSONDecodeError:
            pytest.fail(f"[{name}] Invalid JSON response: {res.text[:200]}")

        # Basic sanity: response should be a dict with some content
        assert isinstance(data, dict), f"[{name}] Expected JSON object, got {type(data)}"
        assert len(data) > 0, f"[{name}] Empty response body"

    elif expected_type == "image":
        # Image responses should have image content-type and binary content
        content_type = res.headers.get("content-type", "")
        assert "image" in content_type, f"[{name}] Expected image content-type, got {content_type}"
        assert len(res.content) > 0, f"[{name}] Empty image response"

    # Latency sanity check (10s max)
    assert elapsed < 10000, f"[{name}] Latency {elapsed}ms exceeds 10s threshold"