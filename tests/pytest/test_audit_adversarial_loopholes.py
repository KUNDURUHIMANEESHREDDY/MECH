"""Adversarial edge-case audit for HTTP endpoints — pytest version.

Tests that malformed, out-of-bounds, and injection payloads are handled
gracefully (4xx) rather than crashing the server (500).
"""

import json
from typing import Any, Dict, List, Tuple

import pytest
from fastapi.testclient import TestClient


def _client():
    from backend.main import app
    return TestClient(app, raise_server_exceptions=False)


# Test cases: (label, method, path, payload, expect_crash)
# expect_crash=True means we expect a 500 (server bug); False means 4xx/2xx is correct handling
CASES: List[Tuple[str, str, str, Any, bool]] = [
    # 1. Out-of-bounds layer / head values (should handle gracefully without 500)
    ("OOB Layer in attention_head", "POST", "/api/gpt2/attention_head", {"layer": 9999, "head": 9999, "prompt": "Hi"}, False),
    ("Negative Layer in attention_head", "POST", "/api/gpt2/attention_head", {"layer": -5, "head": -2, "prompt": "Hi"}, False),
    ("OOB Layer in activations", "POST", "/api/gpt2/activations", {"layer": 100, "prompt": "Test"}, False),
    ("OOB Neuron Index", "POST", "/api/gpt2/neuron", {"layer": 0, "neuron_index": 999999}, False),

    # 2. Type Mismatch / Invalid Types
    ("Float layer in attention_head", "POST", "/api/gpt2/attention_head", {"layer": 1.5, "head": "abc"}, False),
    ("Invalid pos_token type", "POST", "/api/gpt2/patch_head", {"layer": 0, "head": 0, "pos_token": 123, "neg_token": None}, False),
    ("Null prompt in infer", "POST", "/api/infer", {"prompt": None}, False),
    ("Integer prompt in run_prompt", "POST", "/api/gpt2/run_prompt", {"prompt": 9999}, False),

    # 3. Path Traversal / SQL Injection in CRUD & Routes
    ("SQL Injection in Experiment ID", "POST", "/api/experiments", {"id": "'; DROP TABLE experiments; --", "title": "Injected"}, False),
    ("Nonexistent Circuit Query", "GET", "/api/circuits/non_existent_fake_circuit", None, False),
    ("Nonexistent Society Run Query", "GET", "/api/society/runs/fake_run_12345", None, False),

    # 4. Empty / Boundary strings
    ("Empty Prompt in infer", "POST", "/api/infer", {"prompt": ""}, False),
    ("Very Long Prompt (2000 chars)", "POST", "/api/infer", {"prompt": "AI interpretability " * 100}, False),
    ("Special Unicode & Emojis", "POST", "/api/gpt2/run_prompt", {"prompt": "?? Transformer ? ?? \u2603 \x00"}, False),
]


@pytest.mark.parametrize("label,method,path,payload,expect_crash", CASES)
def test_adversarial_edge_case(label, method, path, payload, expect_crash):
    """Each adversarial payload must not cause an unhandled 500 crash."""
    client = _client()

    if method == "GET":
        res = client.get(path)
    elif method == "POST":
        res = client.post(path, json=payload)
    elif method == "DELETE":
        res = client.delete(path)
    else:
        pytest.skip(f"Unsupported method: {method}")

    status = res.status_code
    is_500 = status >= 500 or status == 0

    if expect_crash:
        # If we marked it as expecting a crash, a 500 confirms the bug exists
        assert is_500, f"[{label}] Expected crash (500) but got {status}"
    else:
        # Normal case: must NOT crash with 500
        assert not is_500, (
            f"[{label}] UNHANDLED CRASH (HTTP {status}) on {method} {path}\n"
            f"Response: {res.text[:500]}"
        )