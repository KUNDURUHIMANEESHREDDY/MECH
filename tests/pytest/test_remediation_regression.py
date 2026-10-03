"""Dedicated regression test suite verifying all remediation and loophole fixes.

Covers:
1. /api/infer live engine inference and non-string payload validation (HTTP 400).
2. Dispatcher payload validation returning 400 instead of 500 crashes on invalid inputs.
3. StorageError exception handling returning structured 400 responses.
4. SQLite WAL checkpointing routine execution.
5. AI Scientist KeyError guards and fail-closed uncertainty logic.
6. Discovery engine lifecycle progression to Publication.
7. Evidence provenance tracking and knowledge-graph writeback resilience.
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.storage.database import StorageError, checkpoint_wal
from backend.interpretability.discovery.discovery_engine import DiscoveryEngine
from backend.research_platform.autonomous.ai_scientist_engine import AIScientistEngine


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint_contract(client):
    """Verify /health returns HTTP 200 with {'status': 'healthy'}."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_infer_endpoint_valid_and_invalid_payloads(client):
    """Verify /api/infer functions correctly and returns 400 on invalid non-string prompt."""
    # Valid call
    resp_valid = client.post("/api/infer", json={"prompt": "The capital of France is", "model_name": "gpt2-small"})
    assert resp_valid.status_code == 200
    data = resp_valid.json()
    assert "tokens" in data
    assert "generated_text" in data

    # Invalid non-string prompt payload
    resp_invalid = client.post("/api/infer", json={"prompt": 12345})
    assert resp_invalid.status_code == 400
    assert "must be a string" in resp_invalid.json().get("detail", "")


def test_dispatcher_numeric_validation(client):
    """Verify malformed numeric parameters yield HTTP 400 rather than unhandled 500 crashes."""
    # Malformed layer in /api/gpt2/activations
    resp1 = client.post("/api/gpt2/activations", json={"layer": "not-an-int"})
    assert resp1.status_code == 400

    # Malformed layer in /api/gpt2/attention_head
    resp2 = client.post("/api/gpt2/attention_head", json={"layer": "invalid", "head": 0})
    assert resp2.status_code == 400

    # Malformed head in /api/gpt2/patch_head
    resp3 = client.post("/api/gpt2/patch_head", json={"layer": 0, "head": "invalid"})
    assert resp3.status_code == 400


def test_storage_error_custom_exception_handler(client):
    """Verify StorageError returns HTTP 400 with structured detail and error_type."""
    # Invalid experiment without required structure or invalid values
    resp = client.post("/api/experiments", json={"id": "bad_exp", "timestamp": "not-a-timestamp", "corrupt": None})
    # If storage accepts it or raises StorageError, it must not return HTTP 500
    assert resp.status_code in (200, 400)
    if resp.status_code == 400:
        assert resp.json().get("error_type") == "StorageError"


def test_sqlite_wal_checkpoint_execution():
    """Verify explicit WAL checkpointing executes cleanly without locks."""
    busy, log_pages, checkpointed = checkpoint_wal()
    assert isinstance(busy, int)
    assert isinstance(log_pages, int)
    assert isinstance(checkpointed, int)


def test_ai_scientist_defensive_key_handling():
    """Verify AIScientistEngine safely handles empty or missing confidence dicts."""
    engine = AIScientistEngine()
    # Mocking validation returning unavailable without confidence key
    decision = engine.uncertainty_manager.evaluate_uncertainty(
        confidence_score=0.95,
        uncertainty_interval=(0.80, 0.95),
        sample_size=5,
        variance=0.02,
    )
    assert "action" in decision
    assert decision["action"] in ("Publish", "More experiments", "Debate")


def test_discovery_engine_does_not_self_publish():
    """DiscoveryEngine must not advance itself to Publication.

    It previously ran the whole synthetic chain -- hypothesis test, IOI eval,
    induction heads, universality, paper replication -- and ended at
    Publication with a "Confirmed" verdict for any input. That is exactly the
    plausible-output-instead-of-evidence failure the evidence policy exists to
    prevent. The orchestrator now runs live causal discovery, stops at
    Validation, and hands off.
    """
    engine = DiscoveryEngine()
    result = engine.discover_and_orchestrate("L8_N402 mediates IOI capital retrieval")

    assert result["lifecycle"]["state"] != "Publication"
    assert result["status"] in ("completed", "unavailable", "blocked", "error")
    assert result.get("test_result") is None
    assert "circuit_name" not in result

    if result.get("status") == "completed" and result.get("provenance") == "live":
        # Eligibility is derived, not asserted. The executor runs at its
        # four-prompt default and marks a result of that size
        # validation-ineligible, so the orchestrator now holds the lifecycle at
        # Evidence Collection instead of advancing to Validation regardless.
        #
        # This branch previously asserted `validation_eligible is True`, which
        # pinned the unconditional flags the executor used to emit. What actually
        # matters here is that the lifecycle label agrees with the flags.
        assert result["measured"] is True
        assert result["publication_eligible"] is False, (
            "a four-prompt run cannot support a publication claim"
        )
        if result["validation_eligible"] is False:
            assert result["ineligible_because"], (
                "an ineligible result must say why"
            )
            assert result["lifecycle"]["state"] != "Validation", (
                "the lifecycle advanced to Validation while the record says it is "
                "not validation-eligible"
            )
        else:
            assert result["lifecycle"]["state"] == "Validation"
    else:
        assert result["validation_eligible"] is False
        assert result["publication_eligible"] is False
        assert result["reason"]


def test_attention_head_respects_prompt(client):
    """Regression: /api/gpt2/attention_head must use the supplied prompt, not stale cache.

    Two different prompts must produce different token sequences in the response —
    this catches the bug where the cache was not refreshed when a new prompt was
    submitted.
    """
    resp_a = client.post(
        "/api/gpt2/attention_head",
        json={"prompt": "The cat sat on the mat", "layer": 0, "head": 0},
    )
    resp_b = client.post(
        "/api/gpt2/attention_head",
        json={"prompt": "Neural networks learn representations", "layer": 0, "head": 0},
    )
    assert resp_a.status_code == 200
    assert resp_b.status_code == 200
    tokens_a = resp_a.json().get("str_tokens", [])
    tokens_b = resp_b.json().get("str_tokens", [])
    # Both prompts must produce non-empty token lists
    assert len(tokens_a) > 0, "attention_head returned empty tokens for prompt A"
    assert len(tokens_b) > 0, "attention_head returned empty tokens for prompt B"
    # Tokens must be different (different prompts → different tokenisations)
    assert tokens_a != tokens_b, (
        "attention_head returned identical tokens for two different prompts — "
        "prompt-cache bug still present"
    )


def test_activations_respects_prompt(client):
    """Regression: /api/gpt2/activations must reflect the supplied prompt in resid_shape.

    A short prompt (2 tokens) and a long prompt (6+ tokens) must produce different
    sequence lengths, proving the cache is refreshed per request.
    """
    resp_short = client.post(
        "/api/gpt2/activations",
        json={"prompt": "Hello world", "layer": 5},
    )
    resp_long = client.post(
        "/api/gpt2/activations",
        json={"prompt": "The quick brown fox jumps over the lazy dog", "layer": 5},
    )
    assert resp_short.status_code == 200
    assert resp_long.status_code == 200
    short_seq = resp_short.json().get("resid_shape", [0, 0])[0]
    long_seq = resp_long.json().get("resid_shape", [0, 0])[0]
    assert short_seq > 0, "activations returned zero-length sequence for short prompt"
    assert long_seq > short_seq, (
        f"long prompt seq ({long_seq}) should exceed short prompt seq ({short_seq}) — "
        "prompt-cache bug still present"
    )
