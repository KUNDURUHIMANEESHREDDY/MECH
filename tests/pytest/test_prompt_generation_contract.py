"""Model-only GPT-2 prompt generation must fail closed and never use a text pool."""
from __future__ import annotations


def test_fresh_prompt_endpoint_uses_only_gpt2_generated_text(monkeypatch):
    from backend.api import dispatcher

    class LiveEngine:
        calls = []

        def is_available(self):
            return True

        def generate_text(self, primer, max_new_tokens=12):
            self.calls.append((primer, max_new_tokens))
            return {
                "status": "ok",
                "text": "generated tokens from the fixture model",
                "temperature": 0.9,
                "top_k": 40,
                "tokens_added": 6,
            }

    engine = LiveEngine()
    monkeypatch.setattr(dispatcher, "get_engine", lambda: engine)
    response = dispatcher.gpt2_fresh_prompt({
        # A client-provided prefix must be ignored by the model-only endpoint.
        "primer": "this must never appear in generated output",
        "max_new_tokens": 6,
    })

    assert engine.calls == [("", 6)]
    assert response["status"] == "ok"
    assert response["prompt"] == "generated tokens from the fixture model"
    assert response["primer"] == ""
    assert response["provenance"] == "live"


def test_fresh_prompt_endpoint_does_not_substitute_text_when_gpt2_is_unavailable(monkeypatch):
    from backend.api import dispatcher

    class OfflineEngine:
        def is_available(self):
            return False

        def generate_text(self, *args, **kwargs):
            raise AssertionError("No model call should be attempted when weights are unavailable.")

    monkeypatch.setattr(dispatcher, "get_engine", lambda: OfflineEngine())
    response = dispatcher.gpt2_fresh_prompt({})

    assert response["status"] == "unavailable"
    assert not response.get("prompt")
    assert response["provenance"] == "unavailable"


def test_omitted_run_prompt_does_not_fall_back_to_a_canned_prompt(monkeypatch):
    from backend.api import dispatcher

    class OfflineEngine:
        def is_available(self):
            return False

        def generate_text(self, *args, **kwargs):
            raise AssertionError("Offline GPT-2 must not generate a default input.")

    monkeypatch.setattr(dispatcher, "get_engine", lambda: OfflineEngine())
    response = dispatcher.gpt2_run_prompt({})

    assert response["status"] == "unavailable"
    assert not response.get("prompt")
    assert "no fallback input was substituted" in response["error"].lower()
