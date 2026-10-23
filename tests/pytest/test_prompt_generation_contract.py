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
    # The stub verifies handler inputs/output shaping only; it is not being
    # presented as a real measurement or used to bypass runtime attestation.
    monkeypatch.setattr(dispatcher, "_mark", lambda body, provenance: {**body, "provenance": provenance})
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



def test_layer_activation_route_uses_generated_gpt2_input_when_prompt_is_omitted(monkeypatch):
    from backend.api import dispatcher

    class LiveEngine:
        observed_prompt = None

        def is_available(self):
            return True

        def generate_text(self, primer, max_new_tokens=12):
            assert primer == ""
            return {
                "status": "ok",
                "text": "A fresh continuation sampled by GPT-2",
                "tokens_added": 7,
            }

        def layer_activations(self, layer, prompt):
            self.observed_prompt = prompt
            return {"status": "ok", "prompt": prompt, "layer": layer}

    engine = LiveEngine()
    monkeypatch.setattr(dispatcher, "get_engine", lambda: engine)
    monkeypatch.setattr(dispatcher, "pass_through", lambda body: body)
    response = dispatcher.gpt2_layer_activations({"layer": 3})

    assert engine.observed_prompt == "A fresh continuation sampled by GPT-2"
    assert response["prompt"] == engine.observed_prompt
    assert response["layer"] == 3
