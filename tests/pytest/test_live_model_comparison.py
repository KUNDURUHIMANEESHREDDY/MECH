"""Regression tests for live cross-checkpoint comparison and fail-closed provenance."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from backend.runtime.comparison import (
    HuggingFaceComparisonBackend,
    ModelComparisonEngine,
)


class TinyTokenizer:
    """Synthetic tokenizer used only to exercise metric alignment in unit tests."""

    def __init__(self):
        self.vocab = {"<zero>": 0, "a": 1, "b": 2, "c": 3, "d": 4}
        self.special_tokens_map = {"eos_token": "<zero>"}

    def get_vocab(self):
        return dict(self.vocab)

    def __call__(self, text, return_tensors="pt"):
        # Deterministic *test fixture*, not a production prompt or measurement.
        return {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}

    def convert_ids_to_tokens(self, ids):
        inverse = {index: token for token, index in self.vocab.items()}
        return [inverse[int(index)] for index in ids]

    def decode(self, ids, skip_special_tokens=False):
        inverse = {index: token for token, index in self.vocab.items()}
        tokens = [inverse[int(index)] for index in ids]
        if skip_special_tokens:
            tokens = [token for token in tokens if token != "<zero>"]
        return "".join(tokens)


class TinyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c_proj = torch.nn.Identity()


class TinyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = TinyMLP()


class TinyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.h = torch.nn.ModuleList([TinyBlock(), TinyBlock()])


class TinyCausalModel(torch.nn.Module):
    """Minimal parameterized forward pass that behaves like a HF causal LM."""

    def __init__(self, *, hidden_size=4, scale=1.0, revision="abc123"):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(float(scale)))
        self.bias = torch.nn.Parameter(torch.arange(5, dtype=torch.float32) * scale)
        self.transformer = TinyTransformer()
        self.config = SimpleNamespace(
            model_type="gpt2",
            architectures=["GPT2LMHeadModel"],
            num_hidden_layers=2,
            num_attention_heads=2,
            hidden_size=hidden_size,
            vocab_size=5,
            _name_or_path="test-checkpoint",
            _commit_hash=revision,
        )

    def forward(self, input_ids, output_hidden_states=False,
                output_attentions=False, return_dict=True):
        batch, sequence = input_ids.shape
        width = self.config.hidden_size
        base = input_ids.float().unsqueeze(-1).expand(batch, sequence, width) * self.scale
        hidden_states = [base]
        for block in self.transformer.h:
            base = block.mlp.c_proj(base + self.scale)
            hidden_states.append(base)
        logits = (self.bias + self.scale).reshape(1, 1, -1).expand(batch, sequence, -1)
        attention_logits = torch.ones(
            batch, self.config.num_attention_heads, sequence, sequence,
            dtype=torch.float32, device=input_ids.device,
        ) * self.scale
        attention = torch.softmax(attention_logits, dim=-1)
        return SimpleNamespace(
            logits=logits,
            hidden_states=tuple(hidden_states) if output_hidden_states else None,
            attentions=(attention, attention) if output_attentions else None,
        )


def install_test_models(monkeypatch, *, hidden_a=4, hidden_b=4, fail=False):
    from backend.science.models import model_manager

    models = {
        "fixture/a": TinyCausalModel(hidden_size=hidden_a, scale=1.0, revision="rev-a"),
        "fixture/b": TinyCausalModel(hidden_size=hidden_b, scale=1.7, revision="rev-b"),
    }
    tokenizer_a = TinyTokenizer()
    tokenizer_b = TinyTokenizer()

    def get_model_and_tokenizer(self, repo_id, revision="main"):
        if fail:
            raise OSError("fixture weights are unavailable")
        return models[repo_id], tokenizer_a if repo_id == "fixture/a" else tokenizer_b

    monkeypatch.setattr(
        model_manager.ModelManager,
        "get_model_and_tokenizer",
        get_model_and_tokenizer,
    )


def run_compare():
    return ModelComparisonEngine(backend=HuggingFaceComparisonBackend()).compare(
        prompt="synthetic unit-test input",
        model_a="fixture/a",
        model_b="fixture/b",
        revision_a="rev-a",
        revision_b="rev-b",
    )


def test_success_uses_two_loaded_models_and_measured_tensors(monkeypatch):
    install_test_models(monkeypatch)
    result = run_compare()

    assert result["status"] == "completed"
    assert result["provenance"] == "live"
    assert result["measured"] is True
    assert result["input"]["text"] == "synthetic unit-test input"
    assert result["input"]["sha256"].startswith("sha256:")
    assert result["models"]["a"]["weights_sha256"].startswith("sha256:")
    assert result["models"]["b"]["weights_sha256"].startswith("sha256:")
    assert result["models"]["a"]["weights_sha256"] != result["models"]["b"]["weights_sha256"]
    assert result["predictions"]["model_a_top"]["token"]
    assert isinstance(result["predictions"]["top_token_match"], bool)
    assert isinstance(result["logits"]["kl_divergence_a_to_b"], float)
    assert result["activations"]["layers"]
    assert result["activations"]["layers"][0]["activation_kind"] == "input_to_c_proj"
    assert result["residuals"]["layers"]
    assert result["attention"]["layers"]


def test_missing_weights_return_unavailable_without_numeric_fallback(monkeypatch):
    install_test_models(monkeypatch, fail=True)
    result = run_compare()

    assert result["status"] == "unavailable"
    assert result["provenance"] == "unavailable"
    assert result["measured"] is False
    assert result["metrics_measured"] is False
    assert result["predictions"] == {}
    assert result["residuals"]["layers"] == []
    assert result["attention"]["layers"] == []
    assert "fixture weights are unavailable" in result["reason"]


def test_incompatible_residual_shapes_are_not_coerced_or_zero_filled(monkeypatch):
    install_test_models(monkeypatch, hidden_a=4, hidden_b=3)
    result = run_compare()

    assert result["status"] == "completed"
    assert result["activations"]["layers"] == []
    assert result["activations"]["comparable"] is False
    assert result["residuals"]["layers"] == []
    assert result["residuals"]["comparable"] is False
    assert "hidden dimensions" in result["residuals"]["not_comparable_reason"]
    # Attention tensors have matching shapes, so those numeric measurements can
    # still be compared independently of the residual hidden width.
    assert result["attention"]["layers"]


def test_untrusted_backend_cannot_self_attest_fabricated_values():
    class FabricatedBackend:
        def compare(self, **kwargs):
            return {
                "status": "completed",
                "provenance": "live",
                "attested": True,
                "measured": True,
                "models": {
                    "a": {"weights_sha256": "sha256:" + "0" * 32, "tokenizer_sha256": "sha256:fixture"},
                    "b": {"weights_sha256": "sha256:" + "1" * 32, "tokenizer_sha256": "sha256:fixture"},
                },
                "input": {"sha256": "sha256:fixture"},
                "predictions": {"top_token_match": True},
            }

    engine = ModelComparisonEngine(backend=FabricatedBackend())
    result = engine.compare("input", "a", "b")

    assert engine.compare_available() is False
    assert result["status"] == "unavailable"
    assert result["provenance"] == "unavailable"
    assert result["measured"] is False
    assert result["predictions"] == {}
    assert "custom/self-attested" in result["reason"]


def test_missing_input_is_rejected_without_synthesizing_text():
    result = ModelComparisonEngine(backend=HuggingFaceComparisonBackend()).compare(
        prompt="", model_a="fixture/a", model_b="fixture/b",
    )
    assert result["status"] == "unavailable"
    assert "input" not in result
    assert "non-empty input prompt" in result["reason"]
