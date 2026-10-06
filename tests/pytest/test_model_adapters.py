"""Pytest tests for Unified Model Adapter Layer."""

from __future__ import annotations

import pytest

from science.models.adapter_base import ModelAdapter
from science.models.adapter_registry import ModelAdapterRegistry
from science.models.gpt2_adapter import GPT2Adapter
from science.models.model_adapters import (
    DeepSeekAdapter, GemmaAdapter, LlamaAdapter, MistralAdapter, QwenAdapter,
)


def test_registry_lists_all_families():
    registry = ModelAdapterRegistry()
    adapters = registry.list_adapters()
    families = {a["family"] for a in adapters}
    assert families == {"gpt2", "gemma", "llama", "qwen", "mistral", "deepseek"}


def test_registry_resolves_all_model_ids():
    registry = ModelAdapterRegistry()
    for model_id in ["gpt2-small", "gemma-2b", "llama-3.2-1b",
                     "qwen2.5-1.5b", "mistral-7b", "deepseek-r1-1.5b"]:
        adapter = registry.get_adapter(model_id, mock_mode=True)
        assert isinstance(adapter, ModelAdapter)
        spec = adapter.get_model_spec()
        # Was `spec["model_id"] == model_id or spec["family"] in (...)`. That
        # `or` made the assertion close to vacuous: the second clause is true
        # for every adapter in the registry, so the check passed even when the
        # resolved spec was a different model from the one requested -- which is
        # the identity mismatch this registry exists to prevent.
        assert spec["model_id"] == model_id, (
            f"asked for {model_id!r}, resolved to {spec['model_id']!r}")


def test_a_shorthand_alias_resolves_to_the_same_model():
    """`tinyllama` keeps working, and points at the real spec."""
    registry = ModelAdapterRegistry()
    spec = registry.get_adapter("tinyllama", mock_mode=True).get_model_spec()
    assert spec["model_id"] == "tinyllama-1.1b"
    assert spec["family"] == "llama"


@pytest.mark.parametrize("model_id", [
    "llama-3-8b", "llama-3-70b", "qwen-2-1.5b", "qwen-2-7b",
    "qwen-2.5-7b", "mixtral-8x7b", "deepseek-v2-7b",
])
def test_registry_no_longer_resolves_ids_it_never_had_specs_for(model_id):
    """These ids are gone on purpose. This pins the loss so it stays visible.

    Each is a distinct model, not a renamed one -- `qwen-2-1.5b` is Qwen2-1.5B
    and `qwen2.5-1.5b` is Qwen2.5-1.5B, with different weights. Re-adding one
    means adding a real spec with correct architecture numbers.

    When someone adds one, delete it from this list. If a model is ever pointed
    at a *neighbouring* spec just to keep an id resolving, that is the identity
    mismatch, and this test is not the place to make it pass.
    """
    registry = ModelAdapterRegistry()
    with pytest.raises(ValueError, match=model_id):
        registry.get_adapter(model_id, mock_mode=True)


def test_every_advertised_id_actually_resolves():
    """The guard that would have caught the drift in the first place."""
    registry = ModelAdapterRegistry()
    for row in registry.list_adapters():
        adapter = registry.get_adapter(row["model_id"], mock_mode=True)
        assert adapter.get_model_spec()["family"] == row["family"], row["model_id"]


def test_gpt2_adapter_activations():
    adapter = GPT2Adapter(variant="small", mock_mode=True)
    results = adapter.get_activations("The Eiffel Tower is in", layer=8)
    assert len(results) > 0
    assert all(hasattr(r, "activation_value") for r in results)


def test_gpt2_adapter_attention_patterns():
    adapter = GPT2Adapter(variant="small", mock_mode=True)
    patterns = adapter.get_attention_patterns("When Mary and John went", layer=5)
    assert len(patterns) == 12     # GPT-2 Small has 12 heads
    assert all(hasattr(p, "attn_entropy") for p in patterns)


def test_gpt2_adapter_patch_activation():
    adapter = GPT2Adapter(variant="small", mock_mode=True)
    result = adapter.patch_activation("The capital of France is", layer=9, neuron_index=42, patch_value=3.5)
    assert result.delta == round(3.5 - result.original_logit, 4)


def test_gpt2_adapter_logits():
    adapter = GPT2Adapter(variant="small", mock_mode=True)
    result = adapter.get_logits("The Eiffel Tower is in")
    assert "top_token" in result
    assert result["top_token"] in (" Paris", "Paris")


def test_gpt2_adapter_residual_stream():
    adapter = GPT2Adapter(variant="small", mock_mode=True)
    stream = adapter.get_residual_stream("Hello world")
    assert len(stream) == adapter.spec.num_layers + 1


def test_gemma_adapter_interface():
    adapter = GemmaAdapter(variant="gemma-2b", mock_mode=True)
    spec = adapter.get_model_spec()
    assert spec["family"] == "gemma"
    assert spec["num_layers"] == 18
    activations = adapter.get_activations("The capital is", layer=5)
    assert len(activations) > 0


def test_llama_adapter_interface():
    adapter = LlamaAdapter(variant="tinyllama-1.1b", mock_mode=True)
    spec = adapter.get_model_spec()
    assert spec["family"] == "llama"
    assert spec["model_id"] == "tinyllama-1.1b"
    assert spec["num_heads"] == 32     # TinyLlama's real head count

    # The head count is asserted on the *spec*, not on the length of
    # `get_attention_patterns()`. This used to be `assert len(patterns) == 32`,
    # which was really a claim about the return value of a method that, in
    # mock_mode, has no weights to read heads from -- so it was asserting a
    # coincidence and reporting failures in a model's architecture table.
    #
    # `per_head=False` returns one averaged pattern by design, and says so in
    # its docstring. Under mock_mode the method refuses instead, and refusing
    # is the correct behaviour: there is no attention to report.
    refusal = adapter.get_attention_patterns("Hello world test", layer=10)
    assert isinstance(refusal, dict), (
        f"a mock_mode adapter must refuse rather than return patterns: "
        f"{refusal!r}")
    assert refusal["measured"] is False
    assert refusal["publication_eligible"] is False
    assert refusal["status"] != "completed"
    assert "mock_mode" in refusal["reason"]


def test_qwen_adapter_interface():
    adapter = QwenAdapter(variant="qwen2.5-1.5b", mock_mode=True)
    spec = adapter.get_model_spec()
    assert spec["family"] == "qwen"
    assert spec["model_id"] == "qwen2.5-1.5b"
    assert spec["d_model"] == 1536

    # Was `assert "top_token" in result`. A top token is a *measurement*, and
    # this adapter has no weights loaded, so the correct result is a refusal
    # that says so. Asserting the measurement here is what kept the fiction
    # alive: the old simulated adapters produced a plausible token for a model
    # that was never run.
    result = adapter.get_logits("The largest planet is")
    assert "top_token" not in result, (
        f"a mock_mode adapter returned a top_token: {result!r}")
    assert result["measured"] is False
    assert result["publication_eligible"] is False
    # `provenance` is "synthetic" rather than "unavailable" here: the adapter
    # did not fail to measure, it was asked to simulate and is labelling its
    # own output as generated. Both are non-evidence; the distinction is which
    # of the two happened, and this one is worth being able to tell.
    assert result["provenance"] in ("synthetic", "unavailable")
    assert result["provenance"] != "live"


def test_mock_mode_adapters_refuse_rather_than_simulate():
    """The property the two tests above check one method at a time.

    Every adapter method has to route through `_require_model()`, or one of
    them will keep answering from thin air while its siblings refuse.
    """
    for adapter in (GemmaAdapter(variant="gemma-2b", mock_mode=True),
                    LlamaAdapter(variant="llama-3.2-1b", mock_mode=True),
                    QwenAdapter(variant="qwen2.5-1.5b", mock_mode=True),
                    MistralAdapter(variant="mistral-7b", mock_mode=True),
                    DeepSeekAdapter(variant="deepseek-r1-1.5b", mock_mode=True)):
        logits = adapter.get_logits("The capital of France is")
        assert "top_token" not in logits, f"{adapter.spec.model_id} simulated"
        assert logits["measured"] is False, adapter.spec.model_id


def test_mistral_adapter_interface():
    adapter = MistralAdapter(variant="mistral-7b", mock_mode=True)
    spec = adapter.get_model_spec()
    assert spec["family"] == "mistral"
    assert spec["d_model"] == 4096


def test_deepseek_adapter_interface():
    adapter = DeepSeekAdapter(variant="deepseek-r1-1.5b", mock_mode=True)
    spec = adapter.get_model_spec()
    assert spec["family"] == "deepseek"
    assert spec["model_id"] == "deepseek-r1-1.5b"

    # Was `assert result.patch_value == 2.0`, which requires a patched
    # activation to come back from an adapter with no weights loaded. It could
    # only have passed by echoing the requested value back -- a measurement
    # that confirms whatever it is handed, which is the property this codebase
    # exists to eliminate.
    #
    # The correct result is a refusal that names the reason.
    result = adapter.patch_activation("Hello", layer=5, neuron_index=100,
                                      patch_value=2.0)
    assert isinstance(result, dict), f"expected a refusal, got {result!r}"
    assert "patch_value" not in result
    assert result["measured"] is False
    assert result["publication_eligible"] is False
    assert "mock_mode" in result["reason"]


def test_registry_unknown_model_raises():
    registry = ModelAdapterRegistry()
    try:
        registry.get_adapter("nonexistent-model-xyz", mock_mode=True)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "nonexistent-model-xyz" in str(e)
