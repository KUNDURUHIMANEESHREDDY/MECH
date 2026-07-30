"""Pytest tests for Unified Model Adapter Layer."""

from __future__ import annotations

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
    for model_id in ["gpt2-small", "gemma-2b", "tinyllama", "qwen-2-1.5b", "mistral-7b", "deepseek-r1-1.5b"]:
        adapter = registry.get_adapter(model_id, mock_mode=True)
        assert isinstance(adapter, ModelAdapter)
        spec = adapter.get_model_spec()
        assert spec["model_id"] == model_id or spec["family"] in ("gpt2", "gemma", "llama", "qwen", "mistral", "deepseek")


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
    adapter = LlamaAdapter(variant="tinyllama", mock_mode=True)
    spec = adapter.get_model_spec()
    assert spec["family"] == "llama"
    patterns = adapter.get_attention_patterns("Hello world test", layer=10)
    assert len(patterns) == 32     # TinyLlama has 32 heads


def test_qwen_adapter_interface():
    adapter = QwenAdapter(variant="qwen-2-1.5b", mock_mode=True)
    spec = adapter.get_model_spec()
    assert spec["family"] == "qwen"
    result = adapter.get_logits("The largest planet is")
    assert "top_token" in result


def test_mistral_adapter_interface():
    adapter = MistralAdapter(variant="mistral-7b", mock_mode=True)
    spec = adapter.get_model_spec()
    assert spec["family"] == "mistral"
    assert spec["d_model"] == 4096


def test_deepseek_adapter_interface():
    adapter = DeepSeekAdapter(variant="deepseek-r1-1.5b", mock_mode=True)
    spec = adapter.get_model_spec()
    assert spec["family"] == "deepseek"
    result = adapter.patch_activation("Hello", layer=5, neuron_index=100, patch_value=2.0)
    assert result.patch_value == 2.0


def test_registry_unknown_model_raises():
    registry = ModelAdapterRegistry()
    try:
        registry.get_adapter("nonexistent-model-xyz", mock_mode=True)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "nonexistent-model-xyz" in str(e)
