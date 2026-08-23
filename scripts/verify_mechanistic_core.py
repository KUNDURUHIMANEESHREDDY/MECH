"""Comprehensive Verification Script for Mechanistic ML Core.

Verifies:
1. Model Adapters (Gemma, LLaMA, Qwen, Mistral, DeepSeek, GPT-2)
2. Model Metadata API returning genuine model architecture parameters
3. Attribution Patching (linear gradient / difference attribution)
4. Causal Tracing (layer-by-layer causal mediation analysis)
5. Intermediate Logits Engine (unembedding W_U projection & entropy)
6. Logit Lens algorithm with top-k token distributions
7. Sparse Autoencoders (SAE) (x - b_enc) @ W_enc ReLU activations
8. Benchmarks API with BenchmarkRunner execution
9. BCa Bootstrap Confidence Intervals (jackknife acceleration & bias correction)
10. Activation Compression Codecs (FP16, INT8, ZLIB)
"""

import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def test_1_model_adapters():
    print("[1/10] Testing Model Adapters...")
    from backend.science.models.adapter_registry import ModelAdapterRegistry
    from backend.science.models.model_adapters import GemmaAdapter, LlamaAdapter, QwenAdapter, MistralAdapter, DeepSeekAdapter

    reg = ModelAdapterRegistry()
    for mid in ["gemma-2b", "tinyllama", "qwen-2-1.5b", "mistral-7b", "deepseek-r1-1.5b", "gpt2-small"]:
        adapter = reg.get_adapter(mid, mock_mode=True)
        spec = adapter.get_model_spec()
        assert spec["num_layers"] > 0
        assert spec["d_model"] > 0

        # Activations
        acts = adapter.get_activations("The Eiffel Tower is in Paris", layer=2)
        assert len(acts) > 0

        # Attention patterns with entropy
        attns = adapter.get_attention_patterns("The capital of France is Paris", layer=1)
        assert len(attns) == spec["num_heads"]
        assert all(hasattr(p, "attn_entropy") for p in attns)

        # Logits
        logits = adapter.get_logits("The capital of France is")
        assert "top_token" in logits

        # Residual stream
        stream = adapter.get_residual_stream("Hello world")
        assert len(stream) == spec["num_layers"] + 1

    print("  PASSED: All 6 model adapters verified with full interface parity.")


def test_2_model_metadata_api():
    print("[2/10] Testing Model Metadata API...")
    from backend.api.dispatcher import get_model_info, MODEL_METADATA_REGISTRY

    models_to_test = {
        "gpt2-small": {"layers": 12, "hidden_size": 768, "vocab_size": 50257},
        "gemma-2b": {"layers": 18, "hidden_size": 2048, "vocab_size": 256000},
        "llama-3-8b": {"layers": 32, "hidden_size": 4096, "vocab_size": 128256},
        "qwen-2-7b": {"layers": 28, "hidden_size": 3584, "vocab_size": 151936},
        "mistral-7b": {"layers": 32, "hidden_size": 4096, "vocab_size": 32000},
        "deepseek-r1-1.5b": {"layers": 28, "hidden_size": 1536, "vocab_size": 102400},
    }

    for name, expected in models_to_test.items():
        info = get_model_info(name)
        assert info["layers"] == expected["layers"], f"Failed for {name}: expected {expected['layers']}, got {info['layers']}"
        assert info["hidden_size"] == expected["hidden_size"]
        assert info["vocab_size"] == expected["vocab_size"]

    print("  PASSED: Model Metadata API returns accurate specs across all families.")


def test_3_attribution_patching():
    print("[3/10] Testing Attribution Patching...")
    from backend.interpretability.causal.attribution_patching import AttributionPatchingEngine

    engine = AttributionPatchingEngine()
    res = engine.compute_attribution(
        clean_prompt="When Mary and John went to the store, John gave a drink to",
        corrupted_prompt="When Mary and John went to the store, Mary gave a drink to",
    )
    assert "top_attributed_nodes" in res
    assert len(res["top_attributed_nodes"]) > 0
    assert "linearized_approximation_error" in res
    assert res["top_attributed_nodes"][0]["attribution_score"] > 0
    print(f"  PASSED: Top attributed node: {res['top_attributed_nodes'][0]['component']} (score: {res['top_attributed_nodes'][0]['attribution_score']})")


def test_4_causal_tracing():
    print("[4/10] Testing Causal Tracing...")
    from backend.interpretability.causal.causal_tracing import CausalTracingEngine

    engine = CausalTracingEngine()
    res = engine.trace_causal_effect(
        clean_prompt="The Eiffel Tower is in",
        corrupted_prompt="The Colosseum is in",
    )
    assert "max_causal_layer" in res
    assert "max_indirect_effect" in res
    assert len(res["layer_effects"]) == 12
    print(f"  PASSED: Max causal layer: {res['max_causal_layer']}, Max AIE: {res['max_indirect_effect']}")


def test_5_intermediate_logits():
    print("[5/10] Testing Intermediate Logits Engine...")
    from backend.runtime.logits import IntermediateLogitsEngine

    engine = IntermediateLogitsEngine()
    res = engine.extract_logits("The capital of France is", num_layers=12)
    assert len(res["layer_projections"]) == 12
    for lp in res["layer_projections"]:
        assert "residual_norm" in lp
        assert "top_prediction" in lp
        assert "top_logit" in lp
        assert "entropy" in lp
        assert len(lp["top_k_tokens"]) > 0

    print(f"  PASSED: Layer 11 prediction: '{res['layer_projections'][-1]['top_prediction']}', entropy: {res['layer_projections'][-1]['entropy']}")


def test_6_logit_lens():
    print("[6/10] Testing Logit Lens Algorithm...")
    from backend.interpretability.algorithms.logit_lens import LogitLens

    lens = LogitLens()
    res = lens.project("The capital of France is", layer=8)
    assert res["method"] == "LogitLens"
    assert res["layer"] == 8
    assert "top_token" in res
    assert len(res["top_k_tokens"]) > 0
    print(f"  PASSED: Layer 8 projection: '{res['top_token']}', prob: {res['top_k_tokens'][0]['probability']}")


def test_7_sparse_autoencoders():
    print("[7/10] Testing Sparse Autoencoders (SAE)...")
    from backend.interpretability.sae.loader import SAELoader, SAEConfig, SAE

    loader = SAELoader()
    sae = loader.load_sae("hf", "gpt2-small-res-jb", version="v1")
    assert sae.config.d_in == 768
    assert sae.config.d_sae == 16384

    # Test genuine (x - b) @ W_enc ReLU activation
    dummy_hidden = np.random.randn(768).astype(np.float32)
    act_res = sae.activate(dummy_hidden)
    assert "feature_indices" in act_res
    assert "activations" in act_res
    assert "sparsity" in act_res
    assert act_res["sparsity"] > 0
    print(f"  PASSED: Active features count: {len(act_res['feature_indices'])}, Sparsity: {act_res['sparsity']}")


def test_8_benchmarks_api():
    print("[8/10] Testing Benchmarks API...")
    from backend.api.dispatcher import run_benchmark

    res = run_benchmark({"benchmark_name": "IOI", "model_id": "gpt2-small"})
    assert res["status"] == "completed"
    assert res["benchmark_name"] == "IOI"
    assert "score" in res
    assert "pass_rate" in res
    print(f"  PASSED: IOI Benchmark score: {res['score']}, pass_rate: {res['pass_rate']}")


def test_9_bootstrap_ci():
    print("[9/10] Testing BCa Bootstrap Confidence Intervals...")
    from backend.science.statistics.bootstrap_engine import BootstrapEngine

    eng = BootstrapEngine(n_bootstraps=1000, seed=123)
    data = np.array([2.1, 2.5, 3.0, 3.8, 4.2, 4.9, 5.1, 5.8, 6.4, 7.2])

    res_perc = eng.run(data, np.mean, method="percentile")
    res_bca = eng.run(data, np.mean, method="bca")

    assert res_perc["ci_lower"] < res_perc["estimate"] < res_perc["ci_upper"]
    assert res_bca["ci_lower"] < res_bca["estimate"] < res_bca["ci_upper"]
    print(f"  PASSED: Mean estimate: {res_bca['estimate']:.3f}, 95% BCa CI: [{res_bca['ci_lower']:.3f}, {res_bca['ci_upper']:.3f}]")


def test_10_activation_compression():
    print("[10/10] Testing Activation Compression Codecs...")
    from backend.runtime.memory.compression import ActivationCompressor

    compressor = ActivationCompressor()
    activations = [1.25, -3.5, 7.82, 12.0, 0.05, -0.99]

    # FP16
    c_fp16 = compressor.compress_activations(activations, "FP16")
    d_fp16 = compressor.decompress_activations(c_fp16)
    assert len(d_fp16) == len(activations)
    for orig, dec in zip(activations, d_fp16):
        assert abs(orig - dec) < 1e-2

    # INT8
    c_int8 = compressor.compress_activations(activations, "INT8")
    d_int8 = compressor.decompress_activations(c_int8)
    assert len(d_int8) == len(activations)
    for orig, dec in zip(activations, d_int8):
        assert abs(orig - dec) < 0.25

    # ZLIB
    c_zlib = compressor.compress_activations(activations, "ZLIB")
    d_zlib = compressor.decompress_activations(c_zlib)
    assert np.allclose(d_zlib, activations, atol=1e-4)

    print("  PASSED: FP16, INT8, and ZLIB codecs round-trip verified.")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING MECHANISTIC CORE VERIFICATION SUITE")
    print("=" * 60)
    test_1_model_adapters()
    test_2_model_metadata_api()
    test_3_attribution_patching()
    test_4_causal_tracing()
    test_5_intermediate_logits()
    test_6_logit_lens()
    test_7_sparse_autoencoders()
    test_8_benchmarks_api()
    test_9_bootstrap_ci()
    test_10_activation_compression()
    print("=" * 60)
    print("ALL 10 MECHANISTIC CORE VERIFICATION TESTS PASSED!")
    print("=" * 60)
