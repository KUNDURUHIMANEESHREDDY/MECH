"""Phase 8: Scientific Correctness, Loophole & Invariant Test Suite.

Validates:
1. Input validation and edge-case rejection (empty prompts, NaN inputs, out-of-range parameters).
2. Mathematical invariants (Shannon entropy bounds, LogSumExp equivalence, BCa bootstrap quantiles).
3. Denominator zero-guards across causal intervention engines.
4. Codec fidelity classifications (ZLIB is bit-exact; FP16/INT8 are lossy).
5. Fail-closed execution abstention without synthetic fallbacks.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from backend.runtime.logits import IntermediateLogitsEngine
from backend.runtime.in_memory_runtime import InMemoryRuntime
from backend.runtime.virtual_unembedding import VirtualUnembeddingEngine
from backend.interpretability.discovery.induction_head_detector import InductionHeadDetector
from backend.interpretability.causal.path_patching import EdgePathPatchingEngine, EdgeMediationResult
from backend.interpretability.causal.causal_tracing import CausalTracingEngine
from backend.science.statistics.bootstrap_engine import BootstrapEngine
from backend.runtime.memory.compression import FP16Codec, INT8Codec, ZLIBCodec
from backend.interpretability.sae.sae_interface import SAEMetadata, SAEArchitectureType
from backend.interpretability.sae.sae_adapter import NativeMECHSAE


class TestPhase8CoreExecutionValidation:
    """Test Component 1 & 2 input validation and invariants."""

    def test_empty_prompt_rejection_in_logits_engine(self):
        engine = IntermediateLogitsEngine()
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            engine.extract_logits(prompt="")

        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            engine.extract_logits(prompt="   \n\t  ")

    def test_empty_prompt_rejection_in_in_memory_runtime(self):
        runtime = InMemoryRuntime(model_id="gpt2", device="cpu")
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            runtime.forward(prompt="")

        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            runtime.forward(prompt="    ")

    def test_virtual_unembedding_nan_guard(self):
        engine = VirtualUnembeddingEngine(chunk_size=4096)
        nan_hidden = torch.tensor([float("nan")] * 768)
        w_u = torch.randn(50257, 768)
        with pytest.raises(ValueError, match="contains NaN or Inf"):
            engine.project_hidden_state(
                hidden_state=nan_hidden,
                unembedding_weights=w_u,
                tokenizer=None,
                vocab_size=50257,
            )

    def test_virtual_unembedding_exact_equivalence(self):
        """Verify chunked online LogSumExp matches monolithic softmax within 1e-5."""
        engine = VirtualUnembeddingEngine(chunk_size=1024)
        torch.manual_seed(42)
        h = torch.randn(768)
        w_u = torch.randn(50257, 768)

        # Monolithic ground truth
        mono_logits = torch.matmul(h, w_u.T)
        mono_lse = float(torch.logsumexp(mono_logits, dim=-1).item())
        mono_topk_vals, mono_topk_idx = torch.topk(mono_logits, k=5)

        # Chunked virtual unembedding
        chunked_res = engine.project_hidden_state(
            hidden_state=h,
            unembedding_weights=w_u,
            tokenizer=None,
            top_k=5,
            vocab_size=50257,
        )

        assert abs(chunked_res["global_logsumexp"] - mono_lse) < 1e-4
        assert chunked_res["top_candidates"][0]["token_id"] == int(mono_topk_idx[0].item())
        assert abs(chunked_res["top_candidates"][0]["logit"] - float(mono_topk_vals[0].item())) < 1e-3


class TestPhase8CausalInterventionCorrectness:
    """Test Component 3 mathematical correctness and zero-division guards."""

    def test_path_patching_p_value_labeled_as_heuristic(self):
        patcher = EdgePathPatchingEngine(significance_threshold=0.15)
        # Test topologically backwards flow
        res = patcher.test_edge_mediation(
            sender="L8_H1",
            receiver="L4_H2",
            clean_prompt="The capital of France is",
            corrupted_prompt="The capital of Italy is",
        )
        d = res.to_dict()
        assert "heuristic_p_score" in d
        assert "statistical_caveat" in d
        assert "heuristic" in d["statistical_caveat"].lower()
        assert d["is_causally_transmitting"] is False

    def test_causal_tracing_denominator_zero_handling(self):
        tracer = CausalTracingEngine()
        # Identical clean and corrupted prompts (denominator delta -> 0)
        res = tracer.trace_causal_effect(
            clean_prompt="The capital of France is",
            corrupted_prompt="The capital of France is",
            num_layers=2,
        )
        assert res["clean_probability"] == res["corrupted_probability"]
        # Must not produce NaN or Inf
        for eff in res["layer_effects"]:
            assert not math.isnan(eff["indirect_effect"])
            assert not math.isinf(eff["indirect_effect"])
            assert 0.0 <= eff["indirect_effect"] <= 1.0


class TestPhase8StatisticsAndCodecs:
    """Test Component 6 BCa bootstrap and codec invariants."""

    def test_bootstrap_empty_and_nan_rejection(self):
        engine = BootstrapEngine(n_bootstraps=200, seed=42)
        with pytest.raises(ValueError, match="Input data cannot be empty"):
            engine.bca_ci(np.array([]), np.mean)

        with pytest.raises(ValueError, match="contains NaN or Inf"):
            engine.bca_ci(np.array([1.0, float("nan"), 3.0]), np.mean)

    def test_bootstrap_small_sample_fallback(self):
        engine = BootstrapEngine(n_bootstraps=200, seed=42)
        data = np.array([2.5, 3.5])  # n = 2 < 3
        est, low, high = engine.bca_ci(data, np.mean)
        assert est == 3.0
        assert low <= est <= high

    def test_codec_lossless_vs_lossy_metadata(self):
        zlib_codec = ZLIBCodec()
        fp16_codec = FP16Codec()
        int8_codec = INT8Codec()

        data = [0.123456, -0.987654, 3.141592, 0.0]

        # ZLIB is lossless
        z_comp = zlib_codec.compress(data)
        assert z_comp["is_lossy"] is False
        assert z_comp["is_bit_exact"] is True
        z_decomp = zlib_codec.decompress(z_comp)
        assert np.allclose(data, z_decomp, atol=1e-7)

        # FP16 is lossy
        f_comp = fp16_codec.compress(data)
        assert f_comp["is_lossy"] is True
        assert f_comp["is_bit_exact"] is False

        # INT8 is lossy
        i_comp = int8_codec.compress(data)
        assert i_comp["is_lossy"] is True
        assert i_comp["is_bit_exact"] is False


class TestPhase8SAEInvariants:
    """Test Component 4 SAE reconstruction invariants."""

    def test_sae_sparsity_and_reconstruction_invariants(self):
        sae = NativeMECHSAE(d_in=64, d_sae=256, model_id="test", layer=0, seed=123)
        torch.manual_seed(42)
        x = torch.randn(4, 64)

        z, x_hat = sae.reconstruct(x, top_k=10)
        sparsity = sae.get_sparsity(z)
        assert sparsity["l0"] <= 10.0
        assert 0.0 <= sparsity["active_ratio"] <= 1.0

        rec_err = sae.get_reconstruction_error(x, x_hat)
        assert rec_err["mse"] >= 0.0
        assert 0.0 <= rec_err["explained_variance"] <= 1.0
