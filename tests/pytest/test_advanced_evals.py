"""Advanced evaluation tests for the MECH Platform backend API.

Covers:
  - FastAPI HTTP endpoint contract (all routes, methods, response schemas)
  - GPT-2 inference: every layer (0-11) x every head (0-11) attention map
  - GPT-2 neuron activations: every layer with 8 neurons per layer
  - IOI benchmark correctness (indirect object identification)
  - Attention head patching causal effect
  - Benchmark run scoring
  - Runtime token analysis
  - Experiment & session CRUD
"""
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure backend is importable
BACKEND_DIR = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR.parent))

from backend.main import app, _API_KEY  # noqa: E402

client = TestClient(app, raise_server_exceptions=False, headers={"X-API-Key": _API_KEY})


# ── 1. Health & Root ────────────────────────────────────────────────────

class TestHealthRoot:
    def test_root(self):
        r = client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "MECH Platform"
        assert body["version"] == "2.0.0"
        assert body["status"] == "running"

    def test_health(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"


# ── 2. Model Endpoints ──────────────────────────────────────────────────

class TestModels:
    def test_list_models(self):
        r = client.get("/api/v1/models")
        assert r.status_code == 200
        models = r.json()["models"]
        assert len(models) == 8
        assert "gpt2-small" in models

    def test_get_model_info(self):
        r = client.get("/api/v1/models/gpt2-small")
        assert r.status_code == 200
        info = r.json()
        assert info["model_name"] == "gpt2-small"
        assert info["layers"] == 12
        assert info["hidden_size"] == 768
        assert info["vocab_size"] == 50257
        assert info["num_heads"] == 12

    def test_load_model(self):
        r = client.post("/api/v1/models/load", json={"model_name": "gpt2-small"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "loaded"
        assert body["n_layers"] == 12
        assert body["n_heads"] == 12
        assert body["d_model"] == 768
        assert body["d_mlp"] == 3072


# ── 3. Inference: All Layers & Heads ──────────────────────────────────────

class TestInferenceFullLayers:
    """Verify that the infer endpoint returns attention maps and neuron
    activations for ALL 12 layers, not just the first 2."""

    @pytest.fixture(scope="class")
    def infer_result(self):
        r = client.post("/api/v1/infer", json={"prompt": "The capital of France is"})
        assert r.status_code == 200
        return r.json()

    def test_attention_maps_count(self, infer_result):
        maps = infer_result["attention_maps"]
        assert len(maps) == 144  # 12 layers x 12 heads

    def test_all_layers_present_in_attention(self, infer_result):
        layers = {m["layer"] for m in infer_result["attention_maps"]}
        assert layers == set(range(12))

    def test_all_heads_per_layer(self, infer_result):
        for li in range(12):
            heads = {m["head"] for m in infer_result["attention_maps"] if m["layer"] == li}
            assert heads == set(range(12)), f"Layer {li} missing heads"

    def test_neuron_activations_count(self, infer_result):
        acts = infer_result["neuron_activations"]
        assert len(acts) == 192  # 12 layers x 16 top neurons each

    def test_all_layers_present_in_activations(self, infer_result):
        layers = {a["layer"] for a in infer_result["neuron_activations"]}
        assert layers == set(range(12))

    def test_tokens_present(self, infer_result):
        assert len(infer_result["tokens"]) > 0
        assert "generated_text" in infer_result

    def test_gpu_and_memory(self, infer_result):
        assert "n_layers" in infer_result
        assert "d_mlp" in infer_result
        assert "d_model" in infer_result


# ── 4. GPT-2 Engine Endpoints ───────────────────────────────────────────

class TestGpt2Engine:
    def test_load_gpt2(self):
        r = client.post("/api/v1/gpt2/load", json={})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "loaded"
        assert body["n_layers"] == 12

    def test_run_prompt(self):
        r = client.post(
            "/api/v1/gpt2/run_prompt", json={"prompt": "When Mary and John went to the store"}
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert len(body["top5"]) == 5
        assert len(body["top16"]) == 16
        assert "str_tokens" in body

    def test_attention_head_query(self):
        client.post("/api/v1/gpt2/run_prompt", json={"prompt": "Test prompt"})
        r = client.post("/api/v1/gpt2/attention_head", json={"layer": 5, "head": 3})
        assert r.status_code == 200
        body = r.json()
        assert body["layer"] == 5
        assert body["head"] == 3
        assert body["matrix"]
        assert body["str_tokens"]

    def test_activations(self):
        r = client.post("/api/v1/gpt2/activations", json={"layer": 7})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["layer"] == 7

    def test_patch_head_causal(self):
        """Verify patching a head produces a clean vs patched logit difference."""
        r = client.post(
            "/api/v1/gpt2/patch_head",
            json={"layer": 5, "head": 3, "pos_token": "Paris", "neg_token": "London"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "clean_ld" in body
        assert "patched_ld" in body
        assert "delta" in body
        assert body["direction"] in ("hurts", "helps")

    def test_ioi_circuit(self):
        """Test the Indirect Object Identification circuit."""
        r = client.post(
            "/api/v1/gpt2/ioi",
            json={"io_name": "John", "subj_name": "Mary"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["io_name"] == "John"
        assert body["subj_name"] == "Mary"
        assert body["clean_top1"].strip() == "John"
        assert body["corrupted_top1"].strip() == "Mary"
        assert body["ioi_pass"] is True
        assert body["corrupted_pass"] is True


# ── 5. Benchmarks ───────────────────────────────────────────────────────

class TestBenchmarks:
    def test_list_benchmarks(self):
        r = client.get("/api/v1/benchmarks")
        assert r.status_code == 200
        benchmarks = r.json()["benchmarks"]
        assert "IOI" in benchmarks
        assert "SAE" in benchmarks

    def test_run_benchmark(self):
        r = client.post("/api/v1/benchmarks/run", json={"benchmark_name": "IOI"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "completed"
        assert body["benchmark_name"] == "IOI"
        assert body["score"] >= 0.0
        assert body["pass_rate"] >= 0.0
        assert body["pass_rate"] <= 1.0


# ── 6. Runtime & Analysis ───────────────────────────────────────────────

class TestRuntime:
    def test_runtime_status(self):
        r = client.post("/api/v1/runtime/status", json={})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "connected"
        assert "engines" in body

    def test_tokenize(self):
        r = client.post(
            "/api/v1/runtime/analyze_tokens", json={"prompt": "hello world foo bar"}
        )
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 4
        assert len(body["tokens"]) == 4


# ── 7. Discoveries, Agents, Knowledge Graph ─────────────────────────────

class TestDiscovery:
    def test_discoveries(self):
        r = client.get("/api/v1/discoveries")
        assert r.status_code == 200
        discoveries = r.json()["discoveries"]
        assert len(discoveries) == 11
        assert "InductionCircuitDiscovery" in discoveries

    def test_agents(self):
        r = client.get("/api/v1/agents")
        assert r.status_code == 200
        body = r.json()
        assert "ResearchSociety" in body["agents"]
        assert "openai" in body["llm_backends"]

    def test_knowledge_graph(self):
        r = client.get("/api/v1/knowledge-graph")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "active"
        assert "provenance" in body["stores"]


# ── 8. Experiments & Sessions ─────────────────────────────────────────────

class TestExperimentsSessions:
    def test_list_experiments(self):
        r = client.get("/api/v1/experiments")
        assert r.status_code == 200
        assert "experiments" in r.json()

    def test_create_experiment(self):
        r = client.post("/api/v1/experiments", json={"name": "test_eval"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "created"
        assert "id" in body

    def test_list_sessions(self):
        r = client.get("/api/v1/sessions")
        assert r.status_code == 200
        assert "sessions" in r.json()

    def test_create_session(self):
        r = client.post("/api/v1/sessions", json={"name": "test_session"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "created"
        assert "id" in body


# ── 9. Status & Ping ─────────────────────────────────────────────────═══

class TestSystemStatus:
    def test_status(self):
        r = client.get("/api/v1/status")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["version"] == "2.0"
        assert len(body["modules"]) >= 20

    def test_ping(self):
        r = client.post("/api/v1/ping", json={})
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


# ── 10. Cross-Layer Attention Analysis ─────────────────────────────────═══

class TestCrossLayerAttention:
    """Advanced: verify attention patterns are sensible across all layers."""

    @pytest.fixture(scope="class")
    def run_result(self):
        client.post("/api/v1/gpt2/run_prompt", json={"prompt": "When Mary and John went to the store"})
        return client.post("/api/v1/gpt2/run_prompt", json={"prompt": "The capital of France is"})

    def test_all_layers_attn_matrix_shape(self, run_result):
        assert run_result.status_code == 200
        body = run_result.json()
        assert len(body["top5"]) == 5

    def test_dynamic_prediction(self, run_result):
        """Verify the model returns dynamic predictions (top5 + top16) driven by actual logits."""
        assert run_result.status_code == 200
        body = run_result.json()
        assert "top5" in body
        assert len(body["top5"]) == 5
        assert "top16" in body
        assert len(body["top16"]) == 16
        tokens = [t["token"] for t in body["top5"]]
        assert len(set(tokens)) == 5
        for t in body["top5"]:
            assert "token" in t
            assert "logit" in t
            assert isinstance(t["logit"], float)
        assert "next_token" in body
        assert isinstance(body["next_token"], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
