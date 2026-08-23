"""DAG-Integrated Mechanistic Interpretability Test Suite.

Verifies:
1. Causal Activation Patching over the persistent execution DAG with automatic upstream cache reuse.
2. SAE feature steering and automatic circuit registration to WarmModelKnowledgeBase.
"""

import gc
import tempfile
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.interpretability.causal.dag_intervention_engine import DAGInterventionEngine
from backend.interpretability.warm_model import WarmModelKnowledgeBase
from backend.runtime.artifacts.cas_store import ArtifactStore
from backend.runtime.memory.disk_weight_store import DiskWeightStore
from backend.runtime.memory.layer_pager import LayerPager


def test_dag_activation_patching_and_circuit_sync():
    """Verifies that causal activation patching runs over DAG and syncs findings to WarmModelKnowledgeBase."""
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()

    clean_prompt = "The capital of France is Paris."
    corrupted_prompt = "The capital of Italy is Rome."
    target_token_id = tokenizer.encode(" Paris")[0]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        store = ArtifactStore(storage_dir=f"{tmpdir}/cas", db_path=f"{tmpdir}/cas.db")
        warm_db = WarmModelKnowledgeBase(db_path=f"{tmpdir}/warm.db")
        pager = LayerPager(device="cpu", dtype=torch.float32, store=store)
        engine = DAGInterventionEngine(store=store, warm_db=warm_db, pager=pager)

        # Run activation patching at layer 4
        result = engine.run_activation_patching(
            model=model,
            tokenizer=tokenizer,
            clean_prompt=clean_prompt,
            corrupted_prompt=corrupted_prompt,
            target_layer=4,
            target_token_id=target_token_id,
        )

        assert result.clean_prompt == clean_prompt
        assert result.target_layer == 4
        assert result.upstream_cache_hits == 4
        assert result.downstream_recomputed == 8
        assert isinstance(result.recovery_ratio, float)

        # Check if circuit was recorded in WarmModelKnowledgeBase
        circuits = warm_db.get_circuits()
        assert len(circuits) >= 1
        assert circuits[0].discovered_by == "dag_activation_patching"

        del engine, pager, warm_db, store
        gc.collect()


def test_dag_sae_feature_steering_and_warm_model():
    """Verifies that SAE feature steering applies intervention and registers feature in WarmModelKnowledgeBase."""
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()

    prompt = "Large language models demonstrate emergent in-context learning."

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        store = ArtifactStore(storage_dir=f"{tmpdir}/cas", db_path=f"{tmpdir}/cas.db")
        warm_db = WarmModelKnowledgeBase(db_path=f"{tmpdir}/warm.db")
        pager = LayerPager(device="cpu", dtype=torch.float32, store=store)
        engine = DAGInterventionEngine(store=store, warm_db=warm_db, pager=pager)

        hidden_dim = model.config.n_embd
        steering_vector = torch.randn(1, 1, hidden_dim)

        out = engine.run_sae_feature_steering(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            layer=5,
            feature_direction=steering_vector,
            coefficient=3.0,
            semantic_label="in_context_induction_feature",
        )

        assert out.layers_executed == model.config.n_layer
        assert out.traces[5].status == "INTERVENTION"

        # Check that the feature is stored in WarmModelKnowledgeBase
        features = warm_db.get_features(model_id=model_id)
        assert len(features) >= 1
        assert features[0].semantic_label == "in_context_induction_feature"
        assert features[0].layer == 5

        del engine, pager, warm_db, store
        gc.collect()
