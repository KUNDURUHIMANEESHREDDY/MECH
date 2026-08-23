"""Unit tests for Warm Model Knowledge Base, SAE Cache bridge, and Compute Backends."""

import pytest
import os
import shutil

from backend.interpretability.warm_model import (
    WarmModelKnowledgeBase,
    WarmFeatureRecord,
    WarmCircuitRecord,
)
from backend.interpretability.sae.cache import PersistentActivationCache
from backend.runtime.execution_backend import ExecutionBackendManager, CPUBackend


@pytest.fixture
def temp_warm_db(tmp_path):
    db_path = str(tmp_path / "warm_model.db")
    kb = WarmModelKnowledgeBase(db_path=db_path)
    yield kb
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError:
            pass


def test_warm_model_feature_and_circuit_registry(temp_warm_db):
    kb = temp_warm_db

    # 1. Register Feature
    feat = WarmFeatureRecord(
        model_id="gpt2",
        layer=8,
        feature_idx=1420,
        semantic_label="Indirect Object Identification Target",
        description="Fires strongly on duplicated names in IOI prompts",
        confidence=0.96,
        activation_freq=0.034,
        top_activating_tokens=[" Mary", " John", " Alice"],
    )
    kb.register_feature(feat)

    # 2. Register Circuit
    circ = WarmCircuitRecord(
        circuit_id="circ_ioi_001",
        model_id="gpt2",
        name="Greater-Than Prediction Circuit",
        task_name="ioi_task",
        nodes=[{"layer": 8, "head": 6}, {"layer": 9, "head": 9}],
        edges=[{"source": "L8H6", "target": "L9H9", "weight": 0.42}],
        faithfulness_score=0.89,
        recovery_score=0.94,
    )
    kb.register_circuit(circ)

    # 3. Retrieve and verify
    features = kb.get_features(model_id="gpt2", layer=8)
    assert len(features) == 1
    assert features[0].semantic_label == "Indirect Object Identification Target"
    assert features[0].top_activating_tokens == [" Mary", " John", " Alice"]

    circuits = kb.get_circuits(model_id="gpt2")
    assert len(circuits) == 1
    assert circuits[0].name == "Greater-Than Prediction Circuit"
    assert circuits[0].faithfulness_score == 0.89

    # 4. Check model profile summary
    profile = kb.get_model_profile("gpt2")
    assert profile["total_features_labeled"] == 1
    assert profile["total_verified_circuits"] == 1
    assert profile["layers_analyzed"] == [8]


def test_sae_persistent_activation_cache(tmp_path):
    cache_dir = str(tmp_path / "sae_cache")
    sae_cache = PersistentActivationCache(cache_dir=cache_dir)

    sae_version = "v1_topk32"
    dataset_hash = "ioi_dataset_001"
    prompt = "When Mary and John went to the store, Mary gave a drink to"
    activations = {"latents_sparse": [(1420, 4.25), (8901, 1.12)], "l0_norm": 2}

    key = sae_cache.put(
        sae_version=sae_version,
        dataset_hash=dataset_hash,
        prompt=prompt,
        activations=activations,
        model_sha="gpt2_sha",
        layer=8,
    )
    assert len(key) == 64

    # Fetch and verify
    cached_entry = sae_cache.get(
        sae_version=sae_version,
        dataset_hash=dataset_hash,
        prompt=prompt,
    )
    assert cached_entry is not None
    assert cached_entry["activations"]["l0_norm"] == 2


def test_compute_backend_manager():
    manager = ExecutionBackendManager()
    cpu_be = manager.get_backend("cpu")
    assert isinstance(cpu_be, CPUBackend)
    assert cpu_be.is_available() is True
    
    # Test execution
    res = cpu_be.execute(lambda x, y: x + y, 10, 20)
    assert res == 30
