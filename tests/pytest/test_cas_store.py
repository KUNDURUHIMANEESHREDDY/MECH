"""Unit tests for Content-Addressed Storage (CAS) & Execution Artifact Store."""

import os
import shutil
import pytest
import torch

from backend.runtime.artifacts.models import ArtifactMetadata, ExecutionArtifact, Provenance
from backend.runtime.artifacts.cas_store import ArtifactStore, compute_artifact_key


@pytest.fixture
def temp_store(tmp_path):
    storage_dir = str(tmp_path / "artifacts")
    db_path = str(tmp_path / "artifacts.db")
    store = ArtifactStore(storage_dir=storage_dir, db_path=db_path, l1_max_entries=10)
    yield store
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir, ignore_errors=True)


def test_cas_key_determinism():
    prov = Provenance(
        model_id="gpt2",
        weights_digest="abc12345",
        precision="float32",
        operation="layer_forward",
        operation_params={"layer": 3},
    )
    digest = prov.compute_digest()

    key1 = compute_artifact_key(
        parent_ids=["embed", "layer_2"],
        operation="layer_forward",
        operation_params={"layer": 3},
        provenance_digest=digest,
        layer=3,
        component="residual",
    )
    key2 = compute_artifact_key(
        parent_ids=["layer_2", "embed"],  # Order shouldn't matter due to sort
        operation="layer_forward",
        operation_params={"layer": 3},
        provenance_digest=digest,
        layer=3,
        component="residual",
    )
    assert key1 == key2
    assert len(key1) == 64  # SHA-256 hex string


def test_artifact_store_put_get_tensor(temp_store):
    prov = Provenance(model_id="gpt2", operation="test_op")
    meta = ArtifactMetadata(name="test_res_L3", component="residual", layer=3)
    tensor = torch.randn(1, 5, 768)

    key = compute_artifact_key(
        parent_ids=["L2"],
        operation="test_op",
        operation_params={},
        provenance_digest=prov.compute_digest(),
        layer=3,
        component="residual",
    )

    art = ExecutionArtifact(
        artifact_id=key,
        metadata=meta,
        provenance=prov,
        parent_ids=["L2"],
        tensor=tensor,
    )

    # Put in store
    returned_key = temp_store.put(art, persist_to_disk=True)
    assert returned_key == key
    assert temp_store.exists(key)

    # Get from store
    fetched = temp_store.get(key)
    assert fetched is not None
    assert fetched.artifact_id == key
    assert fetched.metadata.layer == 3
    assert fetched.tensor is not None
    assert torch.allclose(fetched.tensor, tensor)


def test_artifact_store_search_and_delete(temp_store):
    for layer_i in range(3):
        prov = Provenance(model_id="gpt2", operation="forward")
        meta = ArtifactMetadata(name=f"res_L{layer_i}", component="residual", layer=layer_i)
        key = f"key_layer_{layer_i}"
        art = ExecutionArtifact(
            artifact_id=key,
            metadata=meta,
            provenance=prov,
            parent_ids=[],
            tensor=torch.zeros(1, 2, 4),
        )
        temp_store.put(art, persist_to_disk=True)

    results = temp_store.search(model_id="gpt2", component="residual")
    assert len(results) == 3

    assert temp_store.delete("key_layer_1") is True
    assert temp_store.exists("key_layer_1") is False
    assert len(temp_store.search(model_id="gpt2")) == 2
