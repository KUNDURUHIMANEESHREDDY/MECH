"""Unit tests for ExecutionDAG, CacheResolver, and Dependency Invalidation."""

import os
import shutil
import pytest

from backend.runtime.artifacts.models import ArtifactMetadata, ExecutionArtifact, Provenance
from backend.runtime.artifacts.cas_store import ArtifactStore
from backend.runtime.dag.execution_dag import DAGNode, ExecutionDAG
from backend.runtime.dag.cache_resolver import CacheResolver


@pytest.fixture
def temp_store(tmp_path):
    storage_dir = str(tmp_path / "artifacts")
    db_path = str(tmp_path / "artifacts.db")
    store = ArtifactStore(storage_dir=storage_dir, db_path=db_path)
    yield store
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir, ignore_errors=True)


def test_topological_sort():
    dag = ExecutionDAG()
    dag.add_node(DAGNode(node_id="embed", node_type="embedding"))
    dag.add_node(DAGNode(node_id="L0", node_type="layer", parent_ids=["embed"]))
    dag.add_node(DAGNode(node_id="L1", node_type="layer", parent_ids=["L0"]))
    dag.add_node(DAGNode(node_id="logits", node_type="logit_lens", parent_ids=["L1"]))

    order = [n.node_id for n in dag.topological_sort()]
    assert order == ["embed", "L0", "L1", "logits"]


def test_dag_dependency_invalidation_invariant(temp_store):
    """Verifies that an intervention at Layer 6 invalidates downstream nodes 7..11
    while preserving upstream nodes 0..6 in the cache.
    """
    resolver = CacheResolver(store=temp_store)
    prov = Provenance(model_id="gpt2", weights_digest="digest123")

    # 1. Build Clean 12-layer DAG
    clean_dag = ExecutionDAG.build_transformer_pipeline(
        num_layers=12,
        prompt_tokens_hash="prompt_hash_1",
    )

    # Initial resolution: Nothing is cached yet
    plan_1 = resolver.resolve(clean_dag, prov)
    assert plan_1.total_nodes_count == 14  # embed + 12 layers + logits
    assert plan_1.cached_nodes_count == 0
    assert plan_1.uncomputed_nodes_count == 14

    # 2. Simulate storing clean artifacts
    for node_id, cas_key in plan_1.node_cas_keys.items():
        meta = ArtifactMetadata(name=node_id, component="residual")
        art = ExecutionArtifact(artifact_id=cas_key, metadata=meta, provenance=prov, data={"status": "computed"})
        temp_store.put(art, persist_to_disk=False)

    # Clean DAG resolution now has 100% cache hit!
    plan_2 = resolver.resolve(clean_dag, prov)
    assert plan_2.is_fully_cached()
    assert plan_2.cached_nodes_count == 14
    assert plan_2.uncomputed_nodes_count == 0

    # 3. Build Intervened DAG (ablation after Layer 6)
    intervened_dag = ExecutionDAG.build_transformer_pipeline(
        num_layers=12,
        prompt_tokens_hash="prompt_hash_1",
        interventions={6: {"op": "ablation", "scale": 0.0}},
    )

    plan_3 = resolver.resolve(intervened_dag, prov)

    # Invariant verification:
    # Upstream nodes ("embed", "layer_0" ... "layer_6") MUST be cached hits!
    upstream_expected = ["embed"] + [f"layer_{i}" for i in range(7)]
    for u_node in upstream_expected:
        assert u_node in plan_3.cached_node_ids, f"Upstream node {u_node} should be cached!"

    # Intervened node ("intervention_L6") and all downstream nodes ("layer_7".."layer_11", "unembed_logits")
    # MUST be uncomputed misses due to dependency hash propagation!
    downstream_node_ids = [n.node_id for n in plan_3.uncomputed_nodes]
    assert "intervention_L6" in downstream_node_ids
    for d_idx in range(7, 12):
        assert f"layer_{d_idx}" in downstream_node_ids
    assert "unembed_logits" in downstream_node_ids

    # Upstream count = 8 (embed + 7 layers 0..6)
    assert plan_3.cached_nodes_count == 8
    # Downstream count = 1 intervention + 5 layers (7..11) + 1 logits = 7
    assert plan_3.uncomputed_nodes_count == 7
