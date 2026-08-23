"""Unit and integration tests for Redundant & Backup Head Mechanism Discovery."""

import pytest
from backend.science.redundancy.backup_head_engine import (
    CompensatoryNodeResponse,
    SingleKnockoutAnalysis,
    CombinatorialKnockoutLevel,
    RedundantSubnetworkEnvelope,
    RedundantBackupDiscoveryEngine,
)
from backend.core.capability_registry import CapabilityRegistry
from fastapi.testclient import TestClient
from backend.main import app


def test_backup_head_discovery_engine():
    engine = RedundantBackupDiscoveryEngine(seed=42)

    circuit_nodes = [
        {"id": "node_L8_H5", "data": {"label": "L8_H5", "attribution": 0.52}},
        {"id": "node_L6_MLP", "data": {"label": "L6_MLP", "attribution": 0.44}},
    ]

    envelope = engine.discover_redundant_backup_circuits(
        circuit_nodes=circuit_nodes,
        model_name="gpt2",
        prompt="The Eiffel Tower is located in the city of",
        target_token=" Paris",
    )

    assert isinstance(envelope, RedundantSubnetworkEnvelope)
    assert envelope.backup_capacity_ratio >= 0.60
    assert envelope.self_repair_resilience_tier in ("HIGH_RESILIENCE_SELF_REPAIR", "MODERATE_REDUNDANCY")
    assert envelope.critical_collapse_order == 3
    assert len(envelope.tier_1_active_backups) >= 1
    assert len(envelope.single_knockout_analyses) >= 1
    assert len(envelope.candidate_backup_components) >= 2
    assert len(envelope.causally_confirmed_repair_mechanisms) >= 1

    # Check multi-pathway graph
    mpg = envelope.multi_pathway_compensation_graph
    assert "pathways" in mpg
    assert "downstream_heads" in mpg["pathways"]
    assert "same_layer_sibling_heads" in mpg["pathways"]
    assert "downstream_mlps" in mpg["pathways"]
    assert "earlier_feedforward" in mpg["pathways"]
    assert "distributed_residual_bypass" in mpg["pathways"]

    # Check active compensation response
    analysis = envelope.single_knockout_analyses[0]
    assert analysis.ablated_primary_node == "L8_H5"
    assert analysis.total_downstream_compensation > 0
    assert len(analysis.candidate_compensatory_nodes) >= 4
    assert len(analysis.causally_confirmed_nodes) >= 1

    # Check Causal Necessity + Sufficiency Proof
    confirmed_node = analysis.causally_confirmed_nodes[0]
    c_val = confirmed_node.causal_validation
    assert c_val is not None
    assert c_val.is_necessity_proven is True
    assert c_val.is_sufficiency_proven is True
    assert c_val.necessity_drop_on_candidate_ko > 1.0
    assert c_val.sufficiency_gain_on_restoration > 1.0
    assert c_val.causal_verdict == "CONFIRMED_SELF_REPAIR_MECHANISM"

    # Check combinatorial knockout progression
    assert len(envelope.combinatorial_knockout_grid) == 3
    order1 = envelope.combinatorial_knockout_grid[0]
    order2 = envelope.combinatorial_knockout_grid[1]
    order3 = envelope.combinatorial_knockout_grid[2]

    assert order1.remaining_faithfulness > order2.remaining_faithfulness > order3.remaining_faithfulness
    assert order3.collapse_status == "TOTAL_COLLAPSE"



def test_backup_tool_in_capability_registry():
    registry = CapabilityRegistry()
    assert registry.get_tool("discover_backup_redundant_circuits") is not None

    res = registry.execute_tool("discover_backup_redundant_circuits", {
        "circuit_nodes": [{"id": "node_L8_H5", "data": {"label": "L8_H5", "attribution": 0.5}}],
        "model_name": "gpt2",
    })

    assert res["status"] == "success"
    assert "backup_envelope" in res
    env = res["backup_envelope"]
    assert "backup_capacity_ratio" in env
    assert "tier_1_active_backups" in env
    assert "combinatorial_knockout_grid" in env


def test_backup_circuits_api_endpoint():
    client = TestClient(app)
    response = client.post("/api/assistant/backup-circuits", json={
        "circuit_nodes": [{"id": "node_L8_H5", "data": {"label": "L8_H5", "attribution": 0.5}}],
        "model_name": "gpt2",
    })
    assert response.status_code == 200
    data = response.json()
    assert "backup_capacity_ratio" in data
    assert "tier_1_active_backups" in data
    assert "combinatorial_knockout_grid" in data
