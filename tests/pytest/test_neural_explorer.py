"""Pytest tests for Neural Explorer Backend (neuron inspector, model tree, circuit explorer)."""

from __future__ import annotations

from science.explorer.circuit_explorer import CircuitExplorer
from science.explorer.model_tree_builder import ModelTreeBuilder
from science.explorer.neuron_inspector import NeuronInspector


# ── Model Tree Builder ────────────────────────────────────────────────────────

def test_model_tree_gpt2_structure():
    builder = ModelTreeBuilder()
    tree = builder.build_tree("gpt2-small")
    assert tree["model_id"] == "gpt2-small"
    assert tree["num_layers"] == 12
    assert tree["num_heads"] == 12
    assert len(tree["layers"]) == 12


def test_model_tree_layer_contents():
    builder = ModelTreeBuilder()
    tree = builder.build_tree("gpt2-small")
    layer = tree["layers"][5]
    assert layer["layer_index"] == 5
    assert len(layer["attention_heads_preview"]) == 12
    assert len(layer["mlp_neurons_preview"]) == 32


def test_model_tree_induction_heads_flagged():
    builder = ModelTreeBuilder()
    tree = builder.build_tree("gpt2-small")
    layer_5 = tree["layers"][5]
    head_1 = next(h for h in layer_5["attention_heads_preview"] if h["head_index"] == 1)
    assert head_1["is_induction_head"] is True
    head_1_label = head_1["label"]
    assert "L5H1" in head_1_label


def test_model_tree_circuits_on_layers():
    builder = ModelTreeBuilder()
    tree = builder.build_tree("gpt2-small")
    layer_7 = tree["layers"][7]
    assert "ioi_circuit" in layer_7["known_circuits"] or "greater_than_circuit" in layer_7["known_circuits"]


def test_list_neurons_in_layer_pagination():
    builder = ModelTreeBuilder()
    page0 = builder.list_neurons_in_layer("gpt2-small", layer=8, page=0, page_size=32)
    assert page0["page"] == 0
    assert len(page0["neurons"]) == 32
    page1 = builder.list_neurons_in_layer("gpt2-small", layer=8, page=1, page_size=32)
    assert page1["page"] == 1
    # No overlap between pages
    page0_ids = {n["neuron_index"] for n in page0["neurons"]}
    page1_ids = {n["neuron_index"] for n in page1["neurons"]}
    assert page0_ids.isdisjoint(page1_ids)


# ── Neuron Inspector ──────────────────────────────────────────────────────────

def test_neuron_inspector_returns_full_detail():
    inspector = NeuronInspector()
    detail = inspector.get_neuron_detail("gpt2-small", layer=9, neuron_index=42)
    assert detail["layer"] == 9
    assert detail["neuron_index"] == 42
    assert detail["model_id"] == "gpt2-small"


def test_neuron_inspector_activation_histogram():
    inspector = NeuronInspector()
    detail = inspector.get_neuron_detail("gpt2-small", layer=8, neuron_index=100)
    hist = detail["activation_histogram"]
    assert len(hist["bins"]) == 11
    assert len(hist["counts"]) == 10
    assert hist["sparsity"] > 0.0


def test_neuron_inspector_top_tokens():
    inspector = NeuronInspector()
    detail = inspector.get_neuron_detail("gpt2-small", layer=6, neuron_index=200)
    assert len(detail["top_activating_tokens"]) >= 5
    assert all("token" in t for t in detail["top_activating_tokens"])
    assert all("activation" in t for t in detail["top_activating_tokens"])


def test_neuron_inspector_cross_model_analogs():
    inspector = NeuronInspector()
    detail = inspector.get_neuron_detail("gpt2-small", layer=5, neuron_index=10)
    analogs = detail["cross_model_analogs"]
    assert len(analogs) >= 1
    assert all("model_id" in a for a in analogs)
    assert all("similarity" in a for a in analogs)


def test_neuron_inspector_circuit_memberships():
    inspector = NeuronInspector()
    # Layer 7 neuron should be in greater_than_circuit
    detail = inspector.get_neuron_detail("gpt2-small", layer=7, neuron_index=5)
    assert "greater_than_circuit" in detail["circuit_memberships"]


def test_neuron_inspector_literature_references():
    inspector = NeuronInspector()
    detail = inspector.get_neuron_detail("gpt2-small", layer=9, neuron_index=9)
    refs = detail["literature_references"]
    assert len(refs) >= 1
    assert all("arxiv_id" in r for r in refs)


def test_neuron_inspector_patch_history():
    inspector = NeuronInspector()
    inspector.record_patch_experiment(
        model_id="gpt2-small", layer=9, neuron_index=42,
        patch_value=3.5, delta_top_logit=1.2,
        top_token_before=" Paris", top_token_after=" France",
    )
    detail = inspector.get_neuron_detail("gpt2-small", layer=9, neuron_index=42)
    assert len(detail["patch_experiment_history"]) == 1
    assert detail["patch_experiment_history"][0]["patch_value"] == 3.5


def test_neuron_inspector_negative_activations():
    inspector = NeuronInspector()
    detail = inspector.get_neuron_detail("gpt2-small", layer=4, neuron_index=50)
    neg = detail["negative_activating_tokens"]
    assert len(neg) > 0
    assert all(t["activation"] < 0 for t in neg)


# ── Circuit Explorer ──────────────────────────────────────────────────────────

def test_circuit_explorer_lists_circuits():
    explorer = CircuitExplorer()
    circuits = explorer.list_circuits()
    circuit_ids = {c["circuit_id"] for c in circuits}
    assert "ioi_circuit" in circuit_ids
    assert "greater_than_circuit" in circuit_ids


def test_circuit_explorer_ioi_detail():
    explorer = CircuitExplorer()
    circuit = explorer.get_circuit("ioi_circuit")
    assert circuit is not None
    assert circuit["circuit_name"] == "Indirect Object Identification Circuit"
    assert circuit["arxiv_id"] == "2211.00593"
    assert len(circuit["nodes"]) >= 5
    assert len(circuit["edges"]) >= 4


def test_circuit_explorer_faithfulness_scores():
    explorer = CircuitExplorer()
    circuit = explorer.get_circuit("ioi_circuit")
    assert circuit["faithfulness"] >= 0.80
    assert circuit["completeness"] >= 0.75


def test_circuit_explorer_component_navigation():
    explorer = CircuitExplorer()
    detail = explorer.get_component_detail("ioi_circuit", "attn_L9N9")
    assert detail["node_id"] == "attn_L9N9"
    assert "navigate_to_neuron" in detail
    assert "/explorer/neuron" in detail["navigate_to_neuron"]


def test_circuit_explorer_unknown_circuit():
    explorer = CircuitExplorer()
    result = explorer.get_circuit("nonexistent_circuit")
    assert result is None


def test_circuit_explorer_greater_than_detail():
    explorer = CircuitExplorer()
    circuit = explorer.get_circuit("greater_than_circuit")
    assert circuit is not None
    node_types = {n["node_type"] for n in circuit["nodes"]}
    assert "mlp_neuron" in node_types
    assert "attention_head" in node_types
