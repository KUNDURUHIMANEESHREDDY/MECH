"""Tests for all Mechanistic Interpretability inspectors."""
from backend.interpretability.inspectors import (
    AttentionInspector,
    LayerInspector,
    NeuronInspector,
    PredictionInspector,
    ResidualInspector,
    TokenInspector,
)


def test_neuron_inspector():
    insp = NeuronInspector()
    res = insp.inspect(layer=8, neuron_index=402, activations=[0.5, 2.5, 1.0])
    assert res["layer"] == 8
    assert res["neuron_index"] == 402
    assert res["neuron_id"] == "L8_N402"
    assert res["max_activation"] == 2.5


def test_attention_inspector():
    insp = AttentionInspector()
    res = insp.inspect(layer=8, head=9, tokens=["A", "B", "C"])
    assert res["layer"] == 8
    assert res["head"] == 9
    assert res["head_id"] == "L8_H9"
    assert res["head_type"] == "Induction Head"
    assert len(res["attention_matrix"]) == 3


def test_residual_inspector():
    insp = ResidualInspector()
    res = insp.inspect(layer=4, prompt="test prompt")
    assert res["layer"] == 4
    assert "norm" in res
    assert "cosine_similarity_with_input" in res


def test_layer_inspector():
    insp = LayerInspector()
    res = insp.inspect(layer=2)
    assert res["layer"] == 2
    assert res["status"] == "computed"
    assert "mlp_activation_norm" in res


def test_token_inspector():
    insp = TokenInspector()
    res = insp.inspect(token_index=3, token_text="France")
    assert res["position"] == 3
    assert res["token"] == "France"
    assert "peak_activation" in res


def test_prediction_inspector():
    insp = PredictionInspector()
    res = insp.inspect(prompt="The capital of France is", top_k=3)
    assert res["top_k"] == 3
    assert len(res["predictions"]) == 3
    assert "entropy" in res
    assert "logit_lens" in res
