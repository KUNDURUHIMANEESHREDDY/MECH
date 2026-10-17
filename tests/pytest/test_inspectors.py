"""Tests for all Mechanistic Interpretability inspectors."""
from interpretability.inspectors import (
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


def test_prediction_inspector_refuses_to_invent_predictions():
    """The inspector must not manufacture a prediction distribution.

    It previously returned five invented predictions (" Paris" at logit 14.8 /
    probability 0.82 and four more), a fixed entropy of 0.65, and a logit lens
    mapping layers 0/6/11 to fixed strings. This test asserted
    `len(res["predictions"]) == 3`, so it passed on fabrications and would have
    passed no matter which tokens were invented.

    The property that matters is the inverse, so it is asserted directly: with
    no model bound, every result-bearing field is empty and the record says so.
    """
    insp = PredictionInspector()
    res = insp.inspect(prompt="The capital of France is", top_k=3)

    # No result may be present, at any top_k.
    assert res["predictions"] == []
    assert res["entropy"] is None
    assert res["logit_lens"] is None

    # ...and the record must declare that, rather than leave the consumer to
    # infer it from empty fields.
    assert res["status"] == "unavailable"
    assert res["provenance"] == "unavailable"
    assert res["measured"] is False
    assert res["validation_eligible"] is False
    assert res["publication_eligible"] is False
    assert res["reason"], "a refusal must carry a reason"

    # Control: the echo fields are honoured, so the empty results cannot be an
    # accident of the arguments never being read.
    assert res["top_k"] == 3
    assert res["prompt"] == "The capital of France is"

    # A larger top_k must not manufacture predictions to fill the gap.
    assert insp.inspect(prompt="x", top_k=5)["predictions"] == []
