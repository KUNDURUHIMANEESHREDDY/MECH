"""Tests for all Mechanistic Interpretability inspectors.

Each inspector must refuse when nothing is bound to it: no model, no forward
pass, no activations. They previously answered with arithmetic over their own
arguments -- ``(token_index * 3) % 12``, ``round(14.2 + layer, 2)``,
``1.0 / (i + 1)`` as an attention row -- and this file asserted the literals
back, so a number-shaped object with nothing measured behind it passed CI.

The property that matters is the inverse, so it is asserted directly: with no
model bound, every result-bearing field is empty, the record says so, and the
arguments are echoed rather than turned into data.
"""
from interpretability.inspectors import (
    AttentionInspector,
    LayerInspector,
    NeuronInspector,
    PredictionInspector,
    ResidualInspector,
    TokenInspector,
)


def assert_refusal(res, data_fields):
    """A refusal carries no data, says so, and is eligible for nothing."""
    assert res["status"] == "unavailable"
    assert res["provenance"] == "unavailable"
    assert res["measured"] is False
    assert res["validation_eligible"] is False
    assert res["publication_eligible"] is False
    assert str(res.get("reason", "")).strip(), "a refusal must carry a reason"
    for field in data_fields:
        assert res[field] in (None, []), f"{field} = {res[field]!r}"


def test_neuron_inspector():
    res = NeuronInspector().inspect(layer=8, neuron_index=402)
    assert res["layer"] == 8
    assert res["neuron_index"] == 402
    assert res["neuron_id"] == "L8_N402"
    assert_refusal(res, ("mean_activation", "max_activation",
                         "firing_frequency", "feature_label"))


def test_neuron_inspector_does_not_summarise_supplied_activations():
    """A caller's list is not a model measurement, so it is not summarised."""
    res = NeuronInspector().inspect(
        layer=8, neuron_index=402, activations=[0.5, 2.5, 1.0])
    assert_refusal(res, ("max_activation", "firing_frequency"))


def test_attention_inspector():
    res = AttentionInspector().inspect(layer=8, head=9, tokens=["A", "B", "C"])
    assert res["layer"] == 8
    assert res["head"] == 9
    assert res["head_id"] == "L8_H9"
    # The tokens are echoed; the matrix and the classification are not invented.
    assert res["tokens"] == ["A", "B", "C"]
    assert_refusal(res, ("head_type", "attention_matrix"))


def test_residual_inspector():
    res = ResidualInspector().inspect(layer=4, prompt="test prompt")
    assert res["layer"] == 4
    assert res["prompt"] == "test prompt"
    assert_refusal(res, ("norm", "cosine_similarity_with_input",
                         "layer_contribution_pct"))


def test_layer_inspector():
    res = LayerInspector().inspect(layer=2)
    assert res["layer"] == 2
    assert_refusal(res, ("mlp_activation_norm", "attention_output_norm",
                         "active_neurons_count"))


def test_token_inspector():
    res = TokenInspector().inspect(token_index=3, token_text="France")
    assert res["position"] == 3
    assert res["token"] == "France"
    assert_refusal(res, ("peak_layer", "peak_activation", "top_firing_neuron"))


def test_token_inspector_does_not_invent_a_token_name():
    res = TokenInspector().inspect(token_index=3)
    assert res["token"] == "" and res["position"] == 3
    assert_refusal(res, ("peak_layer", "peak_activation"))


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
