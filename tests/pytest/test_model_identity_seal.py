"""Model identity: a result must not be attributable to the wrong weights.

The P0, stated precisely
-----------------------
A run could capture activations under checkpoint A and patch under checkpoint
B. Both are the same family, the same layer count, the same head count, the
same `d_head` -- so **every shape-compatible and tensor-level check passes**.
`delta` is a float. A float from model A and a float from model B are
indistinguishable to everything currently in the codebase.

The defect, measured
--------------------
Before this file, `PatchResult` carried:

    original_logit, patched_logit, delta, top_token_before, top_token_after,
    layer, neuron_index, patch_value

No model identity of any kind. `grep` for `model_id|model_identity|revision|
device|dtype` across `backend/` returns nothing on the adapter, and
`evidence_graph` -- which *signs* run records -- carries none either. A signed
record proves integrity, not provenance: it says the bytes were not altered,
not which weights produced them.

The note in `dataset_manager.compute_fingerprint` ("so a manifest that ships
either is carrying a placeholder") is the same vocabulary this project
already invented -- for datasets, and never extended to checkpoints.

What is red here, and why it stays red
--------------------------------------
`test_the_seal_cannot_be_forgotten` fails. `PatchResult(...)` sites across three
adapter classes still pass no fingerprint, so `model_fingerprint` is `None` and
any two results can be compared without objection. Making that impossible
requires populating every site, and for the two non-GPT-2 adapters it first
requires giving *them* a fingerprint method. That is the correct next step and
it is not a mechanical edit; it is left red rather than half-applied, because a
required field with unpopulated constructors raises at runtime instead of being
absent -- trading a silent gap for a crash is not an improvement.

How many sites that is, is deliberately not written down here. This paragraph
said "nine"; the census reported eight when it was last measured. A count in
prose cannot notice when the code moves, and this one did not.
`test_the_seal_cannot_be_forgotten` counts the offenders from the source on
every run -- read the number there, and nowhere else.

The controls below are the ones that are satisfiable now, and each is checked
against a deliberately-broken twin so it cannot pass vacuously.
"""

from __future__ import annotations

import pytest

from _weight_guard import skip_reason, weights_available

needs_weights = pytest.mark.skipif(
    not weights_available(), reason=skip_reason()
)

PROMPT = "John gave a drink to Mary"
LAYER, HEAD = 5, 3


@pytest.fixture(scope="module")
def adapter():
    from backend.science.models.gpt2_adapter import GPT2Adapter

    instance = GPT2Adapter(variant="small", mock_mode=False)
    assert instance._model is not None
    return instance


# ── the fingerprint is evidence, not a label ────────────────────────

@needs_weights
def test_the_fingerprint_is_derived_from_weights_not_the_label(adapter):
    """It must not be `model_id` wearing a hash costume.

    Two adapters built from the same checkpoint have the same `spec.model_id`
    and must therefore agree. If the fingerprint were the label, this test
    would still pass -- so the negative control below is what gives it teeth.
    """
    from backend.science.models.gpt2_adapter import GPT2Adapter

    twin = GPT2Adapter(variant="small", mock_mode=False)
    assert twin.spec.model_id == adapter.spec.model_id, (
        "the control needs two adapters with the same label; if this fails the "
        "test below is proving nothing about weight-derived identity"
    )
    assert adapter.model_fingerprint() == twin.model_fingerprint(), (
        "two independently loaded copies of the same checkpoint disagree on "
        "their fingerprint, so the fingerprint is not a function of the weights"
    )


@needs_weights
def test_a_mutated_weight_changes_the_fingerprint(adapter):
    """The negative control: the hash must track the weights, not the object.

    Without this, a `model_fingerprint()` returning a constant would pass the
    test above for the wrong reason.
    """
    baseline = adapter.model_fingerprint()
    param = next(iter(adapter._model.parameters()))
    original = param.detach().clone()

    try:
        adapter._fingerprint = None
        param.data.add_(1e-4)
        mutated = adapter.model_fingerprint()

        assert mutated is not None
        assert mutated != baseline, (
            "a changed weight produced an identical fingerprint; the hash is "
            "not reading the weights"
        )
    finally:
        param.data.copy_(original)
        adapter._fingerprint = None

    # And the restore is observable, so the control cannot pass by leaving
    # the model permanently perturbed for every later test in this file.
    assert adapter.model_fingerprint() == baseline, (
        "restoring the weight did not restore the fingerprint; this test "
        "leaves the shared module-scoped adapter in a different state"
    )


@needs_weights
def test_the_cache_is_invalidated_so_a_reload_is_detectable(adapter):
    """Fingerprint, restore, re-fingerprint: back to the original value.

    This is what makes "load a new model, keep the old cache" detectable at
    all. If the cached value were returned regardless of the weights, this
    would not return to `baseline`.
    """
    baseline = adapter.model_fingerprint()
    adapter._fingerprint = None
    assert adapter.model_fingerprint() == baseline, (
        "recomputing the fingerprint after invalidation changed the answer; "
        "the hash is not stable for fixed weights"
    )


@needs_weights
def test_mock_mode_returns_none_rather_than_a_fiction(adapter):
    """No weights means no fingerprint -- not a plausible-looking string."""
    from backend.science.models.gpt2_adapter import GPT2Adapter

    mock = GPT2Adapter(variant="small", mock_mode=True)
    if mock._model is None:
        assert mock.model_fingerprint() is None, (
            "an adapter with no model returned a fingerprint"
        )
    else:
        # A mock that holds tensors can still be fingerprinted honestly.
        assert isinstance(mock.model_fingerprint(), str)


# ── results must carry identity, or refuse to compare ───────────────

@needs_weights
def test_patch_results_record_which_weights_they_were_measured_against(adapter):
    """RED, deliberately.

    The real defect. Nothing in the result says which checkpoint produced it,
    so a value captured under A and read under B is indistinguishable from one
    measured under A.
    """
    result = adapter.patch_head_output(
        PROMPT, LAYER, HEAD, patch_vector=[0.0] * (adapter.spec.d_model
                                                  // adapter.spec.num_heads))

    assert result.model_fingerprint == adapter.model_fingerprint(), (
        "PatchResult carries no model identity, so it cannot be distinguished "
        "from a result measured under different weights. "
        "test_the_seal_cannot_be_forgotten lists the construction sites that "
        "still omit a fingerprint -- read the census there rather than a count "
        "here."
    )


@needs_weights
def test_two_results_from_different_weights_refuse_to_be_compared(adapter):
    """RED, deliberately -- and the property that makes the seal worth having.

    Two checkpoints of one family agree on every shape. If they are ever
    compared, this must refuse rather than return a plausible number.
    """
    from backend.science.models.adapter_base import PatchResult

    real = adapter.patch_head_output(
        PROMPT, LAYER, HEAD, patch_vector=[0.0] * (adapter.spec.d_model
                                                  // adapter.spec.num_heads))

    # A result from another checkpoint: same shapes, same field names, and --
    # today -- no identity at all.
    foreign = PatchResult(
        original_logit=1.0, patched_logit=2.0, delta=1.0,
        top_token_before=" a", top_token_after=" b",
        layer=LAYER, neuron_index=HEAD, patch_value=0.0,
    )

    verdict = real.same_model_as(foreign)
    assert verdict is False, (
        f"a result measured against {adapter.model_fingerprint()} reported "
        f"{verdict!r} when compared against a result from a different "
        f"checkpoint. Shapes are identical across checkpoints, so only a "
        f"fingerprint can answer this -- and it answered None."
    )


def test_comparison_is_unanswerable_rather_than_optimistic_when_unsealed():
    """`None`, never `True`.

    Collapsing the unrecorded case into "compatible" is how a gap becomes a
    result, and it would be indistinguishable from a real comparison.
    """
    from backend.science.models.adapter_base import PatchResult

    unsealed = PatchResult(
        original_logit=0.0, patched_logit=0.0, delta=0.0,
        top_token_before="", top_token_after="",
        layer=0, neuron_index=0, patch_value=0.0,
    )
    other = PatchResult(
        original_logit=0.0, patched_logit=0.0, delta=0.0,
        top_token_before="", top_token_after="",
        layer=0, neuron_index=0, patch_value=0.0,
        model_fingerprint="sha256:abc",
    )

    assert unsealed.same_model_as(other) is None
    assert other.same_model_as(unsealed) is None
    assert unsealed.same_model_as(unsealed) is None


def test_shape_agreement_is_not_identity():
    """Why `same_model_as` must not consult `layer` and `neuron_index`.

    Two results agreeing on every structural field is exactly what a
    cross-checkpoint mix looks like. If that were treated as identity, the
    guard would pass precisely the case it exists to catch.
    """
    from backend.science.models.adapter_base import PatchResult

    a = PatchResult(original_logit=1.0, patched_logit=2.0, delta=1.0,
                    top_token_before=" a", top_token_after=" b",
                    layer=5, neuron_index=3, patch_value=0.0,
                    model_fingerprint="sha256:aaa")
    b = PatchResult(original_logit=9.0, patched_logit=8.0, delta=-1.0,
                    top_token_before=" c", top_token_after=" d",
                    layer=5, neuron_index=3, patch_value=0.0,
                    model_fingerprint="sha256:bbb")

    assert (a.layer, a.neuron_index) == (b.layer, b.neuron_index)
    assert a.same_model_as(b) is False, (
        "results from different weights were judged compatible because their "
        "structural fields agreed"
    )


def test_the_seal_cannot_be_forgotten():
    """RED, deliberately: the census of unpopulated construction sites.

    A guard that can be bypassed by constructing a result the old way is not
    a guard. This fails until every `PatchResult(...)` passes a fingerprint.
    """
    import ast
    from pathlib import Path

    import backend.science.models as models_pkg

    offenders: list = []
    for path in Path(models_pkg.__file__).parent.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Name) and func.id == "PatchResult"):
                continue
            passed = {kw.arg for kw in node.keywords}
            if "model_fingerprint" not in passed:
                offenders.append(f"{path.name}:{node.lineno}")

    assert not offenders, (
        f"{len(offenders)} PatchResult site(s) construct without identity: "
        f"{sorted(offenders)}. Any result from one of these is "
        f"indistinguishable from one measured under different weights."
    )