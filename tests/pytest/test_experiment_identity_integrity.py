"""1.2 -- model identity and experiment isolation.

The acceptance criterion
------------------------
    an experiment either uses one consistent, identifiable configuration or
    fails explicitly

Stated as a property, it has exactly two halves, and this file tests both:

* **consistency** -- a configuration cannot change mid-experiment unnoticed;
* **identifiability** -- every artefact says which configuration produced it.

What exists, measured
---------------------
`ActivationCache` and `CachedActivation` contain **no model identity of any
kind** -- grep for model/fingerprint/variant/revision returns nothing. A
cached activation is keyed by `session_id`, `prompt_id`, `layer`, `head`,
`component` and `shape`.

That last one is the trap. `shape` is the only field that *could* distinguish
two checkpoints, and checkpoints of one family have identical shapes by
construction. So two activations captured under different weights are
indistinguishable in every field, including the ones that look like identity.

    ACT-UNDER-A: session=s1 prompt=p1 layer=5 head=3 shape=(6, 64)
    ACT-UNDER-B: session=s1 prompt=p1 layer=5 head=3 shape=(6, 64)

There is no field in which they differ. Neither does `patch_head_output` know
a capture happened: the adapter exposes `capture_head_outputs` and
`patch_head_output` with no shared transaction, so activations captured under
one checkpoint can be written into another with no possible objection.

The tests below are behavioural throughout -- real adapters, real weights, a
real cache. None of them inspects source to establish a property.

Scope, stated honestly
----------------------
Covered: 1 (partly -- weights only), 2, 3, 4.
Not covered: 5 (forced failures during a model transition) and 6 (every
artefact identifies its configuration). Those need the dispatcher and
serialisation surfaces, which have not been read. The acceptance criterion is
therefore **not** established by this file.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import pytest
import torch

from _weight_guard import skip_reason, weights_available

needs_weights = pytest.mark.skipif(
    not weights_available(), reason=skip_reason()
)

PROMPT_A = "John gave a drink to Mary"
PROMPT_B = "When Alice went to the store she bought"
LAYER, HEAD = 5, 3


@pytest.fixture(scope="module")
def adapter():
    from backend.science.models.gpt2_adapter import GPT2Adapter

    instance = GPT2Adapter(variant="small", mock_mode=False)
    assert instance._model is not None
    return instance


@pytest.fixture
def fresh_cache():
    from backend.runtime.activation_cache import ActivationCache

    return ActivationCache()


@contextmanager
def reweighted(adapter) -> Iterator[None]:
    """Temporarily replace the weights with a different set of the same shapes.

    Not a real checkpoint, and it does not need to be: what matters is that the
    tensors differ while every shape stays identical, because
    shape-compatible but differently-weighted is precisely the case that
    defeats a tensor-level check.

    The weights are **snapshotted and restored**. An earlier version of this
    file called a bare `_reinitialise_weights` and cleared only the
    fingerprint cache, which left the module-scoped adapter permanently
    re-randomised for every later test in the session. Because the seed was
    fixed, the second call was also a no-op -- so the file's own non-vacuity
    control failed when it ran after its siblings, and any test file sharing
    the adapter would have inherited corrupted weights.

    That is P2's "repeated runs contaminate each other", reproduced in the
    file written to test it.
    """
    snapshot = [p.detach().clone() for p in adapter._model.parameters()]
    generator = torch.Generator().manual_seed(1234)
    try:
        with torch.no_grad():
            for param in adapter._model.parameters():
                param.data.copy_(
                    torch.randn(param.shape, generator=generator) * 0.02)
        adapter._fingerprint = None
        yield
    finally:
        with torch.no_grad():
            for param, original in zip(adapter._model.parameters(), snapshot):
                param.data.copy_(original)
        adapter._fingerprint = None


# â”€â”€ 1. what the fingerprint must cover â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@needs_weights
def test_the_fingerprint_covers_weights_but_not_yet_tokenizer_or_config(adapter):
    """The identity a result needs is more than a weight hash.

    A weight fingerprint answers "which weights". It does not answer "which
    tokenizer turned the text into ids" or "which configuration those ids
    were interpreted under". Both change what a number means without changing
    a single parameter.
    """
    fingerprint = adapter.model_fingerprint()

    assert isinstance(fingerprint, str) and fingerprint, (
        "the adapter exposes no weight fingerprint at all"
    )
    assert adapter.spec.model_id, "the spec carries no model id to record"

    # RED for the remainder: there is no tokenizer or configuration identity
    # to compare. Asserted as a contract so the gap is recorded rather than
    # quietly assumed closed.
    assert hasattr(adapter, "tokenizer_fingerprint"), (
        f"the adapter fingerprints weights ({fingerprint[:24]}...) but exposes "
        f"no tokenizer identity. A different tokenizer over the same weights "
        f"produces different ids and a different experiment, with nothing in "
        f"the result distinguishing them."
    )
    assert hasattr(adapter, "config_fingerprint"), (
        "the adapter fingerprints weights but exposes no configuration "
        "identity (layer count, head count, d_head). A config change alters "
        "every index a patch result names."
    )


@needs_weights
def test_two_configurations_with_identical_shapes_are_distinguishable(adapter):
    """The negative control for weight-based identity.

    Same family, same shapes, different weights. If these were
    indistinguishable, no tensor-level check anywhere could catch a
    cross-checkpoint mix.
    """
    before = adapter.model_fingerprint()
    with reweighted(adapter):
        after = adapter.model_fingerprint()

        assert before != after, (
            "replacing every weight left the fingerprint unchanged"
        )
        assert adapter.spec.num_layers == 12 and adapter.spec.d_model == 768, (
            "the control requires the shapes to be unchanged, so the two "
            "configurations are shape-compatible and differ only in values"
        )


# â”€â”€ 2. cached activations must not survive a model switch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@needs_weights
def test_a_cached_activation_records_which_model_produced_it(fresh_cache,
                                                             adapter):
    """RED, deliberately -- the central finding of this file.

    `CachedActivation` has no model field, so an entry cannot say which
    weights it came from. The `shape` field is the only candidate and it is
    identical across checkpoints by construction.
    """
    from backend.runtime.activation_cache import CachedActivation

    entry = CachedActivation(
        session_id="s1", prompt_id="p1", layer=LAYER, head=HEAD,
        shape=(6, 64), tensor=torch.zeros(6, 64),
    )

    fingerprint = adapter.model_fingerprint()
    assert entry.metadata.get("model_fingerprint") == fingerprint, (
        f"a cached activation carries no model identity "
        f"(metadata={entry.metadata!r}). Two activations captured under "
        f"different weights of the same shape are indistinguishable in every "
        f"field the cache records -- session, prompt, layer, head, component, "
        f"shape -- so a stale entry cannot be detected, let alone refused."
    )


@needs_weights
def test_an_entry_cached_under_one_model_is_not_returned_for_another(
    fresh_cache, adapter):
    """RED, deliberately.

    The reuse this forbids is the one `ActivationCache` is designed to make
    easy: store under a session id, look up later by that id. Without model
    identity in the key or the entry, the lookup succeeds and returns data
    measured under different weights.
    """
    from backend.runtime.activation_cache import CachedActivation

    cached = CachedActivation(
        session_id="shared", prompt_id=PROMPT_A, layer=LAYER, head=HEAD,
        shape=(6, 64), tensor=torch.zeros(6, 64), component="attn",
    )
    fresh_cache.put(cached)

    # The model changes. The session id does not, and nothing forces it to.
    with reweighted(adapter):
        stale = fresh_cache.get(cached.activation_id)

        assert stale is None, (
            "an activation captured under the previous weights was returned "
            "unchanged after the model was replaced. Shapes are identical, so "
            "every field of the entry still matches; only a fingerprint "
            "distinguishes them."
        )


# â”€â”€ 3. capture and patch must agree on the model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@needs_weights
def test_capturing_under_one_model_and_patching_under_another_is_refused(
    adapter,
):
    """RED, deliberately -- and the most consequential gap here.

    `capture_head_outputs` and `patch_head_output` share no transaction. The
    vector captured from head L5H3 under one checkpoint is a
    `List[float]` of the right length; writing it into another checkpoint is a
    well-typed operation that no layer of the adapter can object to.
    """
    captured = adapter.capture_head_outputs(PROMPT_A)[LAYER][HEAD]
    vector = list(captured)

    with reweighted(adapter):
        with pytest.raises((ValueError, RuntimeError, LookupError)) as excinfo:
            result = adapter.patch_head_output(
                PROMPT_A, LAYER, HEAD, patch_vector=vector, position=-1)
            raise AssertionError(
                f"a vector captured under the previous weights was accepted "
                f"and written into the current ones (delta={result.delta}). "
                f"The two checkpoints have identical shapes, so every check "
                f"the adapter performs was satisfied. Found no objection."
            )
        raise AssertionError(f"unreachable: {excinfo!r}")


# â”€â”€ 4. interleaved experiments must not contaminate each other â”€â”€â”€â”€

@needs_weights
def test_interleaved_experiments_with_different_prompts_stay_separate(adapter):
    """The one property here that should already hold, pinned so it does not.

    Two experiments with different prompts are run back to back and
    interleaved, and each must report its own prompt's result. A hook that
    leaks between runs would return the first experiment's numbers for the
    second.
    """
    a_first = adapter.patch_head_output(
        PROMPT_A, LAYER, HEAD, patch_vector=[0.0] * 64, position=-1)
    b_first = adapter.patch_head_output(
        PROMPT_B, LAYER, HEAD, patch_vector=[0.0] * 64, position=-1)
    a_second = adapter.patch_head_output(
        PROMPT_A, LAYER, HEAD, patch_vector=[0.0] * 64, position=-1)

    assert a_first.delta == a_second.delta, (
        f"the same experiment gave {a_first.delta} then {a_second.delta}. A "
        f"result that depends on what ran before it is not a measurement."
    )
    assert a_first.delta != b_first.delta or a_first.original_logit != \
        b_first.original_logit, (
        "two different prompts produced an identical result; the "
        "experiments are contaminating each other"
    )


# â”€â”€ the assertions above must be able to fail â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_the_reweighting_control_actually_changes_the_model(adapter):
    """Non-vacuity for every RED above.

    If `_reinitialise_weights` did not actually change anything, the tests
    asserting a refusal would be failing for an unrelated reason and the
    absence of a refusal would prove nothing.
    """
    before = adapter.model_fingerprint()
    with reweighted(adapter):
        assert adapter.model_fingerprint() != before, (
            "the reweighting control did not change the model, so no test in "
            "this file that depends on a model switch is measuring anything"
        )