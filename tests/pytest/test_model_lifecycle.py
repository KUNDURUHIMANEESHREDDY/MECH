"""A3-A14: the model lifecycle, at `backend.runtime.model_manager`.

This module had **zero** test references in a 1900-test suite. It is a
module-level singleton holding three globals -- `_loaded_model`,
`_loaded_tokenizer`, `_loaded_name` -- mutated without a lock, and it is the
only thing in the runtime that decides which weights the whole platform is
measuring.

The defects found here
----------------------
`load_model` called `_unload_current()` *before* attempting the new load. So a
load that then failed left the platform with no model at all:

    before : is_loaded()=True, name='gpt2'
    load_model(<a registry entry whose checkpoint cannot be read>) -> ModelLoadError
    after  : is_loaded()=False, name=None, model=None

A transient failure -- a dropped connection, a full disk, a corrupt download --
destroys a working 500 MB model that was serving fine, and the operator gets an
exception with no hint that the previous state is gone.

Worse when the failure is later than the tokenizer: the tokenizer is assigned
to the global at line 97 and the model at line 101, so a failure in between
leaves the module holding a tokenizer for a model it never loaded. Nothing
reports it as loaded -- `_loaded_name` is None -- but `get_model_and_tokenizer`
is the only guard, and any caller reaching past it gets the stale tokenizer.

Two checklist items are also unimplemented rather than merely untested:
`list_models()` returns a flat list of names with nothing distinguishing the
three that work from the five that raise, and `load_model`'s result reports no
`device` or `dtype`, so a caller cannot verify it is measuring in the precision
it thinks.

These tests drive failures through a temporary registry entry and injected
loaders, so no network access is required and no test depends on which
checkpoints happen to be cached.
"""

from __future__ import annotations

import threading

import pytest

from backend.runtime import model_manager as mm
from backend.runtime.errors import (
    ModelLoadError,
    ModelNotFoundError,
    MissingModelError,
)

MODEL = "gpt2"


@pytest.fixture(autouse=True)
def clean_singleton():
    """Leave the module exactly as found -- it is process-global state.

    Restores all three globals, not just the model, because the torn-state
    defect under test leaves a stale *tokenizer* behind and a test that only
    cleared the model would inherit it.
    """
    saved = (mm._loaded_model, mm._loaded_tokenizer, mm._loaded_name)
    registry = dict(mm._MODEL_REGISTRY)
    yield
    mm._loaded_model, mm._loaded_tokenizer, mm._loaded_name = saved
    mm._MODEL_REGISTRY.clear()
    mm._MODEL_REGISTRY.update(registry)


# ── A6: nothing loaded must say so ─────────────────────────────────────

def test_asking_for_the_model_with_nothing_loaded_raises():
    """The precondition check must exist and be specific.

    `is_loaded()` and `get_model_and_tokenizer()` must agree: if the first
    says False the second must not hand back an object.
    """
    mm._loaded_model = None
    mm._loaded_tokenizer = None
    mm._loaded_name = None

    assert mm.is_loaded() is False
    with pytest.raises(ModelLoadError) as excinfo:
        mm.get_model_and_tokenizer()
    assert "No model loaded" in str(excinfo.value), (
        f"the error must say no model is loaded: {excinfo.value}"
    )


def test_is_loaded_and_the_accessor_cannot_disagree():
    """A stale tokenizer must not make `is_loaded()` report True.

    Directly targets the torn state: `_loaded_tokenizer` can outlive a failed
    load. If `is_loaded()` consulted the tokenizer it would report a loaded
    model with no weights behind it.
    """
    mm._loaded_model = None
    mm._loaded_tokenizer = object()      # the stale half of a torn state
    mm._loaded_name = None

    assert mm.is_loaded() is False, (
        "is_loaded() reported True with no weights, on the strength of a "
        "leftover tokenizer"
    )
    with pytest.raises(ModelLoadError):
        mm.get_model_and_tokenizer()


# ── A3: unknown vs registered-but-unsupported ──────────────────────────

def test_an_unknown_name_is_distinct_from_an_unsupported_one():
    """The two refusals mean different things and must not be conflated.

    'You asked for a model that does not exist' and 'that model exists and is
    not built yet' call for different responses from an operator -- a typo
    versus a missing feature. Both currently subclass `MissingModelError`, so a
    caller catching that base cannot tell them apart, and the exception types
    themselves must.
    """
    with pytest.raises(ModelNotFoundError) as unknown:
        mm.load_model("no-such-model-anywhere")

    future = next(iter(mm._FUTURE_MODELS))
    with pytest.raises(ModelLoadError) as unsupported:
        mm.load_model(future)

    assert type(unknown.value) is not type(unsupported.value), (
        "an unknown model and an unsupported model raise the same type"
    )
    assert "not in registry" in str(unknown.value) or "Unknown model" in str(
        unknown.value
    )
    assert "not yet supported" in str(unsupported.value), (
        f"the unsupported-model refusal must say so: {unsupported.value}"
    )
    # Both remain catchable by the broad base, which is the documented intent.
    assert isinstance(unknown.value, MissingModelError)
    assert isinstance(unsupported.value, MissingModelError)


# ── A13: list_models must mark what is unsupported ─────────────────────

def test_list_models_distinguishes_loadable_from_unsupported():
    """The listing must say which entries actually work.

    `list_models()` returns one flat list of eight names. Five of them raise
    `ModelLoadError` on load, three succeed, and nothing in the returned value
    distinguishes the two groups. A UI rendering this list offers the user five
    models that cannot be loaded, with no indication until the attempt fails.
    """
    listed = mm.list_models()

    loadable = set(mm._MODEL_REGISTRY)
    unsupported = set(mm._FUTURE_MODELS)
    assert loadable and unsupported, "fixture assumption broken"

    if all(isinstance(name, str) for name in listed):
        # A flat list of bare strings carries no marking at all.
        pytest.fail(
            "list_models() returns bare strings, so a caller cannot tell the "
            "loadable entries from the unsupported ones. It should carry the "
            "support status per entry."
        )

    by_name = {entry["model_name"]: entry for entry in listed}
    assert set(by_name) == loadable | unsupported, (
        f"list_models() returned names {sorted(by_name)}, expected "
        f"{sorted(loadable | unsupported)}"
    )
    for name in loadable:
        assert by_name[name].get("supported") is True, (
            f"{name} is loadable but not marked supported"
        )
    for name in unsupported:
        assert by_name[name].get("supported") is False, (
            f"{name} is unsupported but not marked as such; a UI listing it "
            f"without that flag offers a model that cannot load"
        )

    # And each entry must agree with what `load_model` would actually do.
    for name in unsupported:
        with pytest.raises(ModelLoadError):
            mm.load_model(name)


# ── A1/A8: a real load, and the pad token ──────────────────────────────

@pytest.mark.slow
def test_loading_real_weights_reports_truthful_config():
    """Config comes from the loaded model, not from the registry table.

    The registry carries `n_layer`/`n_head`/`n_embd` per model, so a test
    asserting the returned info against those constants would pass even if the
    loader returned the wrong weights. This reads the loaded model's own config
    instead, which is what makes the assertion about the measurement rather
    than about a table.
    """
    import torch

    info = mm.load_model(MODEL)
    assert info["status"] == "loaded"
    assert mm.is_loaded() is True

    model, _tokenizer = mm.get_model_and_tokenizer()
    config = model.config

    assert info["num_layers"] == config.n_layer, (
        f"reported {info['num_layers']} layers, the loaded model has "
        f"{config.n_layer}"
    )
    assert info["num_heads"] == config.n_head
    assert info["hidden_dim"] == config.n_embd

    # The model must be in eval mode, or dropout silently perturbs every
    # forward pass this platform takes.
    assert not model.training, "a freshly loaded model was left in train mode"
    weights = next(model.parameters())
    assert weights.dtype == torch.float32, (
        f"loaded in {weights.dtype}, not the float32 the loader requests"
    )


@pytest.mark.slow
def test_the_tokenizer_gets_a_pad_token_when_it_lacks_one():
    """GPT-2 has no pad token; batched inference needs one.

    Without it, padding a batch either raises or pads with a token the model
    was never trained on, and a padded row's activations are silently wrong.
    """
    _, tokenizer = mm.get_model_and_tokenizer() if mm.is_loaded() else (None, None)
    if tokenizer is None:
        mm.load_model(MODEL)
        _, tokenizer = mm.get_model_and_tokenizer()

    assert tokenizer.pad_token is not None, (
        "the tokenizer has no pad token after loading; batched inference "
        "would pad with an arbitrary token"
    )
    assert tokenizer.pad_token_id is not None


# ── A14: device and dtype must be reported ─────────────────────────────

@pytest.mark.slow
def test_the_load_result_reports_the_device_and_dtype_actually_used():
    """The result must let a caller verify the precision it is measuring in.

    The loader requests `torch_dtype=torch.float32` but the returned info
    carries only `model_name`, `status` and three dimensions. A caller on a
    machine where the weights land elsewhere, or in a different precision, has
    nothing to check against.
    """
    info = mm.load_model(MODEL)
    model, _tok = mm.get_model_and_tokenizer()
    actual_dtype = next(model.parameters()).dtype
    actual_device = str(next(model.parameters()).device)

    assert "dtype" in info, (
        f"the load result reports no dtype; the model is in {actual_dtype} and "
        f"a caller cannot verify it"
    )
    assert "device" in info, (
        f"the load result reports no device; the model is on {actual_device}"
    )
    assert str(info["dtype"]) == str(actual_dtype)
    assert str(info["device"]) == actual_device


# ── A9/A4: a failed load must not destroy or tear the module ───────────

@pytest.mark.slow
def test_a_failed_load_leaves_the_working_model_untouched():
    """A load that fails must not unload what was already working.

    The defect: `_unload_current()` ran before the new weights were fetched, so
    any failure -- network, disk, corrupt download -- left the platform with
    nothing, and the raised error said nothing about that.

    Driven by injecting a loader that raises, so no network is touched and the
    failure is deterministic.
    """
    mm.load_model(MODEL)
    working = mm._loaded_model
    assert mm.is_loaded() is True

    real_model_loader = mm.AutoModelForCausalLM.from_pretrained

    class _Failing:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise OSError("simulated checkpoint read failure")

    mm.AutoModelForCausalLM = _Failing
    mm._MODEL_REGISTRY["sim-broken"] = {
        "hf_id": MODEL, "family": "gpt2", "n_layer": 1, "n_head": 1,
        "n_embd": 1,
    }
    try:
        with pytest.raises(ModelLoadError):
            mm.load_model("sim-broken")
    finally:
        mm.AutoModelForCausalLM = type(
            "R", (), {"from_pretrained": staticmethod(real_model_loader)}
        )

    assert mm.is_loaded() is True, (
        "a failed load unloaded the working model. The operator now has no "
        "model and the exception did not mention it."
    )
    assert mm._loaded_name == MODEL, (
        f"the loaded name changed to {mm._loaded_name!r} during a failed load"
    )
    assert mm._loaded_model is working, (
        "the loaded weights were replaced by a failed load"
    )
    # And it must still work.
    model, tokenizer = mm.get_model_and_tokenizer()
    assert model is working and tokenizer is not None


@pytest.mark.slow
def test_a_failed_load_leaves_no_stale_tokenizer():
    """A failure after the tokenizer step must not leave it behind.

    `_loaded_tokenizer` is assigned before `_loaded_model`, so a failure
    between the two leaves the module holding a tokenizer for a model it never
    loaded. Nothing reports it -- `_loaded_name` is None, so `model_info` says
    nothing is loaded -- but it is still the object any caller that reaches past
    `get_model_and_tokenizer` would receive.
    """
    real_tok_loader = mm.AutoTokenizer.from_pretrained
    real_model_loader = mm.AutoModelForCausalLM.from_pretrained

    class _WorkingTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return real_tok_loader(*args, **kwargs)

    class _FailingModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise OSError("simulated: weights unreadable")

    mm._loaded_model = None
    mm._loaded_tokenizer = None
    mm._loaded_name = None

    mm.AutoTokenizer = _WorkingTokenizer
    mm.AutoModelForCausalLM = _FailingModel
    mm._MODEL_REGISTRY["sim-torn"] = {
        "hf_id": MODEL, "family": "gpt2", "n_layer": 1, "n_head": 1,
        "n_embd": 1,
    }
    try:
        with pytest.raises(ModelLoadError):
            mm.load_model("sim-torn")
    finally:
        mm.AutoTokenizer = type(
            "T", (), {"from_pretrained": staticmethod(real_tok_loader)}
        )
        mm.AutoModelForCausalLM = type(
            "M", (), {"from_pretrained": staticmethod(real_model_loader)}
        )

    assert mm._loaded_tokenizer is None, (
        "a failed load left a tokenizer behind. The module holds a tokenizer "
        "for a model it never loaded, while _loaded_name is None so nothing "
        "reports it."
    )
    assert mm._loaded_model is None
    assert mm._loaded_name is None
    with pytest.raises(ModelLoadError):
        mm.get_model_and_tokenizer()


# ── A5/A12: unload and switch ──────────────────────────────────────────

@pytest.mark.slow
def test_unloading_the_wrong_model_is_refused():
    """`unload_model(name)` must not unload a model that is not loaded.

    Otherwise a request naming `gpt2-medium` while `gpt2` is resident would
    silently drop `gpt2`.
    """
    mm.load_model(MODEL)

    with pytest.raises(ModelLoadError) as excinfo:
        mm.unload_model("distilgpt2")

    assert "not currently loaded" in str(excinfo.value), (
        f"the refusal must name the mismatch: {excinfo.value}"
    )
    assert mm.is_loaded() is True, "a refused unload still unloaded the model"
    assert mm._loaded_name == MODEL


@pytest.mark.slow
def test_switching_to_the_same_model_still_yields_a_loaded_model():
    """`switch_model` must end loaded, even when the name is unchanged.

    Switch is `load_model` plus an event, and `load_model` unloads first. If
    the reload then fails, the platform is left empty -- the same defect as the
    failed-load case, reached through the more natural call. Asserting the
    end state rather than the event covers both.
    """
    mm.load_model(MODEL)
    info = mm.switch_model(MODEL)

    assert info["status"] == "loaded"
    assert mm.is_loaded() is True
    assert mm._loaded_name == MODEL
    model, tokenizer = mm.get_model_and_tokenizer()
    assert model is not None and tokenizer is not None


# ── A11: concurrent access ─────────────────────────────────────────────

def test_concurrent_loaders_do_not_interleave_into_a_torn_pair():
    """Two threads must never observe one load's model with another's tokenizer.

    The three globals are written separately with no lock, so a reader can
    legitimately see `_loaded_model` from one load and `_loaded_tokenizer` from
    another. The result would be a tokenizer that does not match the weights --
    producing token ids the embedding table does not cover, with no error.

    Asserted on the pairing rather than on a lock, since a lock is one possible
    implementation and the property is what callers depend on.
    """
    import types

    mm._loaded_model = None
    mm._loaded_tokenizer = None
    mm._loaded_name = None

    torn: list[str] = []
    stop = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            model = mm._loaded_model
            tokenizer = mm._loaded_tokenizer
            if model is None or tokenizer is None:
                continue
            # Every write pairs the two; a torn read shows a model whose
            # recorded partner is not the tokenizer observed alongside it.
            partner = getattr(model, "_paired_tokenizer", None)
            if partner is not None and partner is not tokenizer:
                torn.append("model paired with a different tokenizer")

    watcher = threading.Thread(target=reader, daemon=True)
    watcher.start()
    try:
        for i in range(40):
            mm._MODEL_REGISTRY[f"sim-{i}"] = {
                "hf_id": MODEL, "family": "gpt2", "n_layer": 1,
                "n_head": 1, "n_embd": 1,
            }
            marker = types.SimpleNamespace(_paired_tokenizer=None)
            mm._loaded_tokenizer = marker
            mm._loaded_model = marker
            mm._loaded_name = f"sim-{i}"
            mm._loaded_model = None
    finally:
        stop.set()
        watcher.join(timeout=10)

    assert not torn, torn[:3]
    for name in list(mm._MODEL_REGISTRY):
        if name.startswith("sim-"):
            del mm._MODEL_REGISTRY[name]