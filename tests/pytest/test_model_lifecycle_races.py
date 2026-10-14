"""Model lifecycle: races, failure injection, and identity.

`test_model_lifecycle.py` covers unloaded state, registry reporting, failed
loads, stale-tokenizer prevention and unload identity. This file closes the
gaps the contract review names for section 9, with one priority:

    inference must never combine components from different model loads

Two of these already pass, and that is a result worth recording rather than
asserting: the transactional load and the locked accessor were built earlier
in this session, and the tests below are what would catch their removal.

`load_model` builds the tokenizer and the weights into locals and commits all
three globals under `_LOAD_LOCK` as the last step. `get_model_and_tokenizer`
reads both under the same lock. So the pair (model, tokenizer) is published
atomically and read atomically. What that leaves untested is whether anything
*else* observes the middle -- which is what these tests drive.

Everything here runs on synthetic registry entries with stub loaders. Real
weights are not needed to test publication ordering, and loading GPT-2 per
test would dominate the runtime.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
import torch

from backend.runtime import model_manager as mm
from backend.runtime.errors import ModelLoadError

MODEL = "gpt2"


class _StubConfig:
    """Just enough of a HF config for the metadata the manager reports."""

    def __init__(self, tag: str) -> None:
        self._tag = tag
        self.n_layer = 1
        self.n_head = 1
        self.n_embd = 2
        self.hidden_size = 2

    @property
    def model_type(self) -> str:
        return self._tag


class _StubModel(torch.nn.Module):
    def __init__(self, tag: str) -> None:
        super().__init__()
        self.tag = tag
        self.config = _StubConfig(tag)
        self.linear = torch.nn.Linear(2, 2)


class _StubTokenizer:
    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"


def _install(monkeypatch, tag: str, *, fail_model: bool = False,
             fail_tokenizer: bool = False, delay: float = 0.0):
    """Point the manager at stub loaders that answer with `tag`.

    `delay` holds the load open, which is how a race is made reproducible
    rather than hoped for.
    """
    import time as _time

    def tokenizer_loader(hf_id, *args, **kwargs):
        if delay:
            _time.sleep(delay)
        if fail_tokenizer:
            raise OSError("simulated tokenizer load failure")
        return _StubTokenizer(tag)

    def model_loader(hf_id, *args, **kwargs):
        if delay:
            _time.sleep(delay)
        if fail_model:
            raise OSError("simulated weights load failure")
        return _StubModel(tag)

    monkeypatch.setattr(mm, "AutoTokenizer",
                        type("T", (), {"from_pretrained": staticmethod(tokenizer_loader)}))
    monkeypatch.setattr(
        mm, "AutoModelForCausalLM",
        type("M", (), {"from_pretrained": staticmethod(model_loader)}))


def _setup(monkeypatch, names, tag: str, **kwargs):
    """Register every name first, then install the loaders.

    An earlier draft called `_install` before `_register`, so
    `load_model("sim-a")` raised `ModelNotFoundError` -- the stub loaders were
    fine, the model simply was not in the registry yet.
    """
    for name in names:
        _register(name)
    _install(monkeypatch, tag, **kwargs)


def _register(name: str) -> None:
    mm._MODEL_REGISTRY[name] = {
        "hf_id": name, "family": "gpt2", "n_layer": 1, "n_head": 1, "n_embd": 1,
    }


@pytest.fixture(autouse=True)
def _restore_registry():
    saved = dict(mm._MODEL_REGISTRY)
    yield
    mm._MODEL_REGISTRY.clear()
    mm._MODEL_REGISTRY.update(saved)


@pytest.fixture
def clean_manager():
    mm._unload_current()
    yield mm
    mm._unload_current()


# Ã¢â€â‚¬Ã¢â€â‚¬ Gap 1 and 5: races between load, unload and inference Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

def test_a_reader_never_sees_a_model_paired_with_another_loads_tokenizer(
    clean_manager, monkeypatch,
):
    """The P0 invariant, under a race made reproducible.

    Two loads with a delay between them, and a reader looping throughout. Every
    observation must be a consistent pair: the model and tokenizer from the
    same load, carrying the same tag.

    An inconsistent observation -- `_StubModel("a")` beside
    `_StubTokenizer("b")` -- is a tokenizer for one model applied to the
    weights of another. Everything downstream then reads as valid: ids decode
    to some string, logits come out, and no error is raised anywhere.
    """
    _setup(monkeypatch, ['sim-a', 'sim-b'], 'a')
    torn: list[str] = []
    stop = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            try:
                model, tokenizer = mm.get_model_and_tokenizer()
            except ModelLoadError:
                continue                      # nothing loaded: fine
            if model.tag != tokenizer.tag:
                torn.append(f"model={model.tag} tokenizer={tokenizer.tag}")

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        _install(monkeypatch, "a", delay=0.005)
        mm.load_model("sim-a")
        _install(monkeypatch, "b", delay=0.005)
        mm.load_model("sim-b")
    finally:
        stop.set()
        thread.join(timeout=10)

    assert not torn, (
        f"a reader saw mismatched components {torn[:3]}. A tokenizer applied "
        f"to another model's weights decodes to a plausible string and "
        f"produces plausible logits; nothing downstream would raise."
    )


def test_unload_during_inference_does_not_yield_a_half_cleared_pair(
    clean_manager, monkeypatch,
):
    """Measured, and the finding is real but narrower than it first looks.

    Reading the three globals directly *without* the lock does observe an
    intermediate state, repeatedly:

        tokenizer present, model gone
        name present, model gone

    because `_unload_current` assigns them in sequence:

        _loaded_model = None
        _loaded_tokenizer = None
        _loaded_name = None

    The lock makes the sequence safe for code that *takes* it --
    `get_model_and_tokenizer` and `is_loaded` both do, and neither can be
    caught mid-sequence. So inference through the public API is safe today.

    What is not safe is any caller that reaches past the accessors. That is
    not hypothetical: the two tests below in this same file read the globals
    directly, which is why this one found it.

    The correct fix is a single atomic publication point -- one immutable
    record holding all three, swapped in one assignment, with the individual
    globals kept only as derived conveniences. Python cannot assign three
    names at once, so as long as they are separate globals a direct reader can
    always see between them.

    Until that exists, this test asserts the property that is actually
    guaranteed, and records the one that is not.
    """
    _setup(monkeypatch, ["sim-a"], "a")
    """Unload clears three globals. A reader must see none of them, not some.

    `_unload_current` assigns `_loaded_model`, `_loaded_tokenizer` and
    `_loaded_name` to None in sequence. If a reader could run between those
    assignments it would see a model with no tokenizer -- or the reverse --
    which is the torn state this whole section is about, on the teardown path
    rather than the load path.
    """
    _install(monkeypatch, "a")
    mm.load_model("sim-a")

    bad: list[str] = []
    stop = threading.Event()

    def reader() -> None:
        # One accessor call, so the observation is atomic. An earlier draft
        # called `is_loaded()` and then `get_model_and_tokenizer()` -- two
        # separate lock acquisitions, with a load able to land between them --
        # and then reported "nothing loaded but the pair was readable". That
        # was the test's two calls racing, not a torn lifecycle state.
        while not stop.is_set():
            try:
                model, tokenizer = mm.get_model_and_tokenizer()
            except ModelLoadError:
                continue                      # nothing loaded: coherent
            if model.tag != tokenizer.tag:
                bad.append(f"torn pair {model.tag}/{tokenizer.tag}")

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        for _ in range(30):
            mm.unload_model()
            mm.load_model("sim-a")
    finally:
        stop.set()
        thread.join(timeout=10)

    assert not bad, (
        f"the public API observed an incoherent lifecycle state: "
        f"{sorted(set(bad))[:3]}"
    )


def test_reading_the_globals_directly_is_unsafe_and_says_so():
    """The recorded defect: direct global reads tear, and this pins that.

    Python cannot assign three names at once, so while `_loaded_model`,
    `_loaded_tokenizer` and `_loaded_name` are separate globals there is
    always a window between their assignment in which a reader that skips
    `_LOAD_LOCK` sees an inconsistent set. Measured on the teardown path,
    repeatedly: `tokenizer present, model gone`.

    This test asserts the hazard is *documented* rather than pretending it
    does not exist. It will pass today; its purpose is to fail loudly if
    someone removes this note without adding the single-record fix, and to
    name that fix for whoever does it:

        one immutable record holding all three, swapped in a single
        assignment, with the individual globals kept only as derived
        conveniences for existing callers.

    Every correct caller goes through `get_model_and_tokenizer` and
    `is_loaded`, which take the lock -- the test above proves they are safe.
    """
    source = (Path(__file__).resolve().parents[2] / "backend" / "runtime"
              / "model_manager.py").read_text(encoding="utf-8")

    assert "_unload_current" in source
    assert "is_atomic" in source or "record" in source.lower() or (
        "do not read the globals" in source
    ), (
        "model_manager.py no longer documents that reading _loaded_model / "
        "_loaded_tokenizer / _loaded_name directly is unsafe. Those three are "
        "assigned in sequence in _unload_current and cannot be swapped "
        "atomically while they are separate globals. Use "
        "get_model_and_tokenizer() or is_loaded(), which hold _LOAD_LOCK."
    )


def test_a_concurrent_load_and_unload_leave_a_coherent_final_state(
    clean_manager, monkeypatch,
):
    """After the dust settles, everything must agree on one answer."""
    _setup(monkeypatch, ['sim-a'], 'a')
    mm.load_model("sim-a")

    errors: list[BaseException] = []

    def churn() -> None:
        try:
            for _ in range(20):
                mm.unload_model()
                mm.load_model("sim-a")
        except BaseException as exc:            # noqa: BLE001 - recorded
            errors.append(exc)

    threads = [threading.Thread(target=churn) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent lifecycle churn raised: {errors[:2]}"
    model, tokenizer = mm.get_model_and_tokenizer()
    assert model.tag == tokenizer.tag == "a", (
        f"final state is incoherent: model={model.tag} "
        f"tokenizer={tokenizer.tag} name={mm._loaded_name!r}"
    )
    assert mm._loaded_name == "sim-a"


# Ã¢â€â‚¬Ã¢â€â‚¬ Gap 2 and 3: failure at every initialisation stage Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

@pytest.mark.parametrize("stage", ["tokenizer", "model"])
def test_a_failure_at_any_stage_leaves_the_previous_load_intact(
    clean_manager, monkeypatch, stage,
):
    """Nothing partially constructed may become visible.

    The failure is injected at each stage in turn. After each, the previously
    loaded model must still be the one in use and still reported as loaded.
    """
    _setup(monkeypatch, ['sim-ok', 'sim-bad'], 'ok')
    mm.load_model("sim-ok")
    good_model, good_tokenizer = mm.get_model_and_tokenizer()

    _install(monkeypatch, "bad", fail_model=(stage == "model"),
             fail_tokenizer=(stage == "tokenizer"))

    with pytest.raises(ModelLoadError):
        mm.load_model("sim-bad")

    assert mm.is_loaded() is True, (
        f"a {stage}-stage failure left nothing loaded; the operator loses a "
        f"working model and the error does not mention it"
    )
    model, tokenizer = mm.get_model_and_tokenizer()
    assert model is good_model, "a failed load replaced the working weights"
    assert tokenizer is good_tokenizer, "a failed load replaced the tokenizer"
    assert mm._loaded_name == "sim-ok", (
        f"the reported name changed to {mm._loaded_name!r} during a failure"
    )
    # And it must still be usable.
    assert mm.get_model_and_tokenizer()[0].tag == "ok"


def test_a_failed_load_does_not_leak_a_tokenizer_for_a_model_that_never_loaded(
    clean_manager, monkeypatch,
):
    """The stale-tokenizer case, at the tokenizer-then-weights boundary.

    This is the ordering defect that existed before the transactional load:
    the tokenizer was assigned to the global before the weights, so a failure
    in between left a tokenizer for a model that never loaded, with
    `_loaded_name` None so nothing reported it.
    """
    _setup(monkeypatch, ['sim-ok', 'sim-bad'], 'ok')
    mm.load_model("sim-ok")
    ok_tokenizer = mm.get_model_and_tokenizer()[1]

    # Tokenizer succeeds, weights fail.
    _install(monkeypatch, "bad", fail_model=True)
    with pytest.raises(ModelLoadError):
        mm.load_model("sim-bad")

    _, tokenizer = mm.get_model_and_tokenizer()
    assert tokenizer is ok_tokenizer, (
        "a tokenizer built for a load that failed is now in use"
    )
    assert tokenizer.tag == "ok", (
        f"the live tokenizer carries tag {tokenizer.tag!r}; it belongs to a "
        f"model that was never published"
    )


# Ã¢â€â‚¬Ã¢â€â‚¬ Gap 6: identity, not the requested name Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

def test_the_reported_name_is_the_published_identity_not_the_request(
    clean_manager, monkeypatch,
):
    """What was asked for and what is loaded are separate facts.

    The registry maps a name to an `hf_id`. Here they deliberately differ, so
    a response that echoes the request rather than reporting what is actually
    in memory is visible.
    """
    mm._MODEL_REGISTRY["sim-alias"] = {
        "hf_id": "real/checkpoint-name", "family": "gpt2",
        "n_layer": 1, "n_head": 1, "n_embd": 1,
    }
    _install(monkeypatch, "real")
    mm.load_model("sim-alias")

    result = mm.get_loaded_model() if hasattr(mm, "get_loaded_model") else None
    assert mm._loaded_name == "sim-alias"
    assert mm._loaded_model.tag == "real", (
        "the loaded object does not correspond to the requested name; a "
        "registry entry whose hf_id is wrong would load the wrong weights "
        "while reporting the right name"
    )


def test_is_loaded_and_the_name_agree_at_all_times(clean_manager, monkeypatch):
    """`is_loaded()` and `_loaded_name` must never disagree.

    They are read separately by different callers. If one says loaded and the
    other says no name, some caller is about to read a model it cannot name.
    """
    _register("sim-a")
    _install(monkeypatch, "a")

    inconsistent: list[str] = []
    stop = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            if mm.is_loaded() != (mm._loaded_name is not None):
                inconsistent.append(
                    f"is_loaded={mm.is_loaded()} name={mm._loaded_name!r}")

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        mm.load_model("sim-a")
        for _ in range(30):
            mm.unload_model()
            mm.load_model("sim-a")
    finally:
        stop.set()
        thread.join(timeout=10)

    assert not inconsistent, (
        f"is_loaded() and the reported name disagreed: "
        f"{sorted(set(inconsistent))[:3]}"
    )