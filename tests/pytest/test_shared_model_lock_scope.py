"""The shared GPT-2 module must only be forwarded under MODEL_LOCK.

``backend/services/model_lock.py`` states the rule: hold MODEL_LOCK for the
entire hook-install -> forward -> hook-remove -> cache-read sequence. Two
threads that install different hooks on the same module interleave, and each
thread's capture then describes a prompt it never ran.

This is checked from the inside rather than by racing threads. A recording
lock wrapper notes, for every forward on the shared model, whether the lock
was held at that moment, so a measurement that forwards outside the lock is
reported directly. A race test would only fail on the runs where the race
happens to land; this one fails on every run where the rule is broken.

The prompt half of the same defect class is checked separately: an
intervention that names no prompt must still measure a reproducible one.
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Tuple

import pytest


def _engine_or_skip() -> Any:
    """The engine with its weights, or a skip that says why there are none."""
    from backend.services import gpt2_engine as eng

    if not eng.is_available():
        pytest.skip("torch/transformers not available")
    res = eng.load()
    if res.get("status") != "loaded":
        pytest.skip(f"GPT-2 weights unavailable: {res.get('error')}")
    return eng


class _RecordingLock:
    """Wraps MODEL_LOCK and reports whether the calling thread holds it."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._state = threading.local()

    def acquire(self, *args: Any, **kwargs: Any) -> None:
        self._inner.acquire(*args, **kwargs)
        self._state.depth = getattr(self._state, "depth", 0) + 1

    def release(self) -> None:
        self._inner.release()
        self._state.depth = getattr(self._state, "depth", 0) - 1

    def __enter__(self) -> "_RecordingLock":
        self.acquire()
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.release()
        return False

    @property
    def held(self) -> bool:
        return getattr(self._state, "depth", 0) > 0


#: Every measurement that touches the shared model, as (name, callable).
#: Each one must forward on the shared module, so each one must hold the
#: lock while it does.
def _measurements() -> List[Tuple[str, Callable[[Any], Any]]]:
    return [
        ("run_prompt",
         lambda eng: eng.run_prompt("The capital of France is")),
        ("attention_head",
         lambda eng: eng.attention_head(0, 0, prompt="The capital of France is")),
        ("activations",
         lambda eng: eng.activations(0, prompt="The capital of France is")),
        ("patch_head",
         lambda eng: eng.patch_head(0, 0, " Paris", " London")),
        ("patch_neuron",
         lambda eng: eng.patch_neuron(0, 0, 1.0, prompt="The capital of France is")),
        ("ablate_layer",
         lambda eng: eng.ablate_layer(0, prompt="The capital of France is")),
        ("steer",
         lambda eng: eng.steer("The capital of France is", 0, " Paris", " London")),
        ("ioi",
         lambda eng: eng.ioi("John", "Mary")),
        ("generate_text",
         lambda eng: eng.generate_text("The", max_new_tokens=2)),
    ]


@pytest.mark.parametrize("name,call", _measurements(),
                         ids=[m[0] for m in _measurements()])
def test_shared_model_forward_runs_under_model_lock(
        monkeypatch: pytest.MonkeyPatch, name: str, call: Callable[[Any], Any]) -> None:
    """A forward on the shared model must happen while MODEL_LOCK is held."""
    eng = _engine_or_skip()
    recorder = _RecordingLock(eng.MODEL_LOCK)
    monkeypatch.setattr(eng, "MODEL_LOCK", recorder)

    observed: List[bool] = []
    handle = eng._model.register_forward_pre_hook(
        lambda module, args: observed.append(recorder.held))

    saved_cache = eng._cache
    try:
        # An empty cache forces the forward to actually run: with the prompt
        # already cached, the engine legitimately skips it, and this test
        # would then prove nothing.
        eng._cache = {}
        if name == "patch_head":
            # patch_head measures the cached prompt, so the cache has to be
            # primed before it. run_prompt holds the lock for its own
            # forward; only the patch_head forwards are under test here.
            eng.run_prompt("The capital of France is")
        observed.clear()
        call(eng)
    finally:
        handle.remove()
        eng._cache = saved_cache

    assert observed, (
        f"{name} ran no forward pass on the shared model, so this test "
        f"proves nothing about it"
    )
    assert all(observed), (
        f"{name} forwarded on the shared model outside MODEL_LOCK "
        f"({observed.count(False)} of {len(observed)} forwards unlocked)"
    )


def test_patch_neuron_measures_the_prompt_it_was_given() -> None:
    """A named prompt is the prompt measured, and the record says so."""
    eng = _engine_or_skip()

    res = eng.patch_neuron(0, 0, 1.0, prompt="The capital of France is")

    assert res.get("status") == "ok", res
    assert res.get("prompt") == "The capital of France is", res
    assert res.get("prompt_defaulted") is False, res


def test_patch_neuron_without_a_prompt_measures_a_reproducible_one() -> None:
    """No prompt named: the engine picks one, and picks the same one twice.

    The previous fallback drew from a pool with ``random.choice``, so two
    identical requests measured different prompts and neither was
    reproducible. The record also has to say the prompt was engine-chosen.
    """
    eng = _engine_or_skip()

    saved = eng._cache
    eng._cache = {}
    try:
        first = eng.patch_neuron(0, 0, 1.0)
        second = eng.patch_neuron(0, 0, 1.0)
    finally:
        eng._cache = saved

    assert first.get("status") == "ok", first
    assert first.get("prompt"), first
    assert first["prompt"] == second["prompt"], (
        "two identical requests measured different prompts: "
        f"{first.get('prompt')!r} and {second.get('prompt')!r}"
    )
    assert first.get("prompt_defaulted") is True, first


def test_engine_never_samples_a_measurement_prompt_at_random() -> None:
    """The prompt fallback is fixed, not sampled.

    A source guard rather than a behavioural one: the sampler could be
    reintroduced behind any other code path, and the pool it drew from was
    exactly the reason two identical requests used to disagree.
    """
    from source_assert import executable_source

    from backend.services import gpt2_engine as eng

    src = executable_source(eng)
    assert "random.choice" not in src, (
        "the measurement-prompt fallback samples at random again"
    )
    assert "import random" not in src, (
        "the engine imports random again; the prompt fallback must be fixed"
    )


def test_executor_patch_neuron_passes_its_prompt_through() -> None:
    """The agent path must not measure a prompt the caller never chose."""
    import asyncio
    from unittest import mock

    from backend.agents import executor as ex

    seen: List[Tuple[Any, ...]] = []

    async def fake_call(_self: Any, name: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        seen.append((name, *args))
        return {"status": "ok"}

    with mock.patch.object(ex.Executor, "_call", fake_call):
        asyncio.run(ex.Executor().patch_neuron(5, 42, 3.5, "The capital of France is"))

    assert seen, "executor.patch_neuron did not call the engine"
    name, layer, neuron_index, patch_value, prompt = seen[0]
    assert name == "patch_neuron"
    assert (layer, neuron_index, patch_value) == (5, 42, 3.5)
    assert prompt == "The capital of France is", (
        "the executor dropped the prompt it was given"
    )


def test_executor_patch_neuron_defaults_to_a_named_prompt() -> None:
    """The default path must also name a prompt, not leave it to the cache."""
    import asyncio
    from unittest import mock

    from backend.agents import executor as ex

    seen: List[Tuple[Any, ...]] = []

    async def fake_call(_self: Any, name: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        seen.append((name, *args))
        return {"status": "ok"}

    with mock.patch.object(ex.Executor, "_call", fake_call):
        asyncio.run(ex.Executor().patch_neuron(5, 42))

    assert seen, "executor.patch_neuron did not call the engine"
    prompt = seen[0][-1]
    assert prompt, (
        "executor.patch_neuron sent no prompt, so the measurement falls "
        "back to whatever prompt the shared cache happens to hold"
    )
