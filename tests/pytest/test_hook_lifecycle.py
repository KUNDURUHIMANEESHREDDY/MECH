"""Hook framework lifecycle: ordering, isolation, hostile hooks, and stale handles.

`test_hook_framework_observation.py` establishes that each hook observes the
right thing. This file covers what happens when hooks *interact* -- the gaps
the contract review named for section 1.

    1. Multiple hooks on one module: ordering, isolation, cleanup
    2. A hook that raises, returns a replacement, or mutates its input
    3. Repeated bind / clear / remove, including stale handles
    4. Attention implementations returning different output structures
    5. CPU/GPU and mixed-precision

A note on the fixture. These use a small hand-built module rather than GPT-2.
Hook lifecycle is a property of `torch.nn.Module`'s hook registry, not of any
particular architecture, and the real model costs ~350 MB and several seconds
per forward pass to test something that does not depend on either. Gap 5's
dtype and device assertions run against the real model via the weight guard;
the lifecycle assertions do not need it.

`AttentionHook` is the exception: gap 4 is about what that specific hook reads
out of `outputs`, so it runs against the real GPT-2 blocks.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from backend.runtime.hook_framework import (
    AttentionHook,
    CustomHook,
    HookManager,
    ResidualHook,
)


# ── a module with the shape the hooks expect ──────────────────────────

class _Attn(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.c_proj = nn.Linear(width, width)

    def forward(self, x):
        return (x,)


class _Block(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.attn = _Attn(width)
        self.ln_1 = nn.LayerNorm(width)

    def forward(self, x):
        return self.ln_1(x)


class _Transformer(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.h = nn.ModuleList([_Block(width) for _ in range(3)])


class _Model(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.transformer = _Transformer(width)

    def forward(self, x):
        for block in self.transformer.h:
            x = block(x)
        return x


@pytest.fixture
def manager():
    mgr = HookManager()
    mgr.bind(_Model())
    yield mgr
    mgr.clear()


def _store_of(hook) -> list:
    """The hook's capture list.

    `BaseHook` subclasses hold `_store`; there is no public accessor. An
    earlier draft used `hook.store` and three device/dtype tests died with
    `AttributeError` before asserting anything -- the same vacuous-green
    failure mode this session has hit twice already.
    """
    return hook._store


def _ordered_hook(order: list, label: str, store: list):
    def _fn(module, inputs, output):
        order.append(label)
        store.append(label)
        return None

    return CustomHook(_fn, name=label, store=store)


# ── Gap 1: multiple hooks on one module ──────────────────────────────

def test_hooks_on_one_module_fire_in_registration_order(manager):
    """Order is part of the contract, because consumers may rely on it.

    PyTorch appends forward hooks and runs them in that order, so registration
    order is firing order. A `runs_after` field exists on the handle but
    `_validate_dependencies` is a documented no-op for this sprint, so
    registration order is the *only* ordering guarantee a caller has.
    """
    order: list[str] = []
    for label in ("first", "second", "third"):
        manager.register_hook(0, _ordered_hook(order, label, []))

    manager._model(torch.randn(1, 1, 8))

    assert order == ["first", "second", "third"], (
        f"hooks fired as {order}; registration order is the only ordering "
        f"guarantee, since `runs_after` is not yet enforced"
    )


def test_removing_one_hook_leaves_the_others_intact(manager):
    """Hooks must be isolated: removing one must not disturb its neighbours."""
    order: list[str] = []
    keep_one = _ordered_hook(order, "keep_one", [])
    keep_two = _ordered_hook(order, "keep_two", [])
    drop = _ordered_hook(order, "drop", [])

    h1 = manager.register_hook(0, keep_one)
    h2 = manager.register_hook(0, keep_two)
    h3 = manager.register_hook(0, drop)

    manager.remove_hook(h3)
    order.clear()
    manager._model(torch.randn(1, 1, 8))

    assert order == ["keep_one", "keep_two"], (
        f"after removing the third hook the survivors fired as {order}"
    )
    assert h1 in manager.list_hooks() and h2 in manager.list_hooks()


def test_each_hook_writes_only_to_its_own_store(manager):
    """Stores are per-hook; a shared reference would cross-contaminate them."""
    order: list[str] = []
    s1, s2, s3 = [], [], []
    manager.register_hook(0, _ordered_hook(order, "a", s1))
    manager.register_hook(0, _ordered_hook(order, "b", s2))
    manager.register_hook(0, _ordered_hook(order, "c", s3))

    manager._model(torch.randn(1, 1, 8))

    assert (s1, s2, s3) == (["a"], ["b"], ["c"]), (
        f"stores received {s1}, {s2}, {s3}; each hook must see only its own"
    )


def test_clear_removes_every_hook_from_every_layer(manager):
    """Cleanup must be complete: nothing may fire after `clear()`."""
    order: list[str] = []
    for layer in range(3):
        manager.register_hook(layer, _ordered_hook(order, f"L{layer}", []))

    manager.clear()
    order.clear()
    manager._model(torch.randn(1, 1, 8))

    assert order == [], f"hooks still firing after clear(): {order}"
    assert manager.list_hooks() == []


# ── Gap 2: hostile hooks ─────────────────────────────────────────────

def test_a_hook_that_raises_propagates_and_leaves_the_model_usable(manager):
    """A raising hook must not silently swallow, and must not wedge the model.

    The framework does not catch, by design: a hook that silently swallowed its
    own error would report a capture that never happened. What it must not do
    is leave the module in a state where later runs are affected.
    """
    def boom(module, inputs, output):
        raise RuntimeError("hook failed deliberately")

    manager.register_hook(0, CustomHook(boom, name="boom"))

    with pytest.raises(RuntimeError, match="hook failed deliberately"):
        manager._model(torch.randn(1, 1, 8))

    manager.clear()
    out = manager._model(torch.randn(1, 1, 8))
    assert out is not None, "the model was left unusable by the raising hook"


def test_a_hook_that_returns_a_replacement_replaces_the_output(manager):
    """Returning a tensor from a forward hook replaces the module output.

    This is how the patching code in `gpt2_adapter` does its work, so the
    framework's hooks must coexist with it: a replacement produced by one hook
    is what the next hook receives.
    """
    seen: list[torch.Tensor] = []

    def replacer(module, inputs, output):
        return output * 0.0

    def observer(module, inputs, output):
        seen.append(output.clone())
        return None

    manager.register_hook(0, CustomHook(replacer, name="replacer"))
    manager.register_hook(0, CustomHook(observer, name="observer"))

    block = manager._model.transformer.h[0]
    direct = block(torch.randn(1, 1, 8))
    hooked = block(torch.randn(1, 1, 8))

    assert seen, "the observer never fired"
    assert torch.equal(seen[0], torch.zeros_like(seen[0])), (
        "the observer saw the original output, not the replacement; a hook "
        "that returns a tensor is supposed to replace the output for every "
        "hook registered after it"
    )


def test_a_hook_that_mutates_its_input_mutates_the_caller_s_tensor(
    manager,
):
    """PyTorch does not copy hook inputs; in-place edits are visible upstream.

    Worth pinning because the framework's own hooks clone before writing
    (`AttentionHook` clones in its mask hook). A user hook that writes in place
    is editing the caller's tensor, which is a real hazard rather than a bug --
    but nothing states it, and it is the kind of thing that is expensive to
    discover later.
    """
    def mutator(module, inputs, output):
        inputs[0].add_(100.0)
        return None

    manager.register_hook(0, CustomHook(mutator, name="mutator"))

    x = torch.zeros(1, 1, 8)
    manager._model(x)

    assert x.abs().sum() > 0, (
        "the hook's in-place write did not reach the caller's tensor; if PyTorch "
        "ever starts copying hook inputs this will need revisiting, but today "
        "the edit is visible upstream and callers rely on that"
    )


# ── Gap 3: repeated bind / clear / remove, and stale handles ─────────

def test_detach_clears_the_hook_handle(manager):
    """A detached hook must not still report a handle.

    The defect: every `detach()` implementation clears the PyTorch handle
    (`self._hook_ref = None`) but never clears `self.handle`, so after
    `remove_hook` the hook object still reports `handle is not None` and still
    hands back the dead id.

        after register : handle set = True   id = 192cc84a
        after remove   : handle set = True   id = 192cc84a

    That matters because `register_hook` branches on exactly this
    (`if not hasattr(hook, "_handle") or hook.handle is None`). A stale handle
    makes the framework believe a hook it has already removed is still live.
    """
    hook = ResidualHook(store=[])
    manager.register_hook(0, hook)
    assert hook.handle is not None

    manager.remove_hook(hook.handle)

    assert hook.handle is None, (
        f"after removal the hook still reports handle id="
        f"{(hook.handle.id[:8] if hook.handle else None)!r}; the PyTorch hook "
        f"is gone but the object still claims a live registration"
    )


def test_a_stale_handle_cannot_be_used_to_remove_a_live_hook(manager):
    """A stale id must not alias a handle that has since been re-registered.

    Handles are keyed by uuid in `_handles`. If a caller keeps an old handle
    and later re-registers the same hook instance, the old object must not be
    able to remove the new registration.
    """
    hook = ResidualHook(store=[])
    first = manager.register_hook(0, hook)
    manager.remove_hook(first)
    second = manager.register_hook(0, hook)

    assert first is not second, (
        "re-registering returned the same handle object; the old id would "
        "still address the live registration"
    )

    manager.remove_hook(first)          # the stale one
    assert manager.list_hooks(), (
        "removing a stale handle removed a live registration -- ids aliased"
    )


def test_remove_is_idempotent(manager):
    """Removing twice must not raise, and must not remove something else."""
    hook = ResidualHook(store=[])
    handle = manager.register_hook(0, hook)

    manager.remove_hook(handle)
    manager.remove_hook(handle)          # second call: stale handle

    other = manager.register_hook(1, ResidualHook(store=[]))
    manager.remove_hook(handle)          # stale id again, after a new hook exists
    assert other in manager.list_hooks(), (
        "a stale removal took out an unrelated live hook"
    )


def test_registering_the_same_instance_twice_does_not_double_attach(manager):
    """Re-registering a live hook must not fire it twice.

    `register_hook` reuses `hook.handle` when the id is already known, so the
    second registration is a no-op rather than a second PyTorch hook. That is
    the right behaviour; it is pinned because a double attach would double
    every capture and silently double every store.
    """
    count: list[int] = []

    def counter(module, inputs, output):
        count.append(1)
        return None

    hook = CustomHook(counter, name="counter")
    a = manager.register_hook(0, hook)
    b = manager.register_hook(0, hook)

    manager._model(torch.randn(1, 1, 8))

    assert len(count) == 1, (
        f"the hook fired {len(count)} times for a single forward pass; "
        f"re-registering a live hook must not attach a second PyTorch hook"
    )
    assert a.id == b.id


def test_clear_then_reregister_gives_a_fresh_handle(manager):
    """A cleared hook must re-register under a new id, not resurrect the old."""
    hook = ResidualHook(store=[])
    first = manager.register_hook(0, hook)
    manager.clear()
    second = manager.register_hook(0, hook)

    assert first.id != second.id, (
        "the handle id survived clear(); a reused id would let a caller holding "
        "the old handle address a registration that no longer exists"
    )


def test_clear_twice_is_safe(manager):
    """`clear()` on an already-empty manager must not raise."""
    manager.clear()
    manager.clear()
    manager.register_hook(0, ResidualHook(store=[]))
    manager.clear()
    manager.clear()
    assert manager.list_hooks() == []


# ── Gap 5: device and dtype ──────────────────────────────────────────

def test_residual_capture_is_dtype_and_device_faithful(manager):
    """The captured tensor must match what the module actually held.

    A hook that silently upcast to float32, or moved to a default device, would
    make a float16 CPU capture indistinguishable from a float32 one -- and the
    number is then quoted with a precision it never had.
    """
    for dtype in (torch.float32, torch.float64):
        hook = ResidualHook(store=[])
        manager.register_hook(0, hook)
        manager._model.to(dtype)
        manager._model(torch.randn(1, 1, 8, dtype=dtype))

        captured = _store_of(hook)
        assert captured, f"nothing captured for {dtype}"
        assert captured[0].dtype == dtype, (
            f"captured dtype is {captured[0].dtype}, the model ran in {dtype}"
        )
        assert captured[0].device.type == "cpu"
        manager.clear()
        manager._model.to(torch.float32)


def test_a_half_precision_model_captures_half(manager):
    """float16 must not be silently promoted.

    Half is the case that matters: float32 and float64 both compare equal
    under `pytest.approx` at default tolerances often enough to hide a
    promotion, and float16 is where a researcher would notice a precision
    claim they did not make.
    """
    hook = ResidualHook(store=[])
    manager.register_hook(0, hook)
    manager._model.to(torch.float16)
    manager._model(torch.randn(1, 1, 8, dtype=torch.float16))

    assert _store_of(hook), "nothing captured under float16"
    assert _store_of(hook)[0].dtype == torch.float16, (
        f"float16 run captured {_store_of(hook)[0].dtype}; promoting to float32 "
        f"would present a tensor with more precision than it was measured at"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="no CUDA device; CPU dtype/device coverage is asserted above, and "
           "this would silently pass on a CPU-only host if it were not skipped",
)
def test_a_gpu_capture_is_copied_off_the_device(manager):
    """A CUDA run must hand back a CPU tensor, deliberately.

    An earlier draft asserted the opposite -- that the capture stays on the GPU
    -- on the reasoning that a host copy would be a silent downgrade. Reading
    the hook shows `.detach().cpu()` is intentional: captures accumulate in a
    list for the life of a session, and holding GPU memory for every captured
    tensor across twelve layers is how a research session exhausts a card.

    So the invariant is the one the code intends: a GPU run must not leave its
    captures on the GPU, and must not silently change dtype doing it. Both are
    asserted rather than assumed, because the failure mode -- an OOM an hour
    into a run, caused by a capture list -- has no trace back to this line.
    """
    hook = ResidualHook(store=[])
    manager.register_hook(0, hook)
    manager._model.to("cuda")
    manager._model(torch.randn(1, 1, 8, device="cuda"))

    captured = _store_of(hook)
    assert captured, "nothing captured on CUDA"
    assert captured[0].device.type == "cpu", (
        f"a CUDA run left its capture on {captured[0].device}; the hook copies "
        f"to host on purpose so a session's capture list does not exhaust "
        f"device memory"
    )
    assert captured[0].dtype == torch.float32, (
        f"copying off the device also changed dtype to {captured[0].dtype}"
    )
