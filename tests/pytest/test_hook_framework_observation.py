"""C1: observing activations must not change the forward pass.

The hook framework exists to *watch* a model. A hook that altered the
computation it observes would make every downstream measurement wrong in a
way no consumer could detect: the numbers would look entirely plausible.

This is the first test in a vertical slice through the runtime hook seam
(`backend.runtime.hook_framework.HookManager`). It is deliberately the
foundation -- if observation is not neutral, nothing else the framework
produces can be trusted, so it is asserted before any capture-shape or
lifecycle test.

The model is a randomly-initialised two-layer GPT-2 built from a small
config rather than downloaded weights. No test in this file needs trained
weights: neutrality of observation is a property of the hook mechanism, not
of what the model has learned. Using a tiny real `nn.Module` instead of a
fake also means the assertion covers genuine PyTorch hook dispatch.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers", reason="hook dispatch needs a real nn.Module")

from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402

from backend.runtime.hook_framework import HookManager  # noqa: E402


@pytest.fixture
def tiny_model():
    """A real GPT-2 module, small enough to run in milliseconds.

    Random weights are fine and intentional: see the module docstring.
    """
    config = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=32,
        n_positions=16,
        vocab_size=64,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


@pytest.fixture
def tokens():
    return torch.tensor([[1, 2, 3]])


def test_observation_does_not_change_the_logits(tiny_model, tokens):
    """A residual hook that only reads must leave the forward pass identical.

    Asserted with `assert_close` at a tolerance tight enough to catch a
    genuine numerical perturbation, loose enough not to trip on float32
    nondeterminism across two separate calls.
    """
    with torch.no_grad():
        baseline = tiny_model(tokens).logits

    manager = HookManager()
    manager.bind(tiny_model)
    captured: list[torch.Tensor] = []
    manager.register_hook(0, "residual", store=captured)

    try:
        with torch.no_grad():
            observed = tiny_model(tokens).logits

        assert captured, "the hook never fired, so observation proved nothing"
        torch.testing.assert_close(
            observed, baseline, rtol=1e-5, atol=1e-6
        )
    finally:
        manager.clear()


def test_a_disabled_hook_stops_capturing(tiny_model, tokens):
    """`disable_hook` must stop the hook observing, not just relabel it.

    The distinction matters because `HookHandle.enabled` and the PyTorch
    forward hook that actually fires are two different things. Toggling the
    flag without the closure consulting it produces a handle that *reports*
    disabled while continuing to capture -- a hook the caller believes is
    dormant is still consuming memory on every forward pass.
    """
    manager = HookManager()
    manager.bind(tiny_model)
    captured: list[torch.Tensor] = []
    handle = manager.register_hook(0, "residual", store=captured)

    try:
        with torch.no_grad():
            tiny_model(tokens)
        assert len(captured) == 1, "setup: the hook should capture once"

        manager.disable_hook(handle)
        assert handle.enabled is False, "the handle must report itself disabled"

        captured.clear()
        with torch.no_grad():
            tiny_model(tokens)

        assert captured == [], (
            "a hook reported as disabled still captured %d activation(s); the "
            "enabled flag is not consulted by the capture path"
            % len(captured)
        )
    finally:
        manager.clear()


def test_a_re_enabled_hook_resumes_capturing(tiny_model, tokens):
    """Disable/enable must be reversible, not a one-way trip to silent.

    Pairs with the disable test. A fix that satisfied that test by simply
    never capturing again would pass it while breaking every caller that
    toggles a hook around a region of interest -- which is the documented
    reason `enable_hook` exists at all.
    """
    manager = HookManager()
    manager.bind(tiny_model)
    captured: list[torch.Tensor] = []
    handle = manager.register_hook(0, "residual", store=captured)

    try:
        manager.disable_hook(handle)
        with torch.no_grad():
            tiny_model(tokens)
        assert captured == [], "setup: a disabled hook should capture nothing"

        manager.enable_hook(handle)
        assert handle.enabled is True
        with torch.no_grad():
            tiny_model(tokens)

        assert len(captured) == 1, (
            "a re-enabled hook captured %d activation(s), expected 1"
            % len(captured)
        )
    finally:
        manager.clear()


def test_remove_hook_detaches_it_from_the_module(tiny_model, tokens):
    """`remove_hook` must unregister, not merely forget.

    `HookManager` keeps its own dict of handles, so a removal that only
    dropped the dict entry would look correct while the PyTorch forward hook
    stayed on the module -- still firing, still filling a store the caller no
    longer holds a reference to. That leak is invisible until memory grows,
    which is precisely why it is worth a test rather than a code reading.

    Asserted against the module's own registry of forward hooks rather than
    against observable behaviour, so a hook that keeps firing but writes
    somewhere harmless still fails.
    """
    manager = HookManager()
    manager.bind(tiny_model)

    captured: list[torch.Tensor] = []
    handle = manager.register_hook(0, "residual", store=captured)

    block = tiny_model.transformer.h[0]
    assert len(block._forward_hooks) == 1, "setup: expected one registered hook"

    manager.remove_hook(handle)

    assert len(block._forward_hooks) == 0, (
        "the forward hook is still registered on the block after remove_hook"
    )
    assert manager.list_hooks() == []

    with torch.no_grad():
        tiny_model(tokens)
    assert captured == [], "a removed hook must not capture"


def test_clear_detaches_every_hook_after_a_failed_forward_pass(
    tiny_model, tokens
):
    """A forward pass that raises must not leave hooks attached.

    A research session that hits one exception -- an OOM mid-ablation, a bad
    token index -- and continues would otherwise accumulate hooks silently.
    Each one keeps its store alive and keeps firing, so the model's cost per
    forward pass grows without any visible symptom, and the next experiment
    measures a model that is no longer the one it thinks it is.

    The failure is raised from inside the hooked forward pass on purpose:
    that is the window in which a caller holding no `try/finally` leaks.
    """
    manager = HookManager()
    manager.bind(tiny_model)

    stores: list[list[torch.Tensor]] = []
    for layer in range(2):
        stores.append([])
        manager.register_hook(layer, "residual", store=stores[layer])

    calls = {"n": 0}
    original_forward = tiny_model.forward

    def exploding_forward(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated forward-pass failure")
        return original_forward(*args, **kwargs)

    tiny_model.forward = exploding_forward

    try:
        with pytest.raises(RuntimeError, match="simulated forward-pass failure"):
            tiny_model(tokens)

        assert manager.list_hooks(), (
            "setup: the exception escaped without the manager losing track"
        )

        manager.clear()

        for layer in range(2):
            block = tiny_model.transformer.h[layer]
            assert len(block._forward_hooks) == 0, (
                "layer %d still has %d forward hook(s) attached after clear() "
                "following a failed forward pass"
                % (layer, len(block._forward_hooks))
            )
        assert manager.list_hooks() == []

        # And the model still works, with nothing watching.
        tiny_model.forward = original_forward
        with torch.no_grad():
            tiny_model(tokens)
        for store in stores:
            assert store == [], "a cleared hook captured after cleanup"
    finally:
        tiny_model.forward = original_forward
        manager.clear()


def test_the_attention_hook_raises_nothing_when_weights_are_unavailable(
    tiny_model, tokens
):
    """Attention capture must fail loudly, never crash the forward pass.

    `AttentionHook` reads `outputs[-1]`, which is the attention weights only
    when the attention implementation actually returns them. Under the
    `sdpa` implementation -- the default in transformers 4.57, which is what
    MECH loads -- that slot is `None`, and `.detach()` on it raises
    `AttributeError` *inside the forward pass*.

    The severity is the point. A hook that raises propagates out of the model
    call, so an observation attempt takes down an experiment that was
    otherwise fine. Worse, the traceback names `'NoneType' object has no
    attribute 'detach'` rather than anything about attention configuration,
    so the operator is sent looking at the wrong layer of the stack.

    Expected value here: no exception, and the hook either captures real
    weights or records nothing -- never a crash.
    """
    manager = HookManager()
    manager.bind(tiny_model)
    captured: list[torch.Tensor] = []
    manager.register_hook(0, "attention", store=captured)

    try:
        # Must not raise. This is the assertion; the old code raised here.
        with torch.no_grad():
            tiny_model(tokens, output_attentions=True)

        for tensor in captured:
            assert tensor.dim() == 4, (
                "attention capture has %d dimensions, expected 4 "
                "[batch, heads, seq, seq]" % tensor.dim()
            )
    finally:
        manager.clear()


def test_the_mlp_hook_captures_the_mlp_hidden_not_the_block_input(
    tiny_model, tokens
):
    """`MLPHook` must capture width `4*n_embd`, not `n_embd`.

    The hook's job, per its own docstring, is "MLP GELU activations". The MLP's
    activations are the post-`c_fc` hidden state, which GPT-2 widens to
    `4*n_embd`. The previous implementation registered on the `mlp` module and
    computed `module.act(inputs[0])` -- that is the activation function
    applied to the block *input*, so it returned width `n_embd` and contained
    no MLP computation whatsoever.

    The widths differ, which is what makes this checkable rather than a
    matter of taste: at `n_embd=32` the hook returned `[batch, seq, 32]`
    where the real MLP hidden is `[batch, seq, 128]`. A caller indexing
    neurons against the true width would read past the end or silently get
    the wrong slice.
    """
    manager = HookManager()
    manager.bind(tiny_model)
    captured: list[torch.Tensor] = []
    manager.register_hook(1, "mlp", store=captured)

    # Independent reference, captured by a plain PyTorch hook on c_fc: the
    # hook under test must equal act(c_fc_output), computed by a different
    # route. Without this the assertion would only pin a width, which any
    # tensor of the right size would satisfy.
    block = tiny_model.transformer.h[1]
    raw_c_fc: list[torch.Tensor] = []
    probe = block.mlp.c_fc.register_forward_hook(
        lambda module, inputs, outputs: raw_c_fc.append(outputs.detach().clone())
    )

    try:
        with torch.no_grad():
            tiny_model(tokens)

        assert captured, "the mlp hook never fired"
        assert raw_c_fc, "the reference probe never fired"

        width = captured[0].shape[-1]
        expected = 4 * tiny_model.config.n_embd

        assert width == expected, (
            "MLPHook captured width %d, expected %d (4 x n_embd = %d). The "
            "capture is the block input, not the MLP hidden state."
            % (width, expected, tiny_model.config.n_embd)
        )

        reference = block.mlp.act(raw_c_fc[0])
        torch.testing.assert_close(
            captured[0], reference.cpu(), rtol=1e-6, atol=1e-7
        )
    finally:
        probe.remove()
        manager.clear()