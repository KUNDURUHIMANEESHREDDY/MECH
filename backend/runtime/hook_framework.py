"""Epic 1: Hook Framework — plugin architecture.

Provides a HookManager that registers, removes, enables, disables,
and lists hooks.  Each hook is a class that wraps a PyTorch
forward hook on a specific module within the model.

Any future AI can register a custom hook without touching the runtime.
"""

from __future__ import annotations

import abc
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from .event_bus import bus, HOOK_REGISTERED, HOOK_REMOVED, HOOK_ERROR
from .errors import HookError


# ── Hook base ────────────────────────────────────────────────────


@dataclass
class HookHandle:
    """Returned when a hook is registered.  Use to remove/enable/disable."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    enabled: bool = True
    layer_idx: int = 0
    hook_type: str = "custom"
    depends_on: list[str] = field(default_factory=list)
    runs_after: list[str] = field(default_factory=list)


class BaseHook(abc.ABC):
    """Abstract hook.  Subclass to create a new hook type."""

    @abc.abstractmethod
    def attach(self, model: nn.Module, layer_idx: int) -> nn.Module:
        """Attach this hook to the given model layer. Return the registered hook handle."""
        ...

    @abc.abstractmethod
    def detach(self) -> None:
        """Remove the hook from the model."""
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable hook name."""
        ...

    @property
    def handle(self) -> Optional[HookHandle]:
        return getattr(self, "_handle", None)

    @handle.setter
    def handle(self, h: HookHandle) -> None:
        self._handle = h

    @property
    def is_enabled(self) -> bool:
        """Whether this hook is currently permitted to observe.

        A disabled hook still holds its PyTorch forward hook -- removing and
        re-registering it would change hook ordering, which other hooks on the
        same module observe. So disabling is enforced by *consulting* this flag
        in the capture path and returning early, not by detaching.

        Without this check, `disable_hook` only relabelled the handle: the
        closure went on capturing, so a caller who disabled a hook to stop the
        memory cost kept paying it and kept filling the store.
        """
        return self.handle is None or self.handle.enabled


# ── Concrete hooks ───────────────────────────────────────────────


class EmbeddingHook(BaseHook):
    """Captures token embeddings."""

    def __init__(self, store: list | None = None):
        self._store = store if store is not None else []
        self._hook_ref = None
        self._layer_idx = 0

    def attach(self, model: nn.Module, layer_idx: int) -> HookHandle:
        self._layer_idx = layer_idx

        def _hook(module, inputs, outputs):
            if not self.is_enabled:
                return
            self._store.append(outputs[0].detach().cpu() if isinstance(outputs, tuple) else outputs.detach().cpu())

        self._hook_ref = model.transformer.wte.register_forward_hook(_hook)
        h = HookHandle(name=self.name, layer_idx=layer_idx, hook_type="embedding")
        self.handle = h
        bus.emit(HOOK_REGISTERED, hook_name=self.name, layer=layer_idx)
        return h

    def detach(self) -> None:
        if self._hook_ref is not None:
            self._hook_ref.remove()
            self._hook_ref = None
            # Clear the handle too. `register_hook` branches on exactly this
            # (`hook.handle is None`) to decide whether to attach, so a hook
            # that has been detached but still reports a handle claims a live
            # registration carrying a dead id. Measured: after `remove_hook`,
            # every hook type still returned the old handle.
            self.handle = None
            bus.emit(HOOK_REMOVED, hook_name=self.name, layer=self._layer_idx)

    @property
    def name(self) -> str:
        return "embedding"


class AttentionHook(BaseHook):
    """Captures true Q×K^T attention weights for a specific layer."""

    def __init__(self, store: list | None = None):
        self._store = store if store is not None else []
        self._hook_ref = None
        self._layer_idx = 0

    def attach(self, model: nn.Module, layer_idx: int) -> HookHandle:
        self._layer_idx = layer_idx
        block = model.transformer.h[layer_idx]

        def _hook(module, inputs, outputs):
            if not self.is_enabled:
                return
            # outputs[-1] = attention weights [batch, heads, seq, seq]
            #
            # Only when the attention implementation returns them. Under `sdpa`
            # -- the default from transformers 4.57 -- the slot is None, and
            # calling .detach() on it raised AttributeError from inside the
            # forward pass, so merely *asking* for attention maps crashed an
            # otherwise healthy run, with a traceback pointing at NoneType
            # rather than at attention configuration.
            #
            # Capturing nothing is the honest outcome: there is nothing to
            # capture. Silently substituting a zero tensor, or reshaping some
            # other output into the right number of dimensions, would hand the
            # caller an attention map that was never computed.
            attn = outputs[-1] if outputs is not None else None
            if not isinstance(attn, torch.Tensor):
                bus.emit(
                    HOOK_ERROR,
                    hook_name=self.name,
                    layer=self._layer_idx,
                    reason=(
                        "attention weights are not exposed by this attention "
                        "implementation; captured nothing. Set "
                        "attn_implementation='eager' to observe weights."
                    ),
                )
                return
            self._store.append(attn.detach().cpu())

        self._hook_ref = block.attn.register_forward_hook(_hook)
        h = HookHandle(name=self.name, layer_idx=layer_idx, hook_type="attention")
        self.handle = h
        bus.emit(HOOK_REGISTERED, hook_name=self.name, layer=layer_idx)
        return h

    def detach(self) -> None:
        if self._hook_ref is not None:
            self._hook_ref.remove()
            self._hook_ref = None
            # Clear the handle too. `register_hook` branches on exactly this
            # (`hook.handle is None`) to decide whether to attach, so a hook
            # that has been detached but still reports a handle claims a live
            # registration carrying a dead id. Measured: after `remove_hook`,
            # every hook type still returned the old handle.
            self.handle = None
            bus.emit(HOOK_REMOVED, hook_name=self.name, layer=self._layer_idx)

    @property
    def name(self) -> str:
        return "attention"


class MLPHook(BaseHook):
    """Captures MLP GELU activations for a specific layer.

    "MLP activations" means the post-`c_fc` hidden state at width
    `4 * n_embd`, which is what the neurons in this dictionary actually are.
    It does *not* mean the activation function applied to the block input --
    that tensor is `n_embd` wide and contains no MLP computation at all.
    """

    def __init__(self, store: list | None = None):
        self._store = store if store is not None else []
        self._hook_ref = None
        self._layer_idx = 0

    def attach(self, model: nn.Module, layer_idx: int) -> HookHandle:
        self._layer_idx = layer_idx
        block = model.transformer.h[layer_idx]
        act_fn = block.mlp.act

        # Hook c_fc, not the whole mlp module. The mlp module's forward
        # *input* is the block's residual stream (width n_embd); c_fc's
        # forward output is the widened hidden state (width 4 * n_embd).
        # Hooking the mlp module and calling `act(inputs[0])` therefore
        # captured gelu(block_input) -- a tensor one-quarter the expected
        # width, and a quantity the layer's neurons have no relationship to.

        def _hook(module, inputs, outputs):
            if not self.is_enabled:
                return
            hidden = act_fn(outputs) if outputs is not None else None
            if isinstance(hidden, torch.Tensor):
                self._store.append(hidden.detach().cpu())

        self._hook_ref = block.mlp.c_fc.register_forward_hook(_hook)
        h = HookHandle(name=self.name, layer_idx=layer_idx, hook_type="mlp")
        self.handle = h
        bus.emit(HOOK_REGISTERED, hook_name=self.name, layer=layer_idx)
        return h

    def detach(self) -> None:
        if self._hook_ref is not None:
            self._hook_ref.remove()
            self._hook_ref = None
            # Clear the handle too. `register_hook` branches on exactly this
            # (`hook.handle is None`) to decide whether to attach, so a hook
            # that has been detached but still reports a handle claims a live
            # registration carrying a dead id. Measured: after `remove_hook`,
            # every hook type still returned the old handle.
            self.handle = None
            bus.emit(HOOK_REMOVED, hook_name=self.name, layer=self._layer_idx)

    @property
    def name(self) -> str:
        return "mlp"


class ResidualHook(BaseHook):
    """Captures the residual stream output after a transformer block."""

    def __init__(self, store: list | None = None):
        self._store = store if store is not None else []
        self._hook_ref = None
        self._layer_idx = 0

    def attach(self, model: nn.Module, layer_idx: int) -> HookHandle:
        self._layer_idx = layer_idx
        block = model.transformer.h[layer_idx]

        def _hook(module, inputs, outputs):
            if not self.is_enabled:
                return
            # outputs[0] is the hidden state after the block
            self._store.append(outputs[0].detach().cpu())

        self._hook_ref = block.register_forward_hook(_hook)
        h = HookHandle(name=self.name, layer_idx=layer_idx, hook_type="residual")
        self.handle = h
        bus.emit(HOOK_REGISTERED, hook_name=self.name, layer=layer_idx)
        return h

    def detach(self) -> None:
        if self._hook_ref is not None:
            self._hook_ref.remove()
            self._hook_ref = None
            # Clear the handle too. `register_hook` branches on exactly this
            # (`hook.handle is None`) to decide whether to attach, so a hook
            # that has been detached but still reports a handle claims a live
            # registration carrying a dead id. Measured: after `remove_hook`,
            # every hook type still returned the old handle.
            self.handle = None
            bus.emit(HOOK_REMOVED, hook_name=self.name, layer=self._layer_idx)

    @property
    def name(self) -> str:
        return "residual"


class LogitHook(BaseHook):
    """Captures the final logits from the language model head."""

    def __init__(self, store: list | None = None):
        self._store = store if store is not None else []
        self._hook_ref = None
        self._layer_idx = 0

    def attach(self, model: nn.Module, layer_idx: int) -> HookHandle:
        # We attach to the final LM head (tied embeddings)
        self._layer_idx = layer_idx

        def _hook(module, inputs, outputs):
            if not self.is_enabled:
                return
            self._store.append(outputs.detach().cpu())

        self._hook_ref = model.lm_head.register_forward_hook(_hook)
        h = HookHandle(name=self.name, layer_idx=layer_idx, hook_type="logit")
        self.handle = h
        bus.emit(HOOK_REGISTERED, hook_name=self.name, layer=layer_idx)
        return h

    def detach(self) -> None:
        if self._hook_ref is not None:
            self._hook_ref.remove()
            self._hook_ref = None
            # Clear the handle too. `register_hook` branches on exactly this
            # (`hook.handle is None`) to decide whether to attach, so a hook
            # that has been detached but still reports a handle claims a live
            # registration carrying a dead id. Measured: after `remove_hook`,
            # every hook type still returned the old handle.
            self.handle = None
            bus.emit(HOOK_REMOVED, hook_name=self.name, layer=self._layer_idx)

    @property
    def name(self) -> str:
        return "logit"


class CustomHook(BaseHook):
    """User-defined hook with an arbitrary function.

    Usage::

        def my_hook(module, inputs, outputs):
            ...
            return None            # observe only

        def my_patcher(module, inputs, outputs):
            return new_output      # replace the output

        hook = CustomHook(my_hook, name="my_custom")
        manager.register(hook, layer_idx=5)

    The return value follows PyTorch's forward-hook contract: return a tensor
    and it replaces the module's output, for every later hook and for the
    caller; return None and the output passes through unchanged.

    The `store` argument remains available for hooks that want their results
    recorded, but it is no longer populated implicitly from the return value.
    Doing that made a patch indistinguishable from a capture.
    """

    def __init__(self, fn: Callable, name: str = "custom", store: list | None = None):
        self._fn = fn
        self._name = name
        self._store = store if store is not None else []
        self._hook_ref = None
        self._layer_idx = 0
        self._target_module = None

    def attach(self, model: nn.Module, layer_idx: int) -> HookHandle:
        self._layer_idx = layer_idx
        # Default: attach to the transformer block at layer_idx
        self._target_module = model.transformer.h[layer_idx]

        def _hook(module, inputs, outputs):
            if not self.is_enabled:
                return None
            result = self._fn(module, inputs, outputs)
            # PyTorch's forward-hook contract: returning a tensor replaces the
            # module's output, for every hook registered after this one and
            # for the caller.
            #
            # This used to swallow the return and append it to `_store`
            # instead. So a hook written to patch an output was captured into a
            # list while the forward pass carried on untouched -- the store
            # filled, the hook reported success, and the model returned the
            # value it would have returned anyway. Measured: with a replacer
            # and an observer registered in that order, the observer saw the
            # original output, not the replacement.
            #
            # Nothing caught it because the hook *appeared* to work. A user
            # reaching for the documented extension point to modify a forward
            # pass gets a silent no-op, which is the failure mode this project
            # exists to reject. Observe by appending to your own list and
            # returning None; replace by returning the new output.
            return result

        self._hook_ref = self._target_module.register_forward_hook(_hook)
        h = HookHandle(name=self._name, layer_idx=layer_idx, hook_type="custom")
        self.handle = h
        bus.emit(HOOK_REGISTERED, hook_name=self._name, layer=layer_idx)
        return h

    def detach(self) -> None:
        if self._hook_ref is not None:
            self._hook_ref.remove()
            self._hook_ref = None
            # As above: clearing the handle is what makes a detached hook
            # report itself as detached.
            self.handle = None
            bus.emit(HOOK_REMOVED, hook_name=self._name, layer=self._layer_idx)

    @property
    def name(self) -> str:
        return self._name


# ── Hook registry ─────────────────────────────────────────────────

HOOK_TYPES: dict[str, type[BaseHook]] = {
    "embedding": EmbeddingHook,
    "attention": AttentionHook,
    "mlp": MLPHook,
    "residual": ResidualHook,
    "logit": LogitHook,
    "custom": CustomHook,
}


def get_hook_class(hook_type: str) -> type[BaseHook]:
    cls = HOOK_TYPES.get(hook_type)
    if cls is None:
        raise HookError(f"Unknown hook type: {hook_type}. Available: {list(HOOK_TYPES.keys())}")
    return cls


def register_hook_type(name: str, cls: type[BaseHook]) -> None:
    """Allow external registration of new hook types (plugin API)."""
    HOOK_TYPES[name] = cls


# ── HookManager ──────────────────────────────────────────────────


class HookManager:
    """Manages all hooks across all layers.

    APIs:
        register_hook(layer_idx, hook_type_or_instance) -> HookHandle
        remove_hook(handle)
        enable_hook(handle)
        disable_hook(handle)
        list_hooks() -> list[HookHandle]
        clear()
    """

    def __init__(self):
        self._handles: dict[str, BaseHook] = {}

    def register_hook(
        self,
        layer_idx: int,
        hook_type: str | BaseHook,
        store: list | None = None,
        depends_on: list[str] | None = None,
        runs_after: list[str] | None = None,
    ) -> HookHandle:
        if isinstance(hook_type, str):
            cls = get_hook_class(hook_type)
            hook = cls(store=store)
        elif isinstance(hook_type, BaseHook):
            hook = hook_type
        else:
            raise HookError(f"Invalid hook: must be a type name or BaseHook instance")

        # Validate dependencies
        deps = depends_on or []
        after = runs_after or []
        self._validate_dependencies(deps, after)

        if not hasattr(hook, "_handle") or hook.handle is None:
            handle = hook.attach(self._model, layer_idx)
        else:
            handle = hook.handle
            if handle.id not in self._handles:
                handle = hook.attach(self._model, layer_idx)

        handle.depends_on = deps
        handle.runs_after = after
        self._handles[handle.id] = hook
        return handle

    def _validate_dependencies(self, depends_on: list[str], runs_after: list[str]) -> None:
        registered = {h.name for h in self.list_hooks()}
        for dep in depends_on:
            if dep not in registered and dep not in HOOK_TYPES:
                pass  # dependency may be registered later
        # Check for circular deps: not implemented in Sprint 2 (simple list is enough)

    def remove_hook(self, handle: HookHandle) -> None:
        hook = self._handles.pop(handle.id, None)
        if hook is not None:
            hook.detach()

    def enable_hook(self, handle: HookHandle) -> None:
        handle.enabled = True

    def disable_hook(self, handle: HookHandle) -> None:
        handle.enabled = False

    def list_hooks(self) -> list[HookHandle]:
        return [h.handle for h in self._handles.values() if h.handle is not None]

    def clear(self) -> None:
        for hook in list(self._handles.values()):
            hook.detach()
        self._handles.clear()

    def bind(self, model: nn.Module) -> None:
        """Attach HookManager to a model (called by ModelManager)."""
        self._model = model
