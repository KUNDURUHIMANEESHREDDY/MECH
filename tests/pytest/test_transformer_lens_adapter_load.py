"""Issue #15: TransformerLensAdapter discarded the model it had just loaded.

The rule under test
------------------
**An adapter that loaded real weights must use them, and must say so.**
Two separate guarantees are asserted here:

1. The loaded model survives construction. ``ModelAdapter.__init__`` calls
   ``self._load_model()``, which assigns ``self._tl_model``. The adapter's
   own ``__init__`` then assigned ``self._tl_model = None`` *after* the base
   constructor returned, so every method took its "no model" branch and the
   adapter served mock fixtures even when real weights had loaded.

2. A degraded load is reported, not hidden. The old failure path set
   ``spec.mock_mode = True`` but left ``simulated`` False. The
   reproducibility pipelines refuse to report a measurement from a simulated
   adapter -- ``greater_than_pipeline`` and its siblings check
   ``adapter.simulated`` -- so a mock result could have been reported as a
   measurement. The same class also reset ``load_error`` to None on the line
   after the base constructor, erasing the reason for the degradation.

The defect is in the adapter's own bookkeeping, so these tests use a fake
TransformerLens rather than the real one: loading GPT-2 would cost minutes
per run and prove nothing extra. The fake carries a real (tiny) tensor, so
the fingerprint assertions exercise the real hashing path.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, List, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_PATH = ROOT / "backend" / "science" / "models" / "transformer_lens_adapter.py"


class _FakeParams:
    """Stands in for ``HookedTransformer.params``.

    ``hash_weights`` reads ``named_parameters()``; returning one real tensor
    makes the fingerprint a genuine content hash rather than a stub.
    """

    def __init__(self) -> None:
        import torch

        self._tensors: List[Tuple[str, Any]] = [
            ("transformer.wte.weight", torch.zeros(4, 3, dtype=torch.float32)),
        ]

    def named_parameters(self) -> List[Tuple[str, Any]]:
        return list(self._tensors)


class _FakeHookedTransformer:
    """The minimum surface ``TransformerLensAdapter`` touches."""

    def __init__(self, params: _FakeParams) -> None:
        self.params = params


@pytest.fixture(autouse=True)
def _restore_identity_providers():
    """Keep this module's adapters out of the process-wide model identity.

    ``ModelAdapter._publish_model_identity`` registers a provider that other
    tests read through ``current_model_identity()``. Left registered, a fake
    adapter's identity would look like the real model's to the next test.
    """
    from backend.runtime import activation_cache

    saved = list(activation_cache._IDENTITY_PROVIDERS)
    yield
    activation_cache._IDENTITY_PROVIDERS[:] = saved


def _install_fake(monkeypatch: pytest.MonkeyPatch, outcome: str) -> _FakeParams:
    """Point the adapter's TransformerLens resolution at a fake."""
    from backend.interpretability import tl_compat

    params = _FakeParams()

    class _Factory:
        @staticmethod
        def from_pretrained(repo_id: str, *args: Any, **kwargs: Any):
            if outcome == "raise":
                raise OSError(f"no weights for {repo_id}")
            return _FakeHookedTransformer(params)

    monkeypatch.setattr(tl_compat, "resolve_hooked_transformer", lambda: _Factory)
    return params


def test_the_loaded_model_survives_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """The weights the adapter loaded must still be there afterwards."""
    from backend.science.models.transformer_lens_adapter import TransformerLensAdapter

    params = _install_fake(monkeypatch, "ok")
    adapter = TransformerLensAdapter(model_name="gpt2-small")

    assert adapter._tl_model is not None, (
        "the adapter threw away the model it loaded; every method below this "
        "point then served mock fixtures instead of model output"
    )
    assert adapter._tl_model.params is params
    assert adapter.load_error is None


def test_the_loaded_weights_are_fingerprinted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A loaded adapter must carry a real fingerprint, not a vacuous None.

    The model-identity seal added when results became attestation-backed was
    empty for this adapter: it reported "no weights held" about a state the
    adapter should never have been in.
    """
    from backend.science.models.transformer_lens_adapter import TransformerLensAdapter

    _install_fake(monkeypatch, "ok")
    adapter = TransformerLensAdapter(model_name="gpt2-small")

    fingerprint = adapter.model_fingerprint()
    assert fingerprint is not None, (
        "an adapter holding real weights must fingerprint them; None means the "
        "result cannot be bound to the model that produced it"
    )
    assert fingerprint.startswith("sha256:")

    identity = adapter.model_identity()
    assert identity is not None
    assert identity["model_fingerprint"] == fingerprint


def test_a_failed_load_is_reported_as_simulated(monkeypatch: pytest.MonkeyPatch) -> None:
    """A load that fails must be visible as a degradation, with its reason."""
    from backend.science.models.transformer_lens_adapter import TransformerLensAdapter

    _install_fake(monkeypatch, "raise")
    adapter = TransformerLensAdapter(model_name="gpt2-small")

    assert adapter.spec.mock_mode is True
    assert adapter.simulated is True, (
        "a degraded load that leaves `simulated` False lets the reproducibility "
        "pipelines report mock fixtures as a measurement"
    )
    assert adapter.simulation_reason, "a simulated adapter must record why"
    assert "gpt2-small" in adapter.simulation_reason
    assert adapter.load_error, "the reason for the failed load must survive"
    assert "OSError" in adapter.load_error


def test_a_failed_load_never_fingerprints_a_fiction(monkeypatch: pytest.MonkeyPatch) -> None:
    """No weights means no fingerprint -- not a hash of a mock."""
    from backend.science.models.transformer_lens_adapter import TransformerLensAdapter

    _install_fake(monkeypatch, "raise")
    adapter = TransformerLensAdapter(model_name="gpt2-small")

    assert adapter.model_fingerprint() is None
    assert adapter.model_identity() is None


def test_the_adapter_does_not_clear_the_model_after_loading_it() -> None:
    """Structural guard against the ordering defect coming back.

    The bug was an ordering mistake, and a behavioural test only catches it
    while a fake happens to exercise the path. This asserts the invariant
    directly: no assignment to ``_tl_model`` or ``load_error`` may follow
    the ``super().__init__`` call inside ``TransformerLensAdapter.__init__``.
    """
    tree = ast.parse(ADAPTER_PATH.read_text(encoding="utf-8"))

    init = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "__init__"
        and any(
            isinstance(child, ast.Attribute) and child.attr == "_tl_model"
            for child in ast.walk(node)
        )
    )

    super_call_index = next(
        index
        for index, statement in enumerate(init.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Attribute)
        and statement.value.func.attr == "__init__"
    )

    offenders = []
    for statement in init.body[super_call_index + 1:]:
        for node in ast.walk(statement):
            if (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Attribute)
                    and target.attr in ("_tl_model", "load_error")
                    for target in node.targets
                )
            ):
                offenders.append(node.lineno)

    assert not offenders, (
        "TransformerLensAdapter.__init__ assigns to the model it loaded after "
        f"the base constructor returned (line(s) {offenders}); the base "
        "constructor is what loads the model, so these assignments discard it"
    )
