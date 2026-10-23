"""Lifecycle tests for temporary comparison-model locks."""

import gc
import weakref

from backend.runtime import comparison


class WeakReferenceableModel:
    pass


def test_repeated_calls_share_lock_while_model_is_alive():
    model = WeakReferenceableModel()

    assert comparison._model_lock(model) is comparison._model_lock(model)


def test_model_lock_is_removed_when_model_is_collected():
    model = WeakReferenceableModel()
    key = id(model)
    model_ref = weakref.ref(model)
    comparison._model_lock(model)

    assert key in comparison._MODEL_LOCKS

    del model
    gc.collect()

    assert model_ref() is None
    assert key not in comparison._MODEL_LOCKS


def test_non_weak_referenceable_object_fails_explicitly():
    try:
        comparison._model_lock(object())
    except TypeError as exc:
        assert "must support weak references" in str(exc)
    else:
        raise AssertionError("non-weak-referenceable model should be refused")
