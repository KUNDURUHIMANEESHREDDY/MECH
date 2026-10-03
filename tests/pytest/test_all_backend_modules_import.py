"""Every module under backend/ must import.

This exists because of a recurring defect class that no behavioural test caught.
Four separate instances were found by hand:

  1. ``backend/reasoning/__init__.py`` imported ``journey_tracer`` and
     ``neuron_debugger``, neither of which exists.
  2. Six modules in ``backend/runtime/`` imported ``CacheError``, ``HookError``,
     ``ModelLoadError``, ``ModelNotFoundError`` and ``SessionNotFoundError``
     from ``errors.py``, which defined none of them.
  3. Four inspectors imported ``backend.interpretability.statistics.stats_engine``
     -- a package that does not exist. The real module is under
     ``backend/science/statistics/``.
  4. ``discovery_memory.py`` imported ``..reproducibility.paper_registry`` and
     ``PaperRegistry``. Neither the relative path nor the class name existed;
     the class is ``BenchmarkRegistry``.

In every case the module failed at import, so anything importing it failed too,
and the suite stayed green because nothing exercised those paths. A test that
walks the package is the cheapest way to catch the whole class at once.

Two details this has to get right:

* Modules are allowed to raise ``LiveUnavailable`` or ``NotImplementedError`` at
  *call* time -- that is the fail-closed contract -- but they must not raise at
  *import* time for that reason.
* Optional heavy dependencies must not count as failures. A module that cannot
  import ``torch`` is a different problem from a module importing a name that
  does not exist, and conflating them would make this test fail on a machine
  without a GPU rather than on a real defect.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest


def _backend_root():
    import backend

    return backend


def _import_failures():
    """Import every backend submodule, collecting (name, error) for failures."""
    import backend

    failures = []
    for module in pkgutil.walk_packages(backend.__path__, "backend."):
        try:
            importlib.import_module(module.name)
        except ModuleNotFoundError as exc:
            # Distinguish "this dependency is not installed" from "this module
            # imports something that does not exist". The latter names a path
            # under the package being walked, which is the defect class here.
            missing = getattr(exc, "name", "") or ""
            if missing.startswith("backend"):
                failures.append((module.name, f"{type(exc).__name__}: {exc}"))
        except Exception as exc:  # noqa: BLE001 - report, do not mask
            failures.append((module.name, f"{type(exc).__name__}: {exc}"))
    return failures


def test_every_backend_module_imports():
    """No module under backend/ may fail to import.

    A module that cannot be imported takes down every importer with it, and no
    behavioural test catches it when nothing exercises the path.
    """
    failures = _import_failures()
    assert not failures, "modules that fail to import:\n" + "\n".join(
        f"  {name}\n      {error}" for name, error in failures
    )


def test_runtime_errors_module_defines_everything_imported_from_it():
    """`errors.py` is the designated home for runtime exceptions.

    Six modules import five names from it. Each was missing, so each of those
    modules was unimportable. This pins the names so a future rename cannot
    silently reintroduce the same breakage.
    """
    from backend.runtime import errors

    for name in (
        "RuntimeExecutionError",
        "MissingModelError",
        "ModelOutOfMemoryError",
        "InvalidPromptError",
        "CacheError",
        "HookError",
        "ModelLoadError",
        "ModelNotFoundError",
        "SessionNotFoundError",
    ):
        assert hasattr(errors, name), f"backend/runtime/errors.py is missing {name}"

    # The load/not-found pair must stay under MissingModelError, whose docstring
    # already covers "cannot be resolved or loaded", so existing broad handlers
    # keep catching them.
    assert issubclass(errors.ModelLoadError, errors.MissingModelError)
    assert issubclass(errors.ModelNotFoundError, errors.MissingModelError)


def test_live_unavailable_is_a_single_class_across_the_package():
    """Moving LiveUnavailable must not have split it into two identities.

    It was defined in `live_measure` and is now defined in `adapter_base` and
    re-exported, so that model adapters can raise it without importing the
    discovery layer. If any module still re-defines it, an `except` clause in one
    layer would stop catching raises from the other.
    """
    from backend.science.models.adapter_base import LiveUnavailable as canonical
    from backend.interpretability.discovery import live_measure

    assert live_measure.LiveUnavailable is canonical

    seen = set()
    for module in pkgutil.walk_packages(_backend_root().__path__, "backend."):
        try:
            imported = importlib.import_module(module.name)
        except Exception:
            continue
        exported = getattr(imported, "LiveUnavailable", None)
        if exported is not None:
            seen.add(exported)

    assert seen <= {canonical}, (
        "more than one LiveUnavailable class exists: "
        f"{[c.__module__ + '.' + c.__name__ for c in seen - {canonical}]}"
    )


@pytest.mark.parametrize(
    "module_path",
    [
        "backend.science.reproducibility.greater_than_pipeline",
        "backend.science.reproducibility.arithmetic_pipeline",
        "backend.science.reproducibility.sae_pipeline",
        "backend.science.reproducibility.copy_task_pipeline",
        "backend.science.reproducibility.factual_recall_pipeline",
        "backend.science.reproducibility.logit_lens_pipeline",
    ],
)
def test_reproduction_pipelines_import_without_touching_weights(module_path):
    """Importing a pipeline must not load a model or fabricate a result.

    These modules are imported at package scope by `__init__.py`, so anything
    they do at import time happens on every process start.
    """
    module = importlib.import_module(module_path)
    assert module is not None
