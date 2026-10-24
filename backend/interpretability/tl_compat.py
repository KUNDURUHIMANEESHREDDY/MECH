"""Resolve TransformerLens' ``HookedTransformer`` across API generations.

Why this module exists
----------------------
TransformerLens 4.0 **removed** ``HookedTransformer``. MECH's TransformerLens
code paths depend on it (and on its hook-name conventions), so the supported
stack uses the compatible TransformerLens 3.x line alongside patched
Transformers 5.x rather than jumping directly to 4.x. A 4.x install used to fail
deep inside the stack with:

    AttributeError: 'HookedTransformer' was removed in TransformerLens 4.0.

That message points at the symptom, not the fix. Importing ``HookedTransformer``
through this module instead turns the failure into an actionable error that
names the supported version range and the exact install command.

The former TransformerLens 2.x pin conflicted with patched Transformers 5.x,
while TransformerLens 4.x no longer exported the API MECH uses. The supported
intersection is TransformerLens 3.9.x (which retains ``HookedTransformer`` and
requires Transformers >=5.9.0), Transformers >=5.20.2, and huggingface-hub
>=1.31.0. These ranges are defined together in ``requirements.txt``. This
module makes an unsupported API version obvious instead of mysterious.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

#: transformer-lens releases that still export ``HookedTransformer``.
SUPPORTED_TL_RANGE = ">=3.9.0,<4.0"


class TransformerLensUnavailable(ImportError):
    """Base class for a TransformerLens stack MECH cannot use."""


class TransformerLensVersionError(TransformerLensUnavailable):
    """The installed transformer-lens release does not export HookedTransformer.

    Raised for transformer-lens 4.x and later. Fixable by changing versions.
    """


class TransformerLensEnvironmentError(TransformerLensUnavailable):
    """transformer-lens is present but could not be imported at all.

    A broken or incomplete environment rather than a version problem. This was
    once routinely caused by MECH's own ``backend/datasets/`` package shadowing
    HuggingFace ``datasets``; that package is now ``backend/research_datasets/``
    and the shadowing is fixed.
    """

_INSTALL_HINT = (
    "MECH's TransformerLens code paths require transformer-lens "
    f"{SUPPORTED_TL_RANGE}, transformers >=5.20.2,<6.0, and "
    "huggingface-hub >=1.31.0,<3.0.\n"
    "  Install the pinned set:\n"
    "      pip install -r requirements.txt\n"
    "  Or repair an existing environment:\n"
    "      pip install 'transformer-lens>=3.9.0,<4.0' "
    "'transformers>=5.20.2,<6.0' 'huggingface-hub>=1.31.0,<3.0'\n"
    "TransformerLens 4.x removed HookedTransformer in favour of "
    "TransformerBridge; the <4.0 bound is intentional until that API migration "
    "is completed and validated."
)


def installed_version(package: str) -> str | None:
    """Return the installed version of ``package``, or None if it is absent."""
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def library_report() -> dict[str, Any]:
    """Version info for diagnostics and for the drift test."""
    return {
        "transformer_lens": installed_version("transformer-lens"),
        "transformers": installed_version("transformers"),
        "torch": installed_version("torch"),
        "supported_range": SUPPORTED_TL_RANGE,
    }


def resolve_hooked_transformer() -> Any:
    """Return the ``HookedTransformer`` class, or raise a clear ImportError.

    Raises
    ------
    ImportError
        If transformer-lens is missing, or is a release that no longer exports
        ``HookedTransformer`` (4.x and later).
    """
    try:
        import transformer_lens
    except ImportError as exc:
        # transformer-lens itself could not be imported. This is a broken
        # environment rather than a version mismatch, and the message matters.
        # Note: MECH's dataset package used to be named `backend/datasets/`,
        # which shadowed HuggingFace `datasets` whenever `backend/` was on
        # sys.path and surfaced here as "No module named 'datasets.arrow_dataset'".
        # That package is now `backend/research_datasets/`, so this specific
        # cause is fixed; a missing `datasets` now means the real dependency is
        # absent.
        raise TransformerLensEnvironmentError(
            "transformer-lens is installed but could not be imported "
            f"({type(exc).__name__}: {exc}).\n"
            "This is an environment problem, not a version mismatch. A missing "
            "'datasets' module here means HuggingFace `datasets` is not "
            "installed -- it is no longer a MECH package-name collision.\n"
            f"{_INSTALL_HINT}"
        ) from exc

    try:
        from transformer_lens import HookedTransformer
    except (ImportError, AttributeError) as exc:
        # The module imports fine but no longer exports the symbol. This is the
        # transformer-lens 4.x case: `from X import Y` raises ImportError
        # ("cannot import name") or, via module __getattr__, AttributeError.
        found = installed_version("transformer-lens") or "unknown"
        raise TransformerLensVersionError(
            f"transformer-lens {found} does not provide HookedTransformer "
            f"({type(exc).__name__}: {exc}).\n{_INSTALL_HINT}"
        ) from exc

    return HookedTransformer


def check_compatibility() -> dict[str, Any]:
    """Report whether the installed stack can serve the TransformerLens paths.

    Returns a report rather than raising, so callers such as
    ``backend/validation/hardware_rigor.py`` and the Startup probe can log the
    mismatch without crashing the backend.

    The live GPT-2 path in ``backend/services/gpt2_engine.py`` uses plain
    ``transformers`` and is unaffected by a TransformerLens mismatch, so this
    must never be used to gate overall readiness.
    """
    report = library_report()
    try:
        resolve_hooked_transformer()
    except TransformerLensVersionError as exc:
        report["compatible"] = False
        report["reason"] = "version"
        report["detail"] = str(exc)
    except TransformerLensUnavailable as exc:
        report["compatible"] = False
        report["reason"] = "environment"
        report["detail"] = str(exc)
    else:
        report["compatible"] = True
        report["detail"] = "HookedTransformer available"
    return report
