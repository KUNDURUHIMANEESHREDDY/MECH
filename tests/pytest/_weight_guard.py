"""Shared live-weight availability probe for the pytest suite.

Problem this fixes: every weight-dependent suite defined its own ``_has_gpt2``
/ ``_needs_weights`` helper shaped like::

    try:
        ...construct adapter with mock_mode=False...
        return adapter._model is not None
    except Exception:
        return False

``ModelAdapter._load_model`` swallows *every* exception into a mock-mode
fallback, so that probe classifies an initialization or import regression
(broken ``ModelManager``, bad config, ``AttributeError`` from a refactor) as
"weights unavailable" and the suite *skips* instead of *failing*. A red tree
reads as green.

This module probes once and classifies the outcome:

* live weights present  -> ``weights_available()`` returns True.
* weights genuinely absent (offline cache miss, no network, gated repo
  without credentials) -> returns False, so callers skip; but when
  ``MECH_REQUIRE_WEIGHTS=1`` it raises instead, so the dedicated live-model
  CI lane fails rather than silently skipping.
* any code/import regression -> re-raised, never converted to a skip.

No model names, ports, tokens, or expected measurements are hardcoded here.
The repo id is read off the adapter spec that was actually constructed, and
the failure classification is by exception type and message, not by literal
return values.
"""

from __future__ import annotations

import functools
import os
from typing import Optional

#: Exceptions that mean "the code is broken", never "weights are absent".
#: These propagate instead of becoming a skip.
_CODE_REGRESSION_TYPES = (
    ImportError,
    SyntaxError,
    AttributeError,
    TypeError,
    NameError,
)

#: Message fragments (lowercased) that indicate genuinely missing weights
#: rather than broken code: offline cache misses, no network, gated repos.
_MISSING_WEIGHT_MARKERS = (
    "localentrynotfounderror",
    "local_files_only",
    "offline",
    "offline mode",
    "not found in cache",
    "no such file",
    "connection",
    "connecttimeout",
    "readtimeout",
    "max retries",
    "401",
    "403",
    "404",
    "gated",
    "unauthorized",
    "must be installed to load real models",
    "torch/transformers not installed",
    "no model loaded",
)


def _variant() -> str:
    return os.environ.get("MECH_GPT2_VARIANT", "small")


def _weights_required() -> bool:
    return os.environ.get("MECH_REQUIRE_WEIGHTS", "0") == "1"


def _looks_like_missing_weights(exc: BaseException) -> bool:
    if isinstance(exc, (OSError, ConnectionError, FileNotFoundError)):
        return True
    name = type(exc).__name__.lower()
    if "localentry" in name or "offline" in name or "connection" in name:
        return True
    msg = str(exc).lower()
    return any(marker in msg for marker in _MISSING_WEIGHT_MARKERS)


class WeightsRequiredError(RuntimeError):
    """Raised when weights are required but unavailable."""


@functools.lru_cache(maxsize=1)
def _probe() -> tuple[bool, Optional[str]]:
    """Run once per session: (available, reason_if_unavailable).

    Raises whatever the underlying code raised when it is a code regression,
    so a broken import can never become a skip.
    """
    try:
        from backend.science.models.gpt2_adapter import GPT2Adapter
    except _CODE_REGRESSION_TYPES:
        raise
    except Exception as exc:
        if _looks_like_missing_weights(exc):
            return False, f"{type(exc).__name__}: {exc}"
        raise

    try:
        adapter = GPT2Adapter(variant=_variant(), mock_mode=False)
    except _CODE_REGRESSION_TYPES:
        raise
    except Exception as exc:
        if _looks_like_missing_weights(exc):
            return False, f"{type(exc).__name__}: {exc}"
        raise

    model = getattr(adapter, "_model", None)
    spec = getattr(adapter, "spec", None)
    mocked = bool(getattr(spec, "mock_mode", False))
    if model is not None and not mocked:
        return True, None

    # Fell back to mock mode: ask the manager directly so the *reason* is
    # recorded instead of guessed. The repo id comes from the spec that was
    # actually built, not from a literal.
    repo_id = getattr(spec, "hf_repo_id", None) or "gpt2"
    try:
        from backend.science.models.model_manager import ModelManager

        ModelManager().get_model_and_tokenizer(repo_id)
        # Manager can load but the adapter did not: that is a wiring bug.
        raise RuntimeError(
            f"ModelManager loaded {repo_id!r} but GPT2Adapter fell back to "
            "mock mode; refusing to report this as missing weights."
        )
    except _CODE_REGRESSION_TYPES:
        raise
    except RuntimeError:
        raise
    except Exception as exc:
        if _looks_like_missing_weights(exc):
            return False, f"{type(exc).__name__}: {exc}"
        raise
    return False, "adapter fell back to mock mode for an unclassified reason"


def weights_available() -> bool:
    """True when live GPT-2 weights are loaded in this process."""
    available, _ = _probe()
    return available


def skip_reason() -> str:
    """Human-readable reason used for skip messages; never a measurement."""
    _, reason = _probe()
    return reason or "GPT-2 weights are not loaded"


def require_weights() -> None:
    """Fail closed when weights are required but unavailable.

    Honor ``MECH_REQUIRE_WEIGHTS=1``: the live-model CI lane sets it so an
    environment without weights fails instead of skipping every live test.
    Local development without weights still skips with the recorded reason.
    """
    available, reason = _probe()
    if available:
        return
    if _weights_required():
        raise WeightsRequiredError(
            f"MECH_REQUIRE_WEIGHTS=1 but live weights are unavailable: {reason}"
        )
    import pytest

    pytest.skip(reason or "GPT-2 weights are not loaded")
