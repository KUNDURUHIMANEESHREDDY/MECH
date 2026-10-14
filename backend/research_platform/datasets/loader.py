"""Dataset Loader & Streaming Engine.

This file used to be a byte-identical copy of
``backend/mech_platform/datasets/loader.py``. Two copies of a dataset
loader is not a convenience: a dataset loading bug fixed in one and not
the other is a bug that can be triggered by importing the other.

Rather than patch the same logic twice and leave two implementations to drift
again, this module delegates. The implementation lives in
``backend.mech_platform.datasets.loader``.
"""

from __future__ import annotations

from backend.mech_platform.datasets.loader import DatasetLoader

__all__ = ["DatasetLoader"]