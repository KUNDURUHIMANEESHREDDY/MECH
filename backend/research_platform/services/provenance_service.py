"""Provenance Service -- re-exported.

This file used to be a byte-identical copy of
``backend/mech_platform/services/provenance_service.py``. Two copies of a
provenance service is not a convenience: a provenance rule that is fixed in one
and not the other is a rule that can be bypassed by importing the other, which
is precisely the "one canonical authority per concern" problem this codebase is
trying to close.

Rather than patch the same bug twice and leave two implementations to drift
again, this module delegates. The implementation lives in
``backend.mech_platform.services.provenance_service``.

Both package __init__ files are bare docstrings with no imports, so the
cross-package import cannot cycle.
"""

from __future__ import annotations

from backend.mech_platform.services.provenance_service import ProvenanceService

__all__ = ["ProvenanceService"]