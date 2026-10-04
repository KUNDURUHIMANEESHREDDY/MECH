"""Integrity primitives: real Ed25519 signing for dataset and manifest provenance.

`signing.py` holds the implementation. This package exists so that
`from backend.science.integrity import sign, verify` reads as the single place
signing happens, which is the point: the previous arrangement had two independent
`sha256(payload + ":" + key)` constructions in two modules, neither of which was a
signature.
"""

from __future__ import annotations

from .signing import (  # noqa: F401
    SIGNING_KEY_PATH_ENV,
    SignatureInvalid,
    SigningUnavailable,
    VerificationResult,
    load_private_key,
    sign,
    verify,
)

__all__ = [
    "SIGNING_KEY_PATH_ENV",
    "SignatureInvalid",
    "SigningUnavailable",
    "VerificationResult",
    "load_private_key",
    "sign",
    "verify",
]