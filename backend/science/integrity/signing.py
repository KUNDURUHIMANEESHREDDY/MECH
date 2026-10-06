"""Ed25519 signing and verification for dataset and manifest integrity.

Replaces a construction that was documented as "asymmetric digital signature" and
implemented as ``sha256(payload + ":" + key)``, verified by recomputing the same
hash with a hardcoded ``"mock_private_key"``.

Why that was not a signature
----------------------------
A keyed hash is a message authentication code: proving it requires the secret. The
verify function did not take a public key -- it accepted a ``public_key``
argument and ignored it, recomputing with the literal secret from the source. So
anyone who could read the repository could produce a signature that verified,
which is the exact property a signature exists to prevent. There was also no key
pair, no asymmetric operation, and the "private key" was a default argument.

What this module does instead
-----------------------------
Real Ed25519, via ``cryptography``. The verification key is the actual public
key, so nothing secret is needed to check a signature, and no secret is present
in the source to find.

Key supply
----------
No key is ever defaulted. ``load_private_key`` accepts, in order of preference:

  * an explicit 32-byte seed, 64-byte private key, or PEM bytes
  * a path to a PEM or raw-seed key file
  * the path named by ``MECH_SIGNING_KEY_PATH``

With none of those it raises. That is deliberate: a signature scheme whose default
key is in the repository authenticates nothing, and a default would let a caller
sign without noticing it had signed with a public secret.

Key material is not written by this module and must not be committed. A key file
belongs outside the tree, referenced by path or environment.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Optional, Union

#: Environment variable naming a private key file. The only supported way to
#: supply a key without threading it through every call site.
SIGNING_KEY_PATH_ENV = "MECH_SIGNING_KEY_PATH"

KeyMaterial = Union[str, bytes, os.PathLike]


class SigningUnavailable(RuntimeError):
    """No usable Ed25519 key was supplied, or the library is missing."""


class SignatureInvalid(ValueError):
    """A signature did not verify under the supplied public key."""


@dataclass(frozen=True)
class VerificationResult:
    """The outcome of a verification attempt, with the reason attached.

    A bare bool cannot distinguish "no signature was recorded", "no key was
    supplied" and "the signature is wrong". Those are three different problems
    with three different fixes, and a caller that cannot tell them apart will
    eventually treat a missing signature as a valid one.
    """

    valid: bool
    reason: Optional[str] = None
    algorithm: str = "Ed25519"

    def __bool__(self) -> bool:
        return self.valid

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "algorithm": self.algorithm,
        }


def _ed25519():
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
    except Exception as exc:  # pragma: no cover - dependency missing
        raise SigningUnavailable(
            "Ed25519 signing requires the 'cryptography' package, which is not "
            f"importable: {type(exc).__name__}: {exc}"
        ) from exc
    return Ed25519PrivateKey, Ed25519PublicKey


def _looks_like_pem(data: bytes) -> bool:
    return b"BEGIN" in data[:64]


def _is_hex(text: str) -> bool:
    if len(text) % 2:
        return False
    return all(c in "0123456789abcdefABCDEF" for c in text)


def load_private_key(material: Optional[KeyMaterial] = None):
    """Resolve an Ed25519 private key. Never falls back to a default.

    Accepts a 32-byte seed, a 64-byte expanded private key, PEM bytes, or a path
    to either. When `material` is None, `MECH_SIGNING_KEY_PATH` is consulted.

    Raises `SigningUnavailable` rather than substituting a key, because a
    default signing key is worse than no signing: it produces signatures that
    verify and attest to nothing.
    """
    Ed25519PrivateKey, _ = _ed25519()

    raw: Optional[bytes] = None
    source = "argument"

    if material is None:
        path = os.environ.get(SIGNING_KEY_PATH_ENV)
        if not path:
            raise SigningUnavailable(
                "No signing key supplied. Pass one explicitly, or set "
                f"{SIGNING_KEY_PATH_ENV} to a key file kept outside the "
                "repository. There is deliberately no default key: a signature "
                "made with a key that ships in the source attests to nothing."
            )
        material = path
        source = os.environ[SIGNING_KEY_PATH_ENV]

    try:
        if isinstance(material, (bytes, bytearray)):
            raw = bytes(material)
        elif isinstance(material, os.PathLike) or (
            isinstance(material, str) and os.path.exists(material)
        ):
            with open(material, "rb") as handle:
                raw = handle.read()
        elif isinstance(material, str):
            # A str that is not an existing path is treated as PEM text or a
            # hex-encoded seed, so a key can be passed inline without ambiguity.
            raw = material.encode("utf-8")
        else:
            raise SigningUnavailable(
                f"Unsupported key material of type {type(material).__name__}"
            )
    except OSError as exc:
        raise SigningUnavailable(
            f"Could not read signing key from {source}: {exc}"
        ) from exc

    try:
        if _looks_like_pem(raw):
            from cryptography.hazmat.primitives import serialization

            key = serialization.load_pem_private_key(raw, password=None)
            if not isinstance(key, Ed25519PrivateKey):
                raise SigningUnavailable(
                    f"Key at {source} is a {type(key).__name__}, not an Ed25519 key"
                )
            return key

        #  Two ambiguities have to be resolved, and both are security-relevant
        #  because resolving them the wrong way signs with a key the caller did
        #  not supply.
        #
        #  1. A 64-character hex seed is ALSO 64 bytes. Deciding by length alone
        #     hands the ASCII characters to `from_private_bytes`, yielding a
        #     valid signature under a completely different identity.
        #  2. A raw 32-byte seed is binary, and `strip()` removes bytes
        #     0x09-0x0d and 0x20 from either end -- so a seed whose first or last
        #     byte was 0x20 arrived as 31 bytes and was rejected as malformed.
        #     That is a ~4.7% failure rate per random key, which is the kind of
        #     intermittent bug that gets blamed on something else.
        #
        #  Resolution: an all-hex text of 64 or 128 characters is treated as hex,
        #  because a genuine 32- or 64-byte *binary* seed made only of hex
        #  characters has probability (16/256)^32 -- effectively zero. Exact
        #  binary lengths are accepted without stripping.
        stripped = raw.strip()
        hex_text = stripped.decode("ascii", errors="ignore")
        if hex_text and _is_hex(hex_text) and len(hex_text) in (64, 128):
            decoded = bytes.fromhex(hex_text)
            if len(decoded) == 64:
                # 128 hex chars = seed || public-seed; only the seed is private.
                decoded = decoded[:32]
            if len(decoded) == 32:
                return Ed25519PrivateKey.from_private_bytes(decoded)
            raise SigningUnavailable(
                f"Hex key at {source} decoded to {len(decoded)} bytes, expected 32"
            )

        if len(raw) == 32:
            return Ed25519PrivateKey.from_private_bytes(raw)
        if len(raw) == 64:
            return Ed25519PrivateKey.from_private_bytes(raw[:32])
        if len(stripped) == 32:
            return Ed25519PrivateKey.from_private_bytes(stripped)
        if len(stripped) == 64:
            return Ed25519PrivateKey.from_private_bytes(stripped[:32])

        raise SigningUnavailable(
            f"Key at {source} is neither PEM, a 32-byte seed, a 64-byte private "
            f"key, nor a hex seed; got {len(raw)} bytes"
        )
    except SigningUnavailable:
        raise
    except Exception as exc:
        raise SigningUnavailable(
            f"Could not load an Ed25519 private key from {source}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def sign(payload: Union[str, bytes], private_key: Optional[KeyMaterial] = None) -> str:
    """Return a hex Ed25519 signature over `payload`.

    Raises `SigningUnavailable` when no key can be resolved. There is no
    unsigned fallback: a caller that wants no signature should not call this.
    """
    key = load_private_key(private_key)
    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    signature = key.sign(data)
    return signature.hex()


def verify(payload: Union[str, bytes], signature: Optional[str],
           public_key: KeyMaterial) -> VerificationResult:
    """Verify an Ed25519 signature using the public key alone.

    Returns a `VerificationResult` rather than a bool so that "no signature",
    "no key" and "wrong signature" stay distinguishable. A missing or empty
    signature is never treated as valid.
    """
    Ed25519PrivateKey, Ed25519PublicKey = _ed25519()

    if not signature:
        return VerificationResult(False, "no signature is recorded")
    if not public_key:
        return VerificationResult(False, "no public key was supplied")

    try:
        if isinstance(public_key, (bytes, bytearray)):
            raw = bytes(public_key)
        elif isinstance(public_key, os.PathLike) or (
            isinstance(public_key, str) and os.path.exists(public_key)
        ):
            with open(public_key, "rb") as handle:
                raw = handle.read()
        else:
            raw = str(public_key).encode("utf-8")
    except OSError as exc:
        return VerificationResult(False, f"could not read public key: {exc}")

    try:
        if _looks_like_pem(raw):
            from cryptography.hazmat.primitives import serialization

            key = serialization.load_pem_public_key(raw)
            if not isinstance(key, Ed25519PublicKey):
                return VerificationResult(
                    False, f"key is a {type(key).__name__}, not Ed25519")
        elif len(raw) == 32:
            # Length checked before stripping: a raw 32-byte public key is
            # binary, and `strip()` would silently shorten it if either end byte
            # happened to be whitespace. See `load_private_key`.
            key = Ed25519PublicKey.from_public_bytes(raw)
        else:
            # Hex before the fixed-length forms, for the same reason as in
            # `load_private_key`: a 64-character hex public key is also 64 bytes.
            stripped = raw.strip()
            hex_text = stripped.decode("ascii", errors="ignore")
            if hex_text and _is_hex(hex_text) and len(hex_text) in (64, 128):
                decoded = bytes.fromhex(hex_text)
                if len(decoded) == 64:
                    decoded = decoded[:32]
                if len(decoded) != 32:
                    return VerificationResult(
                        False,
                        f"public key decoded to {len(decoded)} bytes, expected 32")
                key = Ed25519PublicKey.from_public_bytes(decoded)
            elif len(stripped) == 32:
                key = Ed25519PublicKey.from_public_bytes(stripped)
            else:
                return VerificationResult(
                    False,
                    f"public key is neither PEM, 32 raw bytes, nor hex; "
                    f"got {len(raw)} bytes")
    except Exception as exc:
        return VerificationResult(
            False, f"could not load an Ed25519 public key: {type(exc).__name__}")

    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    try:
        key.verify(bytes.fromhex(signature), data)
    except Exception:
        return VerificationResult(
            False, "signature does not verify under the supplied public key")
    return VerificationResult(True, None)


def public_key_hex(private_key: Optional[KeyMaterial] = None) -> str:
    """Return the hex public key matching `private_key`.

    Needed wherever a signature travels away from the process that made it. A
    signature is only checkable by a party that was *not* the signer if the
    verifying key travels with it, so any artifact that embeds a signature has
    to embed the key too.
    """
    key = load_private_key(private_key)
    from cryptography.hazmat.primitives import serialization

    raw = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return raw.hex()


def key_id_for(public_key: Union[str, bytes]) -> str:
    """A short, stable identifier for a verifying key.

    An artifact that carries both a signature and its key needs a way to state
    *which* key was used. This fingerprint does that, and comparing it against
    the carried key catches a swapped or substituted key -- the check that
    shape-only validation could never perform, because a substituted key
    produces a signature that is internally perfectly consistent.

    It also lets a verifier that knows which executor it trusts pin the
    expected `key_id` instead of trusting whichever key arrived.
    """
    if isinstance(public_key, (bytes, bytearray)):
        raw = bytes(public_key)
    else:
        text = str(public_key or "").strip()
        # Hex before utf-8: a 64-character hex public key is also 64 bytes,
        # for the same reason `load_private_key` checks hex first.
        raw = bytes.fromhex(text) if (_is_hex(text) and len(text) == 64) \
            else text.encode("utf-8")
    return "ed25519:" + hashlib.sha256(raw).hexdigest()[:16]