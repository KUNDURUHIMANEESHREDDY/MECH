"""Run-level attestation: bind a result to the run that produced it.

`ModelFingerprintEngine` proves *which weights* ran. It says nothing about
*which data* they ran on, or *which code* ran them, so two results with
byte-identical weights can come from different experiments entirely. The
roadmap asks for the whole chain -- ``run_id`` + ``dataset_sha256`` +
``code_revision`` alongside the weights -- and this module is that chain.

The rules it follows
--------------------

**The digest is of the data, not of the argument.** :func:`dataset_digest`
hashes the bytes of the dataset or panel it is handed: a file's contents, a
tensor's buffer, a list of records canonicalised to JSON. It never hashes a
caller-supplied digest string, because a digest of a digest attests nothing.
:func:`verify_run_attestation` recomputes the digest *from the data* and
compares it with the recorded one, so the check cannot be satisfied by
re-stating the claim.

**The code revision comes from the repository.** :func:`code_revision` asks
git for ``HEAD`` and records whether the working tree is dirty. When git is
missing, or the checkout is not a repository, or the repository has no
commits, there is no honest fallback: the revision is reported as
unavailable and the chain is left unattested. A made-up forty-character
string would be worse than nothing, because it would look like a revision to
every reader and to git itself.

**Failure is a report, not a value.** Anything that cannot be computed from
real inputs leaves ``attested`` False with a reason naming the cause. An
unattested chain is not evidence of anything, and the admission gate refuses
to promote a result that carries one.

Nothing here imports torch, transformers or numpy: the tensor and array
cases are duck-typed, so importing this module stays cheap and the backend
module walk does not pay for a GPU stack it may not have.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

#: Read size for file and tensor hashing.
_CHUNK = 1 << 20  # 1 MiB

#: Prefix every digest in this module carries, so a consumer can tell a
#: SHA-256 from an eight-character SipHash wearing its name.
DIGEST_PREFIX = "sha256:"

#: Hexadecimal digits, lower case only: an upper-case digest is a different
#: string, and comparing two spellings of one digest is how drift checks
#: start reporting differences that are not there.
_HEX_DIGITS = frozenset("0123456789abcdef")

#: Lengths git may use for an object id (SHA-1 and SHA-256 repositories).
_REVISION_LENGTHS = (40, 64)

#: Substrings that mark a digest as a placeholder rather than a measurement.
#: The deleted evidence boundary rejected these, and so does this: a digest
#: that says "mock" describes no bytes at all.
_PLACEHOLDERS = ("...", "mock", "unattested", "placeholder", "dummy", "fake")

#: How long a git call may take before it is treated as unavailable.
_GIT_TIMEOUT = 15


class EmptyDatasetError(ValueError):
    """The dataset handed in carries no bytes, so there is nothing to attest.

    An empty panel is not a dataset that happens to hash to the digest of the
    empty string. It is the absence of data, and it must not be presented as
    a measurement of nothing.
    """


def is_real_digest(value: Any) -> bool:
    """Whether `value` is a SHA-256 digest rather than a placeholder.

    Accepts the ``sha256:`` prefix this repository uses, and rejects the
    placeholders the old boundary module used to ship -- anything containing
    ``...`` or ``mock``, and the literal ``unattested``.
    """
    if not isinstance(value, str):
        return False
    text = value.strip().lower()
    if text.startswith(DIGEST_PREFIX):
        text = text[len(DIGEST_PREFIX):]
    if len(text) != 64:
        return False
    if not set(text) <= _HEX_DIGITS:
        return False
    lowered = value.lower()
    return not any(marker in lowered for marker in _PLACEHOLDERS)


def is_real_revision(value: Any) -> bool:
    """Whether `value` looks like a git object id."""
    if not isinstance(value, str):
        return False
    text = value.strip().lower()
    if len(text) not in _REVISION_LENGTHS:
        return False
    return set(text) <= _HEX_DIGITS


def _canonical_json(data: Any) -> str:
    """Canonical JSON: sorted keys, no incidental whitespace."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def _file_chunks(path: Any) -> Iterator[bytes]:
    """The bytes of a file, read in chunks."""
    target = Path(os.fspath(path))
    if not target.is_file():
        raise ValueError(f"the dataset file does not exist: {target}")
    with target.open("rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            yield chunk


def _dataset_chunks(data: Any) -> Iterator[bytes]:
    """Yield the canonical byte encoding of `data`.

    The encoding is total and deterministic: the same data always produces
    the same bytes, and different data produces different bytes. A list is
    length-prefixed per item, so ``["a", "bc"]`` and ``["ab", "c"]`` cannot
    collide.
    """
    if data is None:
        raise EmptyDatasetError("no dataset was supplied to hash")

    if isinstance(data, (bytes, bytearray, memoryview)):
        yield bytes(data)
        return

    if isinstance(data, str):
        yield data.encode("utf-8")
        return

    if isinstance(data, os.PathLike):
        yield from _file_chunks(data)
        return

    # A torch tensor, before a numpy array: duck-typed so this module never
    # imports torch. bfloat16 has no numpy equivalent, so the conversion can
    # fail -- and a failure here is a refusal, not a zero.
    detach = getattr(data, "detach", None)
    if callable(detach):
        try:
            tensor = detach().cpu().contiguous()
            yield f"{tensor.dtype}{tuple(tensor.shape)}".encode("utf-8")
            yield bytes(tensor.numpy().tobytes())
        except Exception as exc:
            raise ValueError(
                f"the tensor could not be read for hashing: {exc}") from exc
        return

    tobytes = getattr(data, "tobytes", None)
    if callable(tobytes) and hasattr(data, "dtype"):
        yield f"{data.dtype}{getattr(data, 'shape', ())}".encode("utf-8")
        yield bytes(tobytes())
        return

    if isinstance(data, Mapping):
        if not data:
            raise EmptyDatasetError(
                "the dataset supplied is an empty mapping, so there are no "
                "bytes to attest")
        yield _canonical_json(data).encode("utf-8")
        return

    if isinstance(data, (list, tuple)):
        if not data:
            raise EmptyDatasetError(
                "the dataset supplied is an empty panel, so there are no "
                "bytes to attest")
        for item in data:
            chunk = b"".join(_dataset_chunks(item))
            yield len(chunk).to_bytes(8, "big") + chunk
        return

    yield _canonical_json(data).encode("utf-8")


def dataset_digest(data: Any) -> str:
    """SHA-256 of the dataset or panel actually supplied.

    The digest is of the data, never of a digest: handing this function the
    string ``"sha256:abc..."`` hashes those characters, which says nothing
    about any dataset. Verification therefore recomputes the digest from the
    data and compares, rather than re-stating the claim.

    Raises :class:`EmptyDatasetError` when the data carries no bytes. An
    empty panel is the absence of a measurement, not a measurement of
    nothing, and it must not become a plausible-looking digest.
    """
    digest = hashlib.sha256()
    total = 0
    for chunk in _dataset_chunks(data):
        total += len(chunk)
        digest.update(chunk)
    if total == 0:
        raise EmptyDatasetError(
            "the dataset supplied carries no bytes, so there is nothing to "
            "attest")
    return DIGEST_PREFIX + digest.hexdigest()


@dataclass(frozen=True)
class CodeRevision:
    """What the repository says about itself.

    ``revision`` is empty when the repository could not be asked. There is no
    placeholder value: an invented object id would be indistinguishable from
    a real one to every reader, and it would bind the result to nothing.
    """

    revision: str = ""
    source: str = "unavailable"
    dirty: bool = False
    reason: str = ""

    @property
    def available(self) -> bool:
        return bool(self.revision)


def repository_root(start: Optional[Any] = None) -> Optional[Path]:
    """The nearest directory at or above `start` that holds a repository."""
    here = Path(os.fspath(start)) if start is not None else Path(__file__).resolve()
    if here.is_file():
        here = here.parent
    for candidate in (here, *here.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _git(root: Path, *args: str) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", *args],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT,
        check=False,
    )


def code_revision(repo_root: Optional[Any] = None) -> CodeRevision:
    """The commit the working tree is on, as git reports it.

    The dirty flag is part of the answer, not a footnote: a commit id names
    a snapshot, and a working tree with uncommitted edits is not that
    snapshot. When the tree state cannot be read the revision is reported as
    dirty, because an unknown tree is not a clean one.
    """
    root = repository_root(repo_root)
    if root is None:
        return CodeRevision(
            revision="",
            source="unavailable",
            dirty=True,
            reason=("no git repository was found, so the code revision cannot "
                    "be determined; there is no honest fallback for it"),
        )

    try:
        head = _git(root, "rev-parse", "HEAD")
    except (OSError, subprocess.SubprocessError) as exc:
        return CodeRevision(
            revision="",
            source="unavailable",
            dirty=True,
            reason=f"git could not be run: {exc}",
        )

    if head.returncode != 0:
        detail = (head.stderr or head.stdout or "").strip().splitlines()
        return CodeRevision(
            revision="",
            source="unavailable",
            dirty=True,
            reason=("git could not resolve HEAD: "
                    + (detail[0] if detail else "unknown error")),
        )

    revision = head.stdout.strip()
    if not is_real_revision(revision):
        return CodeRevision(
            revision="",
            source="unavailable",
            dirty=True,
            reason=(f"git reported a HEAD that is not a commit id: {revision!r}"),
        )

    dirty = True
    try:
        status = _git(root, "status", "--porcelain")
        if status.returncode == 0:
            dirty = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        dirty = True

    return CodeRevision(revision=revision, source="git", dirty=dirty, reason="")


@dataclass
class RunAttestation:
    """The chain that binds a result to the run that produced it.

    ``attested`` is True only when every field below was computed from real
    inputs: a real dataset, a real repository state, and a model fingerprint
    that was itself attested. An unattested chain is a record of what could
    not be established, not evidence.
    """

    run_id: str = ""
    dataset_sha256: str = ""
    code_revision: str = ""
    code_revision_source: str = "unavailable"
    code_revision_dirty: bool = False
    weights_sha256: str = ""
    model_id: str = ""
    attested: bool = False
    attestation_reason: str = ""
    recorded_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _weights_from(model_fingerprint: Any) -> Tuple[str, str, str]:
    """The weights digest from a model fingerprint, or why there is none.

    Returns ``(weights_sha256, model_id, reason)``. A non-empty reason means
    the chain cannot be completed.
    """
    if model_fingerprint is None:
        return "", "", ("no model fingerprint was supplied, so the weights "
                        "cannot be attested")

    record: Any = model_fingerprint
    if not isinstance(record, dict):
        to_dict = getattr(record, "to_dict", None)
        if callable(to_dict):
            record = to_dict()
        else:
            record = getattr(record, "__dict__", {})
    if not isinstance(record, dict):
        return "", "", "the model fingerprint could not be read"

    model_id = str(record.get("model_id") or "")
    if record.get("attested") is not True:
        detail = str(record.get("attestation_reason") or "").strip()
        return "", model_id, (
            "the model fingerprint is not attested"
            + (f": {detail}" if detail else ""))

    weights = str(record.get("weights_sha256") or "")
    if not is_real_digest(weights):
        return "", model_id, (
            "the model fingerprint carries no real weights digest")
    return weights, model_id, ""


def attest_run(*, run_id: Any, dataset: Any = None,
               model_fingerprint: Any = None,
               repo_root: Optional[Any] = None) -> RunAttestation:
    """Bind a run to its data, its code and its weights.

    Fails closed. If the dataset cannot be hashed, the repository cannot be
    asked, or the model fingerprint was never attested, the returned
    attestation says so in ``attestation_reason`` and leaves ``attested``
    False. It never substitutes a plausible-looking digest for a missing
    one.
    """
    attestation = RunAttestation(
        recorded_at=datetime.datetime.now(datetime.timezone.utc).isoformat())

    run_id_text = str(run_id or "").strip()
    if not run_id_text:
        attestation.attestation_reason = (
            "the run has no id, so the chain cannot be bound to a run")
        return attestation
    attestation.run_id = run_id_text

    weights, model_id, weights_reason = _weights_from(model_fingerprint)
    attestation.weights_sha256 = weights
    attestation.model_id = model_id
    if weights_reason:
        attestation.attestation_reason = weights_reason
        return attestation

    try:
        attestation.dataset_sha256 = dataset_digest(dataset)
    except EmptyDatasetError as exc:
        attestation.attestation_reason = str(exc)
        return attestation
    except Exception as exc:  # noqa: BLE001 - a failure here is a refusal
        attestation.attestation_reason = (
            f"the dataset could not be hashed: {exc}")
        return attestation

    revision = code_revision(repo_root)
    attestation.code_revision = revision.revision
    attestation.code_revision_source = revision.source
    attestation.code_revision_dirty = revision.dirty
    if not revision.available:
        attestation.attestation_reason = revision.reason or (
            "the code revision could not be determined")
        return attestation

    attestation.attested = True
    attestation.attestation_reason = ""
    return attestation


def verify_run_attestation(attestation: Any, dataset: Any, *,
                           expected_code_revision: Optional[str] = None,
                           expected_weights_sha256: Optional[str] = None,
                           ) -> Dict[str, Any]:
    """Check a run attestation against the data and code it claims.

    `dataset` is required: the dataset digest is recomputed from the data
    and compared with the recorded one. A check that took the recorded
    digest as its input would only prove that the record agrees with itself.
    """
    record: Any = attestation
    if not isinstance(record, dict):
        to_dict = getattr(record, "to_dict", None)
        record = to_dict() if callable(to_dict) else record
    if not isinstance(record, dict):
        return {"valid": False, "checks": {},
                "reason": "the run attestation is not a record"}

    checks: Dict[str, Any] = {
        "attested": record.get("attested") is True,
        "run_id": bool(str(record.get("run_id") or "").strip()),
        "dataset_digest_is_real": is_real_digest(record.get("dataset_sha256")),
        "code_revision_is_real": is_real_revision(record.get("code_revision")),
        "weights_digest_is_real": is_real_digest(record.get("weights_sha256")),
    }

    try:
        recomputed = dataset_digest(dataset)
    except Exception as exc:  # noqa: BLE001 - a failure here is a refusal
        checks["dataset_matches"] = False
        return {"valid": False, "checks": checks,
                "reason": f"the dataset could not be hashed for verification: {exc}"}
    checks["dataset_matches"] = recomputed == record.get("dataset_sha256")

    if expected_code_revision:
        checks["code_revision_matches"] = (
            str(expected_code_revision) == str(record.get("code_revision")))
    if expected_weights_sha256:
        checks["weights_match"] = (
            str(expected_weights_sha256) == str(record.get("weights_sha256")))

    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        return {"valid": False, "checks": checks,
                "reason": "failed checks: " + ", ".join(failed)}
    return {"valid": True, "checks": checks, "reason": None}
