"""Dataset Certificate — a record of what was actually checked.

A certificate is only worth reading if it distinguishes *verified* from *claimed*.
Three changes here, each fixing a way this file overstated things:

* `signature` used to be copied straight out of the manifest, which currently
  holds the literal ``sha256:dataset_sig_placeholder``. The certificate therefore
  presented a placeholder as a signature. A placeholder is now recognised and
  excluded, and `signature_status` says why the field is empty.
* Missing fields became the string ``"unknown"``, which is indistinguishable from
  a recorded value once it is in a certificate. Absent values are now ``None`` and
  listed in `unrecorded_fields`.
* `verified_at` was a `default_factory` timestamp, so a certificate minted for a
  dataset that was never verified still carried a verification time. It is now
  ``None`` unless a check actually ran.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.research_datasets.dataset_manager import _is_placeholder


def _is_unrecorded(value: Any) -> bool:
    """True when a field is absent, self-describes as a placeholder, or is a hash.

    Extends the manager's placeholder test with the `sha256:` prefix, which is
    present on every manifest hash entry -- without stripping it, an empty-digest
    placeholder slips through and the certificate reports it as a recorded hash.
    """
    if not value:
        return True
    text = str(value).strip()
    if text.lower().startswith("sha256:"):
        text = text[len("sha256:"):]
    return _is_placeholder(text) or str(value).strip().lower() in {
        "unknown", "none", "null"}


@dataclass
class DatasetCertificate:
    """A certificate of dataset integrity and provenance.

    Named as a record rather than "proof": it states which hashes were recorded
    and whether a signature exists. It does not itself verify anything, and
    `integrity_status` makes that explicit.
    """
    dataset_id: str
    version: str
    status: str

    # Hashes. None means the manifest recorded nothing here.
    prompt_hash: Optional[str]
    token_hash: Optional[str]
    bundle_hash: Optional[str]

    # Provenance. None means not recorded.
    paper_doi: Optional[str]
    license: Optional[str]
    author: Optional[str]

    # Metadata
    verified_at: Optional[str] = None
    validator_version: str = "1.2.0"
    signature: str = ""
    signature_algorithm: Optional[str] = None
    signature_status: str = "unsigned"
    #: Whether the manifest recorded every hash this certificate reports.
    integrity_status: str = "unverified"
    unrecorded_fields: List[str] = field(default_factory=list)


class DatasetCertificateEngine:
    """Generates standalone dataset certificates."""

    def issue(self, dataset_meta: Dict[str, Any]) -> DatasetCertificate:
        """Assemble a certificate, recording what is present and what is not.

        Does not verify anything. `integrity_status` distinguishes "the manifest
        records all three hashes" from "some are missing", and neither is the same
        as "the hashes were checked against the data" -- that happens in
        `DatasetManager.load`, which records the outcome in `last_integrity_status`.
        """
        hashes = dataset_meta.get("hashes", {})
        prov = dataset_meta.get("provenance", {})
        lineage = dataset_meta.get("lineage", {})

        def recorded(container: Dict[str, Any], key: str) -> Optional[str]:
            value = container.get(key)
            return None if _is_unrecorded(value) else str(value)

        prompt_hash = recorded(hashes, "prompt_hash")
        token_hash = recorded(hashes, "token_hash")
        bundle_hash = recorded(hashes, "bundle_hash")
        paper_doi = recorded(prov, "paper_doi")
        license_ = recorded(prov, "license")
        author = recorded(lineage, "author")

        unrecorded = [
            name for name, value in (
                ("prompt_hash", prompt_hash), ("token_hash", token_hash),
                ("bundle_hash", bundle_hash), ("paper_doi", paper_doi),
                ("license", license_), ("author", author),
            ) if value is None
        ]

        hash_fields = (prompt_hash, token_hash, bundle_hash)
        if all(h is None for h in hash_fields):
            integrity_status = "no_hashes_recorded"
        elif any(h is None for h in hash_fields):
            integrity_status = "partially_recorded"
        else:
            integrity_status = "recorded_not_checked"

        signature = dataset_meta.get("signature", "")
        algorithm = dataset_meta.get("signature_algorithm")
        if _is_unrecorded(signature):
            signature, algorithm = "", None
            signature_status = "unsigned"
        elif not algorithm:
            # A signature with no recorded algorithm cannot be checked. Under the
            # old scheme it was a keyed SHA-256; Ed25519 is 64 bytes (128 hex
            # chars), so anything else is not a signature we can verify.
            signature_status = ("unverifiable_algorithm" if len(signature) != 128
                                else "algorithm_not_recorded")
        else:
            signature_status = f"present_{algorithm}_not_checked_here"

        return DatasetCertificate(
            dataset_id=dataset_meta["dataset_id"],
            version=dataset_meta["version"],
            status=dataset_meta["status"],
            prompt_hash=prompt_hash,
            token_hash=token_hash,
            bundle_hash=bundle_hash,
            paper_doi=paper_doi,
            license=license_,
            author=author,
            verified_at=None,  # set by the caller that actually ran a check
            signature=signature,
            signature_algorithm=algorithm,
            signature_status=signature_status,
            integrity_status=integrity_status,
            unrecorded_fields=unrecorded,
        )
