r"""Adversarial Holdout Selector & Blind Challenge Sealer for MECH.

Presents observable challenge metadata to MECH while cryptographically hiding:
1. The adversary's scoring objective
2. The expected failure probability
3. The empirical ground truth
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .adversarial_replication_oracle import AdversarialChallengeCase


@dataclass
class SealedAdversarialPackage:
    package_id: str
    sealed_manifest_hash: str
    blind_challenges: List[Dict[str, Any]]
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "package_id": self.package_id,
            "sealed_manifest_hash": self.sealed_manifest_hash,
            "blind_challenges": self.blind_challenges,
            "timestamp_utc": self.timestamp_utc,
        }


class AdversarialHoldoutSelector:
    """Packages and seals adversarial challenges without leaking evaluation targets."""

    def seal_adversarial_battery(self, cases: List[AdversarialChallengeCase]) -> SealedAdversarialPackage:
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        manifest_payload = [c.to_dict() for c in cases]
        raw_manifest = json.dumps(manifest_payload, sort_keys=True)
        manifest_hash = hashlib.sha256(raw_manifest.encode("utf-8")).hexdigest()

        blind_items: List[Dict[str, Any]] = []
        for c in cases:
            blind_items.append({
                "case_id": c.case_id,
                "source_model": c.source_model,
                "target_model": c.target_model,
                "task_name": c.task_name,
                "probes": [f"PROBE_{c.task_name.upper()}_ADV_ALPHA", f"PROBE_{c.task_name.upper()}_ADV_BETA"],
                "model_hash": hashlib.sha256(c.target_model.encode("utf-8")).hexdigest()[:12],
                "task_hash": hashlib.sha256(c.task_name.encode("utf-8")).hexdigest()[:12],
            })

        pkg_id = f"ADV_PKG_{manifest_hash[:10]}"
        return SealedAdversarialPackage(
            package_id=pkg_id,
            sealed_manifest_hash=manifest_hash,
            blind_challenges=blind_items,
            timestamp_utc=ts,
        )
