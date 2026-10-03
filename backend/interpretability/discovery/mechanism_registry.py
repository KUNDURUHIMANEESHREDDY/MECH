"""Persistent Mechanism Registry with Reusable ProvenanceRecord & Algorithm Provenance."""

from __future__ import annotations

import datetime as _dt
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ProvenanceRecord:
    """Reusable provenance object capturing execution & algorithm provenance."""

    id: str = "prov_default"
    dataset_version: str = "openwebtext-v2.1"
    model_checkpoint_sha: str = "sha256:e9a4f81c9b201a34d"
    commit_sha: str = "git:a4f81c9"
    environment_hash: str = "env_py311_torch2.2"
    cuda_driver: str = "CUDA 12.2 / Driver 535.12"
    algorithm_name: str = "Circuit Discovery - Path Patching"
    algorithm_version: str = "v2.1.0"
    algorithm_parameters: Dict[str, Any] | None = None
    quality_weights_version: str = "qw_v1.0"
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.algorithm_parameters is None:
            self.algorithm_parameters = {"threshold": 0.05, "max_depth": 12}

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> ProvenanceRecord:
        if not data:
            return cls()
        return cls(
            id=data.get("id", "prov_default"),
            dataset_version=data.get("dataset_version", "openwebtext-v2.1"),
            model_checkpoint_sha=data.get("model_checkpoint_sha", "sha256:e9a4f81c9b201a34d"),
            commit_sha=data.get("commit_sha", "git:a4f81c9"),
            environment_hash=data.get("environment_hash", "env_py311_torch2.2"),
            cuda_driver=data.get("cuda_driver", "CUDA 12.2 / Driver 535.12"),
            algorithm_name=data.get("algorithm_name", "Circuit Discovery - Path Patching"),
            algorithm_version=data.get("algorithm_version", "v2.1.0"),
            algorithm_parameters=data.get("algorithm_parameters", {"threshold": 0.05, "max_depth": 12}),
            quality_weights_version=data.get("quality_weights_version", "qw_v1.0"),
            random_seed=data.get("random_seed", 42),
        )


class MechanismRegistry:
    """Persistent registry turning mechanisms into first-class research objects with complete execution and algorithm provenance."""

    def __init__(self) -> None:
        init_prov = ProvenanceRecord(id="prov_mech_ioi")
        self.mechanisms: Dict[str, Dict[str, Any]] = {
            # Placeholder so the registry is not empty on first read. Every score
            # here was invented: `replication_score: 0.985`, `confidence_score:
            # 0.96`, `evidence_count: 5` and `status: "Validated"` described a
            # paper-derived circuit as though this platform had validated it five
            # times over. Nothing was measured, and the scores are now absent
            # rather than plausible.
            #
            # The circuit itself is Wang et al.'s, transcribed -- not a MECH
            # result. `provenance: seeded` and the two ineligibility flags stop a
            # consumer ranking this above a real measurement.
            "mech_ioi": {
                "id": "mech_ioi",
                "name": "Name Recognition & IOI Circuit",
                "evidence_count": 0,
                "replication_score": None,
                "known_paper": "Wang et al. (2022)",
                "confidence_score": None,
                "confidence": None,
                "scientific_quality_score": None,
                "expected_impact_score": None,
                "status": "Registered",
                "provenance": "seeded",
                "validation_eligible": False,
                "publication_eligible": False,
                "provenance_record": asdict(init_prov),
            }
        }

    def register_mechanism(
        self,
        mechanism_id: str,
        name: str,
        confidence_score: float | None = None,
        scientific_quality_score: float | None = None,
        expected_impact_score: float | None = None,
        provenance: ProvenanceRecord | Dict[str, Any] | None = None,
        confidence: float | None = None,
        evidence_count: int = 0,
    ) -> Dict[str, Any]:
        """Record that a mechanism has been *registered*.

        Registration is bookkeeping, not evidence. Nothing here measures anything,
        so this method does not assert that it did.

        Previously every entry created here claimed, unconditionally:

            evidence_count: 1
            replication_score: 0.95
            status: "Validated"

        with `confidence_score` defaulting to 0.95 and the two quality scores to
        0.90 and 0.85. So calling this function turned an unevidenced claim about
        a circuit into a validated one carrying a 0.95 replication score, in one
        line, with no measurement performed anywhere. `status: "Validated"` was
        the worst of it: it is a claim about evidence, and the function's entire
        contribution was the absence of evidence.

        Now the scores default to None -- absent, meaning unmeasured -- and
        status is "Registered", which is the one statement here that is actually
        true. Callers that have measured something pass the number in; callers
        that have not are recorded as not having.

        Pass `evidence_count` for the number of evidence items actually attached.
        It defaults to 0 because registering a name attaches no evidence.
        """
        conf = confidence if confidence is not None else confidence_score
        prov_obj = provenance if isinstance(provenance, ProvenanceRecord) else ProvenanceRecord.from_dict(provenance)
        entry = {
            "id": mechanism_id,
            "name": name,
            "evidence_count": evidence_count,
            # No replication was run by calling this, so there is no score.
            "replication_score": None,
            "confidence_score": conf,
            "confidence": conf,  # Legacy alias for test compatibility
            "scientific_quality_score": scientific_quality_score,
            "expected_impact_score": expected_impact_score,
            # True statement about what happened, rather than about the evidence.
            "status": "Registered",
            "registered_at": _dt.datetime.utcnow().isoformat() + "Z",
            "provenance": asdict(prov_obj),
        }
        self.mechanisms[mechanism_id] = entry
        return entry

    def list_mechanisms(self) -> List[Dict[str, Any]]:
        return list(self.mechanisms.values())

    def get_mechanism_provenance(self, mechanism_id: str) -> Optional[Dict[str, Any]]:
        mech = self.mechanisms.get(mechanism_id)
        return mech.get("provenance") if mech else None
