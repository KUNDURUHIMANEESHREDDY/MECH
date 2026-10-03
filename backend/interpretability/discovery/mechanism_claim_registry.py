"""Persistent Scientific Mechanism Claim Registry.

Transforms the platform from an experiment tracker into a persistent 
scientific knowledge accumulation system.

Tracks long-term claim status, composite confidence, replication counts,
validated models, supporting/contradicting experiment counts, literature citations,
and algorithm provenance.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RegisteredMechanismClaim:
    """A persistent, accumulated scientific claim in the knowledge base.

    Defaults are the honest state for a claim that has just been *written down*,
    not the state a claim arrives in from the literature. A caller passing only a
    id, title and description has run no experiments and replicated nothing, so
    the defaults say so:

        status="Hypothesized", confidence=None, replications=0,
        supporting_experiments=0

    These were `status="Validated"`, `confidence=0.95`, `replications=1` and
    `supporting_experiments=1`, which meant constructing a claim produced one that
    claimed to be validated at 0.95 confidence with a replication and a
    supporting experiment already in hand. `supporting_experiments=1` is the
    sharpest edge of that: it asserted that an experiment had been run and had
    come out in favour, when the only thing that had happened was the
    constructor.

    `contradicting_experiments=0` is left alone -- zero contradicting
    experiments is a true statement about a brand-new claim.
    """
    claim_id: str
    title: str
    description: str
    # Options: Validated, Hypothesized, Contradicted, Under_Revision, Reported
    status: str = "Hypothesized"
    confidence: Optional[float] = None
    replications: int = 0
    models: List[str] = field(default_factory=list)
    supporting_experiments: int = 0
    contradicting_experiments: int = 0
    literature: List[str] = field(default_factory=list)
    algorithms_used: List[str] = field(default_factory=list)
    evidence_summary: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")
    updated_at: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "confidence": self.confidence,
            "replications": self.replications,
            "models": self.models,
            "supporting_experiments": self.supporting_experiments,
            "contradicting_experiments": self.contradicting_experiments,
            "literature": self.literature,
            "algorithms_used": self.algorithms_used,
            "evidence_summary": self.evidence_summary,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RegisteredMechanismClaim:
        return cls(
            claim_id=data["claim_id"],
            title=data["title"],
            description=data.get("description", ""),
            # Same reasoning as the field defaults. This one is worse: it is the
            # *deserialization* path, so every stored record written before these
            # fields existed, and every record missing them for any reason, was
            # silently promoted to Validated at 0.95 confidence the moment it was
            # loaded back.
            status=data.get("status", "Hypothesized"),
            confidence=data.get("confidence"),
            replications=data.get("replications", 0),
            models=data.get("models", []),
            supporting_experiments=data.get("supporting_experiments", 0),
            contradicting_experiments=data.get("contradicting_experiments", 0),
            literature=data.get("literature", []),
            algorithms_used=data.get("algorithms_used", []),
            evidence_summary=data.get("evidence_summary", {}),
            created_at=data.get("created_at", _dt.datetime.utcnow().isoformat() + "Z"),
            updated_at=data.get("updated_at", _dt.datetime.utcnow().isoformat() + "Z"),
        )


class MechanismClaimRegistry:
    """Persistent storage & accumulation engine for scientific mechanism claims."""

    def __init__(self, storage_dir: str = "backend/research_datasets") -> None:
        self.storage_dir = storage_dir
        self.storage_file = os.path.join(storage_dir, "claims_registry.json")
        self._claims: Dict[str, RegisteredMechanismClaim] = {}
        self.load()

    def load(self) -> None:
        """Loads claims from persistent JSON storage on disk."""
        if not os.path.exists(self.storage_file):
            self._seed_defaults()
            return

        try:
            with open(self.storage_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data.get("claims", []):
                    claim = RegisteredMechanismClaim.from_dict(item)
                    self._claims[claim.claim_id] = claim
        except Exception:
            self._seed_defaults()

    def save(self) -> None:
        """Persists all claims to disk JSON file."""
        os.makedirs(self.storage_dir, exist_ok=True)
        data = {
            "version": "1.0",
            "updated_at": _dt.datetime.utcnow().isoformat() + "Z",
            "claims": [c.to_dict() for c in self._claims.values()]
        }
        with open(self.storage_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _seed_defaults(self) -> None:
        """Seed the registry with claims *as reported in the literature*.

        These two entries describe what Wang et al. 2022 and Olsson et al. 2022
        found. They are not MECH results, and the numbers they previously carried
        were measurements of nothing on this platform:

            status="Validated", confidence=0.962 / 0.941,
            replications=14 / 22, supporting_experiments=103 / 145,
            contradicting_experiments=4 / 2

        `supporting_experiments=103` is the clearest fiction -- there is no record
        of 103 experiments, and MECH has run IOI circuit discovery once. The
        precise counts together with the "Validated" status presented a literature
        transcription as a long-running accumulation of this system's own
        evidence, inside a class whose own docstring calls it "persistent
        scientific knowledge accumulation".

        They are seeded as `status="Reported"`: the claim is in the literature,
        which is true and checkable against the citation. No confidence, no
        replication count, no evidence counts. The citations stay, because they
        are the provenance. MECH's own IOI measurement (faithfulness 0.724
        discovered; 0.598 for the published circuit through this harness) is
        recorded by the IOI pipeline, not by inventing counts here.
        """
        seed_claims = [
            RegisteredMechanismClaim(
                claim_id="claim_ioi_name_mover",
                title="IOI Name Mover Circuit",
                description="L9H9 and L10H0 act as primary Name Mover Heads writing directly to IO token logits.",
                status="Reported",
                confidence=None,
                replications=0,
                models=[],
                supporting_experiments=0,
                contradicting_experiments=0,
                literature=["Wang et al. 2022: Interpretability in the Wild", "Conmy et al. 2023: ACDC"],
                algorithms_used=["Attribution Patching", "ACDC", "Path Patching", "Causal Scrubbing", "Universality"]
            ),
            RegisteredMechanismClaim(
                claim_id="claim_induction_heads",
                title="Induction Head Sequence Repeater",
                description="Previous-token head L4H2 attends to token K-1 while Induction Head L5H1 copies token K to current position.",
                status="Reported",
                confidence=None,
                replications=0,
                models=[],
                supporting_experiments=0,
                contradicting_experiments=0,
                literature=["Olsson et al. 2022: In-context Learning and Induction Heads"],
                algorithms_used=["Attribution Patching", "ACDC", "Causal Scrubbing", "Universality"]
            )
        ]
        for claim in seed_claims:
            self._claims[claim.claim_id] = claim
        self.save()

    def register(self, claim: RegisteredMechanismClaim) -> None:
        """Registers a new claim or updates an existing claim in the registry."""
        claim.updated_at = _dt.datetime.utcnow().isoformat() + "Z"
        self._claims[claim.claim_id] = claim
        self.save()

    def record_replication(
        self,
        claim_id: str,
        model_id: str,
        success: bool,
        algorithm_used: str = "Causal Scrubbing",
    ) -> RegisteredMechanismClaim:
        """Accumulate a replication run into an existing claim.

        Confidence is the share of recorded experiments that supported the
        claim:

            confidence = supporting / (supporting + contradicting)

        and is None while no experiment has been recorded at all.

        It was previously a fixed step per event --
        `min(0.99, confidence + (1 - confidence) * 0.05)` on success,
        `- score_delta` (default 0.05) on failure, floored at 0.10 -- applied to a
        field that defaulted to 0.95. That number was not derived from anything
        measured: it moved by a constant per recorded event regardless of what
        the event was, and because it started at 0.95 and was capped at 0.99,
        twenty consecutive successful replications could only lift it to 0.99. It
        also crashed outright once `confidence` was allowed to be `None`, which is
        the correct state for a claim that has not been tested.

        The `score_delta` parameter is removed rather than kept. It had no callers
        and it encoded the fabricated step size.

        Known limitation, stated rather than hidden: the ratio is not weighted by
        sample size, so three successful replications and thirty both give
        `confidence = 1.000`. That is a base rate, not a certainty, and the counts
        (`replications`, `supporting_experiments`, `contradicting_experiments`)
        travel with it so a reader can see how few experiments produced it. A
        Wilson or Laplace-smoothed interval would be the right next step; it is
        not done here because that changes what the number means and deserves its
        own decision rather than arriving inside a fabrication fix.
        """
        if claim_id not in self._claims:
            raise ValueError(f"Claim '{claim_id}' not found in registry.")

        claim = self._claims[claim_id]
        claim.replications += 1

        if model_id not in claim.models:
            claim.models.append(model_id)

        if algorithm_used not in claim.algorithms_used:
            claim.algorithms_used.append(algorithm_used)

        if success:
            claim.supporting_experiments += 1
        else:
            claim.contradicting_experiments += 1
            if claim.contradicting_experiments > claim.supporting_experiments * 0.5:
                claim.status = "Under_Revision"

        total = claim.supporting_experiments + claim.contradicting_experiments
        claim.confidence = claim.supporting_experiments / total if total else None

        claim.updated_at = _dt.datetime.utcnow().isoformat() + "Z"
        self.save()
        return claim

    def get(self, claim_id: str) -> Optional[RegisteredMechanismClaim]:
        """Retrieves a single claim by ID."""
        return self._claims.get(claim_id)

    def list_all(self, status_filter: Optional[str] = None) -> List[RegisteredMechanismClaim]:
        """Lists all registered mechanism claims."""
        claims = list(self._claims.values())
        if status_filter:
            claims = [c for c in claims if c.status.lower() == status_filter.lower()]
        return claims
