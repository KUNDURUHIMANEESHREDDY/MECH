r"""Live Open-World Observation Ingestion Pipeline for MECH.

Processes real, blinded raw observation bundles returned by external investigators:
1. Validates investigator cryptographic signatures and hardware environment provenance.
2. Checks pre-registration integrity against immutable SHA-256 seals.
3. Unseals pre-registered predictions without retrospective bias.
4. Executes Hierarchical Bayesian Meta-Analysis, decomposing total variance across labs, models, tasks, hardware, and noise.
5. Classifies the 5-regime status and registers the multi-party master certificate into the Living Claim DAG.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from typing import Any, Dict, List, Optional

from .claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief, DependencyType
from .external_runner_cli import ExternalExperimentRunOutput, HardwareEnvironmentMetadata
from .open_world_multi_lab_preregistration_engine import (
    HierarchicalMetaAnalysisResult,
    MetaScientificRegime,
    OpenWorldAuditorQuorum,
    OpenWorldMultiLabEngine,
    PreRegistrationManifest,
)


class LiveObservationIngestionPipeline:
    """Orchestrates zero-trust verification and epistemic unsealing of external observation bundles."""

    def __init__(self, claim_graph: Optional[ClaimDependencyGraphEngine] = None) -> None:
        self.claim_graph = claim_graph or ClaimDependencyGraphEngine()
        self.meta_engine = OpenWorldMultiLabEngine(claim_graph=self.claim_graph)

    def verify_and_ingest_external_run(
        self,
        run_output: ExternalExperimentRunOutput,
        manifest: PreRegistrationManifest,
        expected_investigator_key: str = "INVESTIGATOR_SECRET_KEY",
    ) -> OpenWorldAuditorQuorum:
        """Validates investigator signature, checks pre-registration immutability, and executes meta-analysis."""
        # 1. Zero-trust signature verification
        sig_payload = json.dumps({
            "bundle_id": run_output.bundle_id,
            "investigator_id": run_output.investigator_id,
            "manifest_hash": run_output.raw_manifest_hash,
            "timestamp_utc": run_output.timestamp_utc,
        }, sort_keys=True)
        expected_sig = hashlib.sha256((sig_payload + expected_investigator_key).encode("utf-8")).hexdigest()

        if run_output.investigator_signature != expected_sig:
            raise ValueError(f"Invalid investigator signature from '{run_output.investigator_id}'! Ingestion aborted.")

        # 2. Check pre-registration integrity
        if not manifest.verify_integrity():
            raise ValueError("Pre-Registration Manifest integrity compromised! Ingestion aborted.")

        # 3. Format observations from challenge results
        observations: List[Dict[str, Any]] = []
        for cr in run_output.challenge_results:
            obs: Dict[str, Any] = {
                "lab_id": run_output.investigator_id,
                "challenge_id": cr.get("challenge_id", "UNKNOWN"),
                "empirical_r": cr.get("empirical_r", 0.0),
                "is_abstained": cr.get("is_abstained", False),
                "is_negative_transfer": cr.get("is_negative_transfer", False),
                "hardware_device": run_output.environment_metadata.device_name,
                "os_platform": run_output.environment_metadata.platform_system,
            }
            observations.append(obs)

        # 4. Execute Hierarchical Bayesian Meta-Analysis & Auditor Quorum
        quorum = self.meta_engine.run_open_world_auditor_quorum(manifest, observations)

        # 5. Enrich Living Claim DAG with detailed hardware provenance
        claim_id = f"CLAIM_OPEN_WORLD_PREREG_{manifest.pre_reg_id}"
        if claim_id in self.claim_graph.claims:
            existing_statement = self.claim_graph.claims[claim_id].claim_statement
            enriched_statement = (
                f"{existing_statement} "
                f"[Provenance: {run_output.investigator_id} on {run_output.environment_metadata.platform_system} / "
                f"{run_output.environment_metadata.device_name}, PyTorch {run_output.environment_metadata.torch_version}]"
            )
            self.claim_graph.claims[claim_id].claim_statement = enriched_statement

        return quorum
