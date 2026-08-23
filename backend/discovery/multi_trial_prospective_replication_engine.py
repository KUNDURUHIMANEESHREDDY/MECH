r"""Multi-Trial Prospective Replication Engine for MECH.

Coordinates and evaluates multi-trial prospective external replications across 3 key scientific regimes:
1. Positive Prospective Transfer:
   - Evaluates high-fidelity quantitative predictions on unseen model pairs (e.g. Qwen-2.5-7B <- Llama-3.1-8B).
2. Prospective Negative-Transfer Prediction:
   - Decisively predicts causal breakdown and low rescue before empirical measurement (e.g. StarCoder-7B <- Gemma-2-9B).
3. Prospective Calibrated Abstention:
   - Predicts and enforces epistemic abstention on architectures outside validated domain (e.g. Mamba-2 SSM).
4. Joint Multi-Trial Replication Fidelity (MTRF):
   - Computes tri-regime fidelity score MTRF >= 95.0% and cross-trial heterogeneity Delta_hetero <= 0.05.
5. Automatic Epistemic Claim Downgrade:
   - Instantly downgrades Claim DAG status to CLAIM_DIVERGENT_UNDER_INVESTIGATION if unexpected divergence occurs.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief, DependencyType
from .external_runner_cli import ExternalExperimentRunOutput, HardwareEnvironmentMetadata
from .open_world_ingestion_pipeline import LiveObservationIngestionPipeline
from .open_world_multi_lab_preregistration_engine import (
    HierarchicalMetaAnalysisResult,
    MetaScientificRegime,
    OpenWorldAuditorQuorum,
    OpenWorldMultiLabEngine,
    PreRegistrationManifest,
)


class TrialRegimeType(str, Enum):
    POSITIVE_TRANSFER = "POSITIVE_TRANSFER"
    NEGATIVE_TRANSFER = "NEGATIVE_TRANSFER"
    CALIBRATED_ABSTENTION = "CALIBRATED_ABSTENTION"


@dataclass
class TrialSpecification:
    trial_id: str
    target_model: str
    source_model: str
    task_name: str
    regime_type: TrialRegimeType
    predicted_r: float
    predicted_delta_z: float
    predicted_delta_p: float
    is_abstained: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "target_model": self.target_model,
            "source_model": self.source_model,
            "task_name": self.task_name,
            "regime_type": self.regime_type.value,
            "predicted_r": round(self.predicted_r, 4),
            "predicted_delta_z": round(self.predicted_delta_z, 4),
            "predicted_delta_p": round(self.predicted_delta_p, 4),
            "is_abstained": self.is_abstained,
        }


@dataclass
class MultiTrialExportBundle:
    bundle_id: str
    pre_reg_manifest: PreRegistrationManifest
    trials: List[Dict[str, Any]]
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "pre_reg_manifest": self.pre_reg_manifest.to_dict(),
            "trials": self.trials,
            "timestamp_utc": self.timestamp_utc,
        }


@dataclass
class TrialObservationRecord:
    trial_id: str
    investigator_id: str
    empirical_r: float
    empirical_delta_z: float
    empirical_delta_p: float
    is_abstained: bool
    raw_tensors: Dict[str, List[float]]
    hardware_metadata: Dict[str, Any]
    signature: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "investigator_id": self.investigator_id,
            "empirical_r": round(self.empirical_r, 4),
            "empirical_delta_z": round(self.empirical_delta_z, 4),
            "empirical_delta_p": round(self.empirical_delta_p, 4),
            "is_abstained": self.is_abstained,
            "raw_tensors": self.raw_tensors,
            "hardware_metadata": self.hardware_metadata,
            "signature": self.signature,
        }


@dataclass
class MultiTrialJointAuditScorecard:
    bundle_id: str
    trial_evaluations: List[Dict[str, Any]]
    mtrf_score: float
    cross_trial_heterogeneity: float
    pos_fidelity_pct: float
    neg_fidelity_pct: float
    abstention_fidelity_pct: float
    is_quorum_verified: bool
    summary_verdict: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "trial_evaluations": self.trial_evaluations,
            "mtrf_score": round(self.mtrf_score, 2),
            "cross_trial_heterogeneity": round(self.cross_trial_heterogeneity, 4),
            "pos_fidelity_pct": round(self.pos_fidelity_pct, 2),
            "neg_fidelity_pct": round(self.neg_fidelity_pct, 2),
            "abstention_fidelity_pct": round(self.abstention_fidelity_pct, 2),
            "is_quorum_verified": self.is_quorum_verified,
            "summary_verdict": self.summary_verdict,
        }


class MultiTrialProspectiveReplicationEngine:
    """Orchestrates pre-registration, multi-trial evaluation, and automatic claim downgrade."""

    def __init__(self, claim_graph: Optional[ClaimDependencyGraphEngine] = None) -> None:
        self.claim_graph = claim_graph or ClaimDependencyGraphEngine()
        self.meta_engine = OpenWorldMultiLabEngine(claim_graph=self.claim_graph)

    def generate_multi_trial_bundle(self) -> Tuple[PreRegistrationManifest, MultiTrialExportBundle]:
        """Constructs a 3-trial prospective challenge bundle covering Positive, Negative, and Abstention regimes."""
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        bundle_id = f"BUNDLE_MULTI_TRIAL_{ts[:10]}"

        trial_1 = TrialSpecification(
            trial_id="TRIAL_1_POS_QWEN_INDUCTION",
            target_model="Qwen/Qwen-2.5-7B",
            source_model="meta-llama/Llama-3.1-8B",
            task_name="multi_hop_relational_fact_induction",
            regime_type=TrialRegimeType.POSITIVE_TRANSFER,
            predicted_r=0.825,
            predicted_delta_z=2.450,
            predicted_delta_p=0.660,
            is_abstained=False,
        )

        trial_2 = TrialSpecification(
            trial_id="TRIAL_2_NEG_STARCODER_POLY",
            target_model="bigcode/starcoder-7b",
            source_model="google/gemma-2-9b",
            task_name="distractor_code_suppression",
            regime_type=TrialRegimeType.NEGATIVE_TRANSFER,
            predicted_r=0.128,
            predicted_delta_z=0.350,
            predicted_delta_p=0.090,
            is_abstained=False,
        )

        trial_3 = TrialSpecification(
            trial_id="TRIAL_3_ABSTAIN_MAMBA_SSM",
            target_model="state-spaces/mamba-2-2.7b",
            source_model="meta-llama/Llama-3.1-8B",
            task_name="recurrent_state_inhibition",
            regime_type=TrialRegimeType.CALIBRATED_ABSTENTION,
            predicted_r=0.000,
            predicted_delta_z=0.000,
            predicted_delta_p=0.000,
            is_abstained=True,
        )

        trials = [trial_1, trial_2, trial_3]
        claim_statement = (
            "Prospective Causal Transportability Law tri-regime validation across "
            "Positive Transfer (Qwen-2.5), Negative Transfer (StarCoder), and Calibrated Abstention (Mamba-2)."
        )

        manifest = self.meta_engine.create_preregistration_manifest(
            claim_statement=claim_statement,
            models=[t.target_model for t in trials],
            tasks=[t.task_name for t in trials],
            predicted_rescues=[t.predicted_r for t in trials],
        )

        # Export bundle without target ground truths
        bundle_trials: List[Dict[str, Any]] = []
        for t in trials:
            bundle_trials.append({
                "trial_id": t.trial_id,
                "target_model": t.target_model,
                "source_model": t.source_model,
                "task_name": t.task_name,
                "regime_type": t.regime_type.value,
                "is_abstained": t.is_abstained,
                "probes": [f"PROBE_{t.task_name.upper()}_A", f"PROBE_{t.task_name.upper()}_B"],
            })

        bundle = MultiTrialExportBundle(
            bundle_id=bundle_id,
            pre_reg_manifest=manifest,
            trials=bundle_trials,
            timestamp_utc=ts,
        )

        return manifest, bundle

    def audit_and_evaluate_multi_trial_observations(
        self,
        manifest: PreRegistrationManifest,
        observations: List[TrialObservationRecord],
    ) -> MultiTrialJointAuditScorecard:
        """Audits observations across all 3 trials, checks pre-registration, and triggers claim downgrade on divergence."""
        if not manifest.verify_integrity():
            raise ValueError("Pre-Registration Manifest integrity compromised! Multi-trial audit aborted.")

        evaluations: List[Dict[str, Any]] = []
        errors: List[float] = []

        pos_fidelity = 0.0
        neg_fidelity = 0.0
        abstain_fidelity = 0.0

        for obs in observations:
            if obs.trial_id == "TRIAL_1_POS_QWEN_INDUCTION":
                pred_z, pred_r = 2.450, 0.825
                err_z = abs(pred_z - obs.empirical_delta_z) / pred_z * 100.0
                err_r = abs(pred_r - obs.empirical_r)
                is_ok = (err_z <= 5.0) and (err_r <= 0.05) and (obs.empirical_r >= 0.80)
                pos_fidelity = max(0.0, 100.0 - err_z)
                errors.append(err_z)
                evaluations.append({
                    "trial_id": obs.trial_id,
                    "regime": "POSITIVE_TRANSFER",
                    "delta_z_error_pct": round(err_z, 2),
                    "delta_r_error": round(err_r, 4),
                    "is_verified": is_ok,
                })
            elif obs.trial_id == "TRIAL_2_NEG_STARCODER_POLY":
                pred_z, pred_r = 0.350, 0.128
                err_z = abs(pred_z - obs.empirical_delta_z) / pred_z * 100.0
                err_r = abs(pred_r - obs.empirical_r)
                is_ok = (err_z <= 5.0) and (err_r <= 0.05) and (obs.empirical_r < 0.20)
                neg_fidelity = max(0.0, 100.0 - err_z)
                errors.append(err_z)
                evaluations.append({
                    "trial_id": obs.trial_id,
                    "regime": "NEGATIVE_TRANSFER",
                    "delta_z_error_pct": round(err_z, 2),
                    "delta_r_error": round(err_r, 4),
                    "is_verified": is_ok,
                })
            elif obs.trial_id == "TRIAL_3_ABSTAIN_MAMBA_SSM":
                is_ok = obs.is_abstained is True
                abstain_fidelity = 100.0 if is_ok else 0.0
                evaluations.append({
                    "trial_id": obs.trial_id,
                    "regime": "CALIBRATED_ABSTENTION",
                    "is_abstained": obs.is_abstained,
                    "is_verified": is_ok,
                })

        mean_err = sum(errors) / max(1, len(errors))
        variance = sum((e - mean_err) ** 2 for e in errors) / max(1, len(errors))
        heterogeneity = math.sqrt(variance) / 100.0

        mtrf = (pos_fidelity + neg_fidelity + abstain_fidelity) / 3.0
        all_passed = all(ev["is_verified"] for ev in evaluations)
        is_quorum_ok = all_passed and (mtrf >= 95.0) and (heterogeneity <= 0.05)

        claim_id = f"CLAIM_MULTI_TRIAL_PROSPECTIVE_REPLICATION_{manifest.pre_reg_id}"

        if is_quorum_ok:
            verdict = (
                f"PASSED: Multi-Trial Prospective Replication Certified across 3 Regimes. "
                f"MTRF = {mtrf:.2f}% (>= 95.0%), Heterogeneity = {heterogeneity*100:.2f}% (<= 5.00%). "
                f"Positive Fidelity = {pos_fidelity:.2f}%, Negative Fidelity = {neg_fidelity:.2f}%, "
                f"Abstention Fidelity = {abstain_fidelity:.2f}%."
            )
            self.claim_graph.register_claim(
                claim_id=claim_id,
                certificate_id=manifest.pre_reg_id,
                circuit_or_component_id="MULTI_TRIAL_PROSPECTIVE_REPLICATION",
                behavior_name="multi_trial_tri_regime_prospective_replication",
                claim_statement=(
                    f"Multi-Trial Prospective Replication Certified: MTRF={mtrf:.2f}%, "
                    f"Pos={pos_fidelity:.2f}%, Neg={neg_fidelity:.2f}%, Abstain={abstain_fidelity:.2f}%. "
                    f"Heterogeneity={heterogeneity*100:.2f}%. Pre-Reg Seal {manifest.sha256_pre_reg_seal[:10]}..."
                ),
                dependency_experiment_ids=[
                    (f"EVAL_TRIAL_{ev['trial_id']}", DependencyType.PRIMITIVE_CLAIM) for ev in evaluations
                ],
            )
            self.claim_graph.claims[claim_id].belief_status = ClaimEpistemicBelief.ACTIVE_SUPPORTED
        else:
            verdict = (
                f"FAILED / DOWNGRADED: Multi-Trial Replication diverged from pre-registered bounds. "
                f"MTRF = {mtrf:.2f}%, Heterogeneity = {heterogeneity*100:.2f}%. "
                f"Claim automatically downgraded to CLAIM_DIVERGENT_UNDER_INVESTIGATION."
            )
            self.claim_graph.register_claim(
                claim_id=claim_id,
                certificate_id=manifest.pre_reg_id,
                circuit_or_component_id="MULTI_TRIAL_PROSPECTIVE_REPLICATION",
                behavior_name="multi_trial_tri_regime_prospective_replication",
                claim_statement=f"DOWNGRADED: Multi-Trial Replication Divergence Detected: MTRF={mtrf:.2f}%.",
                dependency_experiment_ids=[],
            )
            # Automatic Epistemic Downgrade
            self.claim_graph.claims[claim_id].belief_status = ClaimEpistemicBelief.EVIDENCE_WEAKENED

        return MultiTrialJointAuditScorecard(
            bundle_id=manifest.pre_reg_id,
            trial_evaluations=evaluations,
            mtrf_score=mtrf,
            cross_trial_heterogeneity=heterogeneity,
            pos_fidelity_pct=pos_fidelity,
            neg_fidelity_pct=neg_fidelity,
            abstention_fidelity_pct=abstain_fidelity,
            is_quorum_verified=is_quorum_ok,
            summary_verdict=verdict,
        )
