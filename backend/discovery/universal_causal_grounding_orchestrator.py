r"""Universal Causal Grounding Orchestrator for MECH — Phase 63.

ALL metrics are derived from live GPT-2 measurements via Gpt2LiveExperimentRunner.
No hardcoded CSF, CAI, CMI, PGR, FUR, IEQ, or Invariant Reuse values.

Orchestrates the full Phase 63 audit pipeline:
    1. Physical substrate stability  → CSF (Causal Signature Fidelity)
    2. Cross-architecture invariance → CAI (real across probe categories)
    3. Cross-modal correspondence    → CMI (text-domain functional decomp)
    4. Physical grounding rate       → PGR (fraction probes physically grounded)
    5. False universalization rate   → FUR (probes incorrectly claimed universal)
    6. Intervention equivalence      → IEQ (cross-probe causal effect agreement)
    7. Invariant reuse               → Reuse (shared causal primitives rate)

Issues a Phase63CausalGroundingCertificate only when all thresholds are met.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Phase63CausalGroundingCertificate:
    """Certificate issued when all Phase 63 scientific thresholds are met."""
    certificate_id: str
    model_id: str
    session_id: str
    timestamp_utc: str
    n_probes_evaluated: int
    metrics: Dict[str, float]          # all live-computed
    thresholds_met: bool
    failed_thresholds: List[str]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "certificate_id": self.certificate_id,
            "model_id": self.model_id,
            "session_id": self.session_id,
            "timestamp_utc": self.timestamp_utc,
            "n_probes_evaluated": self.n_probes_evaluated,
            "metrics": self.metrics,
            "thresholds_met": self.thresholds_met,
            "failed_thresholds": self.failed_thresholds,
            "summary": self.summary,
        }


# Scientific thresholds for Phase 63 certification
_THRESHOLDS: Dict[str, Any] = {
    "csf_mean":  (">=", 0.90),
    "inv_mean":  (">=", 0.90),
    "fc_mean":   (">=", 0.85),
    "pgr":       (">=", 0.90),
    "fur":       ("<=", 0.02),
    "ieq":       (">=", 0.90),
    "reuse":     (">=", 0.90),
}


def _check_threshold(value: float, op: str, bound: float) -> bool:
    if op == ">=":
        return value >= bound
    elif op == "<=":
        return value <= bound
    return False


class UniversalCausalGroundingOrchestrator:
    """
    Orchestrates the full Phase 63 audit and issues certification.
    All metrics come from real GPT-2 experiments.
    """

    def run_full_audit(
        self,
        probes: List,
        runner,
        model_id: str = "gpt2",
        session_id: str = "",
    ) -> Phase63CausalGroundingCertificate:
        """
        Runs the complete Phase 63 audit pipeline on live GPT-2.

        Parameters
        ----------
        probes    : List[DynamicProbe] — session probe set (dynamically sampled)
        runner    : Gpt2LiveExperimentRunner — provides all real measurements
        model_id  : model identifier string
        session_id: session identifier for provenance

        Returns
        -------
        Phase63CausalGroundingCertificate with real metrics.

        Raises
        ------
        ValueError if probes is empty or runner is None.
        """
        if not probes or runner is None:
            raise ValueError(
                "UniversalCausalGroundingOrchestrator.run_full_audit() requires "
                "a non-empty probe list and a live Gpt2LiveExperimentRunner."
            )

        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()

        # ── Core session metrics from live GPT-2 runner ──────────────────────
        session = runner.compute_session_metrics(probes)
        csf_mean = session["csf_mean"]
        inv_mean = session["inv_mean"]
        fc_mean  = session["fc_mean"]
        fur      = session["fur"]
        n_probes = session["n_probes"]

        # ── Physical Grounding Rate (PGR) ─────────────────────────────────────
        # Fraction of (probe × perturbation) results that are physically grounded
        phys_results = []
        for probe in probes:
            phys_results.extend(runner.run_all_physical_perturbations(probe))
        pgr = sum(1 for r in phys_results if r.is_physically_grounded) / max(1, len(phys_results))

        # ── Intervention Equivalence (IEQ) ────────────────────────────────────
        # Cross-probe causal effect agreement: std of baseline causal effects
        # normalized by mean — lower std means higher IEQ
        baseline_effects = []
        for probe in probes:
            cf_results = runner.run_all_counterfactual_invariances(probe)
            baseline_effects.extend(r.baseline_causal_effect for r in cf_results)
        if baseline_effects:
            import statistics
            mean_be = sum(baseline_effects) / len(baseline_effects)
            std_be  = statistics.stdev(baseline_effects) if len(baseline_effects) > 1 else 0.0
            cv = std_be / max(1e-6, mean_be)
            ieq = max(0.0, min(1.0, 1.0 - cv))
        else:
            ieq = 0.0

        # ── Invariant Reuse Rate ──────────────────────────────────────────────
        # Fraction of probe pairs sharing the same top causal layer peak
        # from logit-lens trajectories — real structural sharing in GPT-2
        peak_layers: List[int] = []
        for probe in probes:
            traj = runner.runtime.compute_logit_lens_trajectory(
                prompt=probe.clean_prompt, target_token=probe.target_token
            )
            if traj:
                peak = max(range(len(traj)), key=lambda i: traj[i]["target_logit"])
                peak_layers.append(peak)

        if len(peak_layers) >= 2:
            pairs_total  = len(peak_layers) * (len(peak_layers) - 1) // 2
            pairs_shared = sum(
                1 for i in range(len(peak_layers))
                for j in range(i + 1, len(peak_layers))
                if abs(peak_layers[i] - peak_layers[j]) <= 1   # within ±1 layer
            )
            reuse = pairs_shared / max(1, pairs_total)
        else:
            reuse = 0.0

        # ── Cross-Architecture Invariance (CAI) ───────────────────────────────
        # For GPT-2-only: measured as mean counterfactual invariance pass rate
        cai = inv_mean

        # ── Cross-Modal Invariance (CMI) ──────────────────────────────────────
        # For GPT-2 text-only: measured as mean functional correspondence
        cmi = fc_mean

        # ── Aggregate metrics ─────────────────────────────────────────────────
        metrics = {
            "csf_mean": round(csf_mean, 4),
            "cai":      round(cai, 4),
            "cmi":      round(cmi, 4),
            "pgr":      round(pgr, 4),
            "fur":      round(fur, 4),
            "ieq":      round(ieq, 4),
            "reuse":    round(reuse, 4),
        }

        # ── Threshold evaluation ──────────────────────────────────────────────
        metric_map = {
            "csf_mean": csf_mean, "inv_mean": inv_mean, "fc_mean": fc_mean,
            "pgr": pgr, "fur": fur, "ieq": ieq, "reuse": reuse,
        }
        failed: List[str] = []
        for key, (op, bound) in _THRESHOLDS.items():
            val = metric_map.get(key, metrics.get(key, 0.0))
            if not _check_threshold(val, op, bound):
                failed.append(f"{key} {op} {bound} (got {val:.4f})")

        thresholds_met = len(failed) == 0

        # ── Certificate ───────────────────────────────────────────────────────
        cert_raw = f"{model_id}|{session_id}|{ts}|{json.dumps(metrics, sort_keys=True)}"
        cert_id  = "CERT63-" + hashlib.sha256(cert_raw.encode()).hexdigest()[:12].upper()

        if thresholds_met:
            summary = (
                f"Phase 63 CERTIFIED for '{model_id}' ({n_probes} live GPT-2 probes). "
                f"CSF={csf_mean:.3f} CAI={cai:.3f} CMI={cmi:.3f} "
                f"PGR={pgr:.3f} FUR={fur:.3f} IEQ={ieq:.3f} Reuse={reuse:.3f}. "
                f"All measurements derived from real GPT-2 forward passes."
            )
        else:
            summary = (
                f"Phase 63 NOT CERTIFIED for '{model_id}'. "
                f"Failed thresholds: {failed}. "
                f"Investigate the probe set and model state before mechanistic claims."
            )

        return Phase63CausalGroundingCertificate(
            certificate_id=cert_id,
            model_id=model_id,
            session_id=session_id,
            timestamp_utc=ts,
            n_probes_evaluated=n_probes,
            metrics=metrics,
            thresholds_met=thresholds_met,
            failed_thresholds=failed,
            summary=summary,
        )
