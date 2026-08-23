r"""Physical Invariant Probe Engine for MECH.

ALL measurements come from the live GPT-2 model via Gpt2LiveExperimentRunner.
No hardcoded numeric values exist in this module.

Runs four physical substrate perturbations on real GPT-2 weights:
    GAUSSIAN_NOISE        — N(0, 0.05) noise on all parameters
    WEIGHT_SCALING_075X   — MLP weights × 0.75
    SEQUENCE_EXTENSION    — prompt padded to 2× length with EOS tokens
    LAYER_ABLATION        — mid-layer residual stream zeroed out

CSF = 1 - |ΔY_baseline - ΔY_perturbed| / max(ε, |ΔY_baseline|)
computed from actual GPT-2 logit differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PhysicalProbeResult:
    perturbation_type: str
    baseline_intervention_response: float
    perturbed_intervention_response: float
    causal_signature_fidelity: float
    is_physically_grounded: bool
    probe_id: str = ""
    target_token: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "perturbation_type": self.perturbation_type,
            "baseline_intervention_response": round(self.baseline_intervention_response, 4),
            "perturbed_intervention_response": round(self.perturbed_intervention_response, 4),
            "causal_signature_fidelity": round(self.causal_signature_fidelity, 4),
            "is_physically_grounded": self.is_physically_grounded,
            "probe_id": self.probe_id,
            "target_token": self.target_token,
        }


class PhysicalInvariantProbeEngine:
    """
    Probes causal response stability across physical substrate perturbations.
    Requires a Gpt2LiveExperimentRunner to obtain real measurements.
    """

    def probe_substrate_stability(
        self,
        probe=None,
        runner=None,
        baseline_delta_y: Optional[float] = None,
    ) -> List[PhysicalProbeResult]:
        """
        Executes four physical substrate perturbations on the live GPT-2 model.

        Parameters
        ----------
        probe   : DynamicProbe (required) — the probe to measure
        runner  : Gpt2LiveExperimentRunner (required) — provides real measurements
        baseline_delta_y : ignored (legacy param, real value computed from model)

        Returns
        -------
        List[PhysicalProbeResult] with real CSF values from GPT-2.

        Raises
        ------
        ValueError if probe or runner is None.
        """
        if probe is None or runner is None:
            raise ValueError(
                "PhysicalInvariantProbeEngine.probe_substrate_stability() requires "
                "both `probe` and `runner` (Gpt2LiveExperimentRunner). "
                "Hardcoded results are not permitted — pass a live GPT-2 runtime."
            )

        perturbation_types = [
            "GAUSSIAN_NOISE",
            "WEIGHT_SCALING_075X",
            "SEQUENCE_EXTENSION",
            "LAYER_ABLATION",
        ]

        results: List[PhysicalProbeResult] = []
        for ptype in perturbation_types:
            live = runner.measure_physical_perturbation(probe, ptype)
            results.append(PhysicalProbeResult(
                perturbation_type=live.perturbation_type,
                baseline_intervention_response=live.baseline_delta_logit,
                perturbed_intervention_response=live.perturbed_delta_logit,
                causal_signature_fidelity=live.causal_signature_fidelity,
                is_physically_grounded=live.is_physically_grounded,
                probe_id=probe.probe_id,
                target_token=probe.target_token,
            ))

        return results
