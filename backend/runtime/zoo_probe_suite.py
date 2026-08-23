"""Standardized Mechanistic Probe Suite for Cross-Model Evaluation.

Defines invariant factual and relational probes that run identically across any
model scale (0.1B -> 7B+) to measure causal fidelity, Logit Lens dynamics,
and path-patching mediation rescue.

Every probe result now carries a BehavioralValidation that cleanly separates:

    execution_pass  — the infrastructure ran without error
    behavioral_pass — the model actually produced the expected token

The gate enforced downstream:

    execution_pass AND behavioral_pass
    ──────────────────────────────────
    BEFORE any result enters the causal discovery pipeline.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .behavioral_validation import (
    BehavioralValidation,
    BehavioralGateDecision,
    build_behavioral_validation,
    check_promotion_gate,
)
from .interfaces import ModelRuntimeInterface, PathHopSpec
import logging
logger = logging.getLogger(__name__)



@dataclass(frozen=True)
class StandardMechanisticProbe:
    """A standardized causal probe specification."""
    probe_id: str
    category: str                       # "factual_recall" | "relational" | "subject_attribute"
    clean_prompt: str
    target_token: str
    corrupted_prompt: str
    distractor_token: str
    target_layer_fraction: float = 0.66  # e.g., 2/3 depth through the model
    target_neuron_idx: int = 412

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Built-in Standard Benchmark Probes ───────────────────────────────────────

STANDARD_PROBES: List[StandardMechanisticProbe] = [
    StandardMechanisticProbe(
        probe_id="probe_capital_france",
        category="factual_recall",
        clean_prompt="The capital of France is",
        target_token=" Paris",
        corrupted_prompt="The capital of Italy is",
        distractor_token=" Rome",
        target_layer_fraction=0.66,
        target_neuron_idx=412,
    ),
    StandardMechanisticProbe(
        probe_id="probe_landmark_eiffel",
        category="subject_attribute",
        clean_prompt="The Eiffel Tower is located in",
        target_token=" Paris",
        corrupted_prompt="The Colosseum is located in",
        distractor_token=" Rome",
        target_layer_fraction=0.66,
        target_neuron_idx=412,
    ),
    StandardMechanisticProbe(
        probe_id="probe_language_spain",
        category="relational",
        clean_prompt="The official language of Spain is",
        target_token=" Spanish",
        corrupted_prompt="The official language of Germany is",
        distractor_token=" German",
        target_layer_fraction=0.66,
        target_neuron_idx=412,
    ),
]


@dataclass
class StandardProbeExecutionResult:
    """Measurements and causal metrics captured from executing a standard probe on a model.

    behavioral_validation
        Upstream gate result.  Callers MUST check::

            result.behavioral_validation.gate_decision == BehavioralGateDecision.PROMOTE

        before using this result in the causal discovery pipeline.
        If gate_decision is BLOCK, inspect ``block_reason`` and the
        ``top_k`` list to diagnose the prompt/model configuration.
    """
    probe_id: str
    model_id: str
    architecture: str
    precision: str
    runtime_type: str                   # "in_memory" | "out_of_core"
    clean_target_logit: float
    clean_target_probability: float
    clean_target_rank: int
    corrupted_target_logit: float
    corrupted_target_rank: int
    intervene_layer: int
    causal_delta_z: float
    indirect_effect: float
    mediation_rescue_fraction: float
    logit_lens_trajectory: List[float]
    execution_duration_ms: float
    behavioral_validation: BehavioralValidation = field(default=None)  # type: ignore[assignment]

    # Convenience accessors -------------------------------------------------
    @property
    def execution_pass(self) -> bool:
        return self.behavioral_validation.execution_pass if self.behavioral_validation else False

    @property
    def behavioral_pass(self) -> bool:
        return self.behavioral_validation.behavioral_pass if self.behavioral_validation else False

    @property
    def is_promotable(self) -> bool:
        """True only when BOTH execution_pass AND behavioral_pass are True."""
        return self.execution_pass and self.behavioral_pass

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.behavioral_validation is not None:
            d["behavioral_validation"] = self.behavioral_validation.to_dict()
        return d


def run_probe_on_runtime(
    runtime: ModelRuntimeInterface,
    probe: StandardMechanisticProbe,
) -> StandardProbeExecutionResult:
    """Executes a standardized probe on any model runtime and extracts empirical causal metrics.

    The returned result carries a ``behavioral_validation`` field that cleanly
    separates whether the *infrastructure* ran (execution_pass) from whether
    the *model* actually produced the expected token (behavioral_pass).

    Callers MUST check ``result.is_promotable`` before feeding this result
    into the causal discovery pipeline.  If ``is_promotable`` is False,
    inspect ``result.behavioral_validation`` to determine whether the failure
    is an infrastructure error (gate=ERROR) or a model configuration issue
    (gate=BLOCK — the model produced a different top-1 token).
    """
    execution_pass = True
    t0 = time.perf_counter()

    # 1. Determine target intervention layer proportional to model depth
    num_layers = runtime.num_layers
    intervene_layer = max(1, min(num_layers - 1, int(num_layers * probe.target_layer_fraction)))
    source_hop_layer = max(0, intervene_layer - 2)

    # 2. Clean and Corrupted forward runs
    fwd_clean = runtime.forward(probe.clean_prompt, target_token=probe.target_token, capture_layer_residuals=True)
    fwd_corrupted = runtime.forward(probe.corrupted_prompt, target_token=probe.target_token)

    # 3. Live top-k vocabulary projection from the final hidden state
    #    Source: model's own output — never hardcoded.
    try:
        vocab_proj = runtime.project_to_vocabulary(
            hidden_state=fwd_clean.layer_residuals.get(num_layers - 1) if fwd_clean.layer_residuals else None,
            top_k=10,
            target_token=probe.target_token,
        )
        live_top_k: List[Tuple[str, float]] = [
            (entry["token"], entry["probability"])
            for entry in vocab_proj.get("top_k", [])
        ]
    except Exception as exc:  # noqa: BLE001
        logger.debug("Swallowed exception: %s", exc)
        # Graceful fallback: use only what the forward pass already gives us
        live_top_k = [(fwd_clean.top_predicted_token, fwd_clean.target_probability or 0.0)]

    # 4. Build BehavioralValidation from live forward-pass measurements
    behavioral_validation = build_behavioral_validation(
        probe_id=probe.probe_id,
        expected_token=probe.target_token,
        observed_token=fwd_clean.top_predicted_token,         # live argmax from model
        expected_rank=fwd_clean.target_rank if fwd_clean.target_rank is not None else -1,
        expected_probability=fwd_clean.target_probability or 0.0,
        top_k=live_top_k,
        execution_pass=execution_pass,
    )

    # 5. Logit Lens trajectory
    ll_traj = runtime.compute_logit_lens_trajectory(prompt=probe.clean_prompt, target_token=probe.target_token)
    traj_logits = [f["target_logit"] for f in ll_traj]

    # 6. Targeted Causal Intervention (Ablation on intermediate layer)
    interv_out = runtime.apply_intervention(
        prompt=probe.clean_prompt,
        target_token=probe.target_token,
        layer=intervene_layer,
        component_type="neuron",
        component_index=probe.target_neuron_idx,
        ablation_scale=0.0,
    )

    # 7. Multi-Hop Path Patching and Mediation Rescue
    path_out = runtime.patch_path(
        source_prompt=probe.clean_prompt,
        target_prompt=probe.corrupted_prompt,
        target_token=probe.target_token,
        path_hops=[PathHopSpec(source_layer=source_hop_layer, target_layer=intervene_layer)],
    )

    duration_ms = (time.perf_counter() - t0) * 1000.0

    meta = runtime.get_runtime_metadata()

    return StandardProbeExecutionResult(
        probe_id=probe.probe_id,
        model_id=meta.model_id,
        architecture=meta.architecture,
        precision=meta.precision.weight_dtype if hasattr(meta.precision, "weight_dtype") else "fp32",
        runtime_type=meta.runtime_type,
        clean_target_logit=fwd_clean.target_logit or 0.0,
        clean_target_probability=fwd_clean.target_probability or 0.0,
        clean_target_rank=fwd_clean.target_rank if fwd_clean.target_rank is not None else -1,
        corrupted_target_logit=fwd_corrupted.target_logit or 0.0,
        corrupted_target_rank=fwd_corrupted.target_rank if fwd_corrupted.target_rank is not None else -1,
        intervene_layer=intervene_layer,
        causal_delta_z=interv_out.delta_logit or 0.0,
        indirect_effect=path_out.indirect_effect,
        mediation_rescue_fraction=path_out.mediation_rescue_fraction,
        logit_lens_trajectory=traj_logits,
        execution_duration_ms=round(duration_ms, 2),
        behavioral_validation=behavioral_validation,
    )
