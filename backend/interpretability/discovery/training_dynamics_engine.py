"""Training Dynamics Engine â€” Concept Evolution across Checkpoints.

Intended to track how semantic representations form and stabilize during model
training: the "birth" and "convergence" of a concept across checkpoints.

What this module used to do
---------------------------
`track_concept_birth(concept_name, checkpoint_results)` ignored
`checkpoint_results` entirely and returned literals:

    birth_step          500
    stabilization_step  2500
    is_stable           True
    evolution_timeline  [{100, noise}, {500, detected, 0.72},
                         {1000, stable, 0.88}, {5000, converged, 0.94}]

`analyze_learning_velocity(domain)` ignored `domain` and returned 0.65.

So a caller asking when a concept appeared received a specific training step
along with a confidence for it, and could not tell that no checkpoint had been
examined. The four timeline confidences were the sharpest edge: 0.72, 0.88 and
0.94 are exactly the shape of a measured formation curve, and they were
constants.

What it does now
----------------
The caller supplies per-checkpoint observations; the timeline is computed from
them. With nothing to compute from, the answer is "not determined" with a reason,
never a plausible step number.

Expected shape of `checkpoint_results`
--------------------------------------
One dict per checkpoint, ascending by training step::

    {"step": 500,  "detected": True,  "score": 0.72, "concept_id": "cap_5013"}
    {"step": 1000, "detected": True,  "score": 0.88}
    {"step": 5000, "detected": True,  "score": 0.94}

`step` is required. `detected` defaults to `score > threshold`. `score` is the
strength of the concept at that checkpoint. Entries lacking both `detected` and
`score` carry no observation and are reported as unobserved rather than counted
as absences.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .concept_evolution_engine import ConceptEvolutionEngine


@dataclass
class CheckpointEvent:
    step: int
    concept_id: str
    event_type: str            # BIRTH, STABILIZED, DRIFTED, DIED, UNOBSERVED
    #: Optional. An event derived from an observation carries the observation's
    #: own score; one that merely records an absence carries None, because "no
    #: evidence here" is not a weak positive.
    confidence: Optional[float]
    description: str
    #: How many checkpoints support this event.
    supporting_checkpoints: int = 0


class TrainingDynamicsEngine:
    """Analyzes the temporal formation of semantic representations.

    Reports what a set of checkpoints shows. With no checkpoints it reports
    nothing, which is the whole difference from before.
    """

    #: A concept must clear this score at `min_consecutive` consecutive
    #: checkpoints to count as stabilized. Expressed as named constants because
    #: they are a policy choice, not a measurement -- unlike the timeline
    #: confidences this module used to hardcode.
    DEFAULT_STABILITY_THRESHOLD = 0.80
    DEFAULT_MIN_CONSECUTIVE = 2

    def __init__(
        self,
        stability_threshold: float = DEFAULT_STABILITY_THRESHOLD,
        min_consecutive: int = DEFAULT_MIN_CONSECUTIVE,
    ) -> None:
        self.evolution = ConceptEvolutionEngine()
        self.events: List[CheckpointEvent] = []
        self.stability_threshold = stability_threshold
        self.min_consecutive = min_consecutive

    def track_concept_birth(
        self,
        concept_name: str,
        checkpoint_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Find when a concept first appears, and when it stabilizes.

        Returns `birth_step: None` and `stabilization_step: None` when the
        supplied checkpoints do not settle either. `timeline` carries one entry
        per checkpoint actually examined, each labelled with the score it was
        given -- not with a fabricated curve.
        """
        observations = self._observations(checkpoint_results)

        if not observations:
            reason = (
                f"No checkpoint observations were supplied for "
                f"'{concept_name}', so neither its birth nor its "
                f"stabilization can be located. Previously this returned a birth "
                f"step of 500 and a stabilization step of 2500 regardless of "
                f"input."
            )
            self.events.append(CheckpointEvent(
                step=-1,
                concept_id=f"TEMP-{concept_name.upper()}",
                event_type="UNOBSERVED",
                confidence=None,
                description=reason,
                supporting_checkpoints=0,
            ))
            return {
                "concept_name": concept_name,
                "measured": False,
                "birth_step": None,
                "stabilization_step": None,
                "is_stable": False,
                "is_stable_established": False,
                "n_checkpoints_examined": 0,
                "n_checkpoints_observed": 0,
                "stability_threshold": self.stability_threshold,
                "reason": reason,
                # Same key as the measured path below, so a caller never has to
                # check which branch ran to know where the timeline lives.
                "evolution_timeline": [],
            }

        # Timeline: one entry per examined checkpoint, in step order, labelled by
        # its own measured score. No interpolating between checkpoints and no
        # inventing checkpoints that were not run.
        timeline = [
            {
                "step": step,
                "status": status,
                "confidence": score,      # the observation's score, or None
                "measured": score is not None,
            }
            for step, status, score in self._classify(observations)
        ]

        detected = [(step, score) for step, score in observations
                    if score is not None and score >= self.stability_threshold]
        absent = [step for step, score in observations
                  if score is not None and score < self.stability_threshold]

        birth_step = detected[0][0] if detected else None

        stabilization_step = self._first_sustained_run(
            [step for step, _ in detected], observations
        )

        # `is_stable` is only True once a sustained run was actually located.
        # An absent detection at some step and a detected one at another is a
        # mixed picture, not a stable concept.
        is_stable = stabilization_step is not None

        if birth_step is not None:
            score_at_birth = detected[0][1]
            self.events.append(CheckpointEvent(
                step=birth_step,
                concept_id=self._concept_id(checkpoint_results, concept_name),
                event_type="BIRTH",
                confidence=score_at_birth,
                description=(
                    f"Concept '{concept_name}' first cleared "
                    f"{self.stability_threshold} at step {birth_step} "
                    f"(score {score_at_birth:.4f})."),
                supporting_checkpoints=1,
            ))
        if is_stable:
            self.events.append(CheckpointEvent(
                step=stabilization_step,
                concept_id=self._concept_id(checkpoint_results, concept_name),
                event_type="STABILIZED",
                confidence=self._score_at(observations, stabilization_step),
                description=(
                    f"Concept '{concept_name}' cleared "
                    f"{self.stability_threshold} at "
                    f"{self.min_consecutive} consecutive checkpoints from step "
                    f"{stabilization_step}."),
                supporting_checkpoints=self._run_length(
                    [step for step, _ in detected], stabilization_step),
            ))

        reason = None
        if birth_step is None:
            reason = (
                f"'{concept_name}' never cleared {self.stability_threshold} at "
                f"any of the {len(observations)} observed checkpoints "
                f"(highest score "
                f"{max((s for _, s in observations if s is not None), default=0.0):.4f}). "
                + (f"It was observed absent at steps {absent}."
                   if absent else "")
            )

        return {
            "concept_name": concept_name,
            "measured": True,
            "birth_step": birth_step,
            "stabilization_step": stabilization_step,
            "is_stable": is_stable,
            "is_stable_established": is_stable,
            "n_checkpoints_examined": len(checkpoint_results),
            "n_checkpoints_observed": len(observations),
            "stability_threshold": self.stability_threshold,
            "min_consecutive": self.min_consecutive,
            "reason": reason,
            "evolution_timeline": timeline,
        }

    def analyze_learning_velocity(
        self,
        domain: str,
        checkpoint_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """How quickly a domain's representation strengthens across checkpoints.

        Returns a dict, not a float. The old signature returned the constant
        `0.65` for any domain, so a caller had no way to tell a measured rate
        from a placeholder -- and a float has nowhere to record that it was one.
        """
        observations = self._observations(checkpoint_results or [])
        scored = [(step, score) for step, score in observations if score is not None]

        if len(scored) < 2:
            return {
                "domain": domain,
                "measured": False,
                "velocity_per_1k_steps": None,
                "first_step": scored[0][0] if scored else None,
                "last_step": scored[-1][0] if scored else None,
                "reason": (
                    f"Velocity needs at least two checkpoints with a measured "
                    f"score; {len(scored)} were supplied for '{domain}'. "
                    f"Previously this returned 0.65 for every domain."
                ),
            }

        scored.sort()
        (first_step, first_score), (last_step, last_score) = scored[0], scored[-1]
        span = last_step - first_step
        if span <= 0:
            return {
                "domain": domain,
                "measured": False,
                "velocity_per_1k_steps": None,
                "first_step": first_step,
                "last_step": last_step,
                "reason": (
                    f"Checkpoints for '{domain}' share step {first_step}, so no "
                    f"rate can be computed."
                ),
            }

        velocity = (last_score - first_score) / (span / 1000.0)
        return {
            "domain": domain,
            "measured": True,
            "velocity_per_1k_steps": round(velocity, 6),
            "first_step": first_step,
            "last_step": last_step,
            "first_score": round(first_score, 4),
            "last_score": round(last_score, 4),
            "n_checkpoints": len(scored),
            "reason": None,
        }

    # â”€â”€ internals â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _observations(
        self,
        checkpoint_results: List[Dict[str, Any]],
    ) -> List[tuple]:
        """Extract (step, score) pairs, sorted by step.

        An entry with neither `detected` nor `score` is skipped: it says nothing
        about the concept, so counting it as an absence would invent a dropout.
        """
        observations: List[tuple] = []
        for entry in checkpoint_results or []:
            if not isinstance(entry, dict):
                continue
            if "step" not in entry:
                continue
            try:
                step = int(entry["step"])
            except (TypeError, ValueError):
                continue

            if "score" in entry and entry["score"] is not None:
                try:
                    score = float(entry["score"])
                except (TypeError, ValueError):
                    continue
            elif "detected" in entry:
                # A binary detection is recorded as a score of 1.0 or 0.0. That
                # is a real observation, just a coarse one, and it is labelled
                # `binary` below so a reader knows the precision.
                score = 1.0 if entry["detected"] else 0.0
            else:
                continue

            observations.append((step, score))
        observations.sort(key=lambda pair: pair[0])
        return observations

    def _classify(self, observations: List[tuple]) -> List[tuple]:
        """Label each observed checkpoint from its own score."""
        out = []
        for step, score in observations:
            if score is None:
                status = "unobserved"
            elif score >= self.stability_threshold:
                status = "detected"
            else:
                status = "below_threshold"
            out.append((step, status, score))
        return out

    def _first_sustained_run(
        self,
        detected_steps: List[int],
        observations: List[tuple],
    ) -> Optional[int]:
        """First step from which `min_consecutive` detections follow consecutively.

        "Consecutively" means consecutive *examinations*, not consecutive step
        numbers -- checkpoints are irregularly spaced, and requiring adjacent
        integers would silently never fire.
        """
        observed_steps = [step for step, _ in observations]
        index = {step: i for i, step in enumerate(observed_steps)}

        for step in detected_steps:
            start = index.get(step)
            if start is None:
                continue
            window = observed_steps[start:start + self.min_consecutive]
            if len(window) == self.min_consecutive and all(
                    s in set(detected_steps) for s in window):
                return step
        return None

    def _run_length(self, detected_steps: List[int], from_step: int) -> int:
        ordered = sorted(detected_steps)
        if from_step not in ordered:
            return 0
        count = 0
        for step in ordered:
            if step < from_step:
                continue
            count += 1
        return count

    def _score_at(self, observations: List[tuple], step: int) -> Optional[float]:
        for candidate, score in observations:
            if candidate == step:
                return score
        return None

    def _concept_id(
        self,
        checkpoint_results: List[Dict[str, Any]],
        concept_name: str,
    ) -> str:
        """Prefer a real concept_id from the observations.

        The old code always produced `TEMP-{NAME}`, so a caller that supplied real
        concept ids had them replaced by a placeholder announcing that no id was
        known -- even when one was sitting in the input.
        """
        for entry in checkpoint_results or []:
            if isinstance(entry, dict):
                candidate = entry.get("concept_id")
                if candidate:
                    return str(candidate)
        return f"TEMP-{concept_name.upper()}"

    def get_events(self) -> List[Dict[str, Any]]:
        from dataclasses import asdict
        return [asdict(e) for e in self.events]