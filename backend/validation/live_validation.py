"""Live scientific validation executor.

Independently re-measures a live discovery on held-out prompts (name pairs
the discovery panel never used) and issues a verdict only from those fresh
measurements:

  * replication — each discovered head must reproduce its causal effect sign
    and at least half its magnitude on held-out prompts;
  * recovery — the discovered circuit must recover corrupted behavior via
    clean-to-corrupted injection on held-out prompts;
  * minimality — each circuit head must remain individually necessary
    (single-ablation effect >= 10% of baseline) on held-out prompts.

Sample-size caveat, stated up front: the held-out panel is small
(``HELD_OUT_PROMPTS`` prompts, currently three). A deterministic threshold
pass on three prompts is weak evidence of generalization — the Wilson
95% interval for 3/3 still spans roughly [0.44, 1.0]. The payload
therefore reports sample sizes, uncertainty intervals, and per-prompt
results alongside the verdict, and scopes the claim to the prompts,
model, and deterministic forward passes actually used. It is not
broad scientific replication.

The automated checks object is rule-based, not human peer review, and says
so explicitly via ``peer_review.kind == "automated-rule-check"`` and
``peer_review.is_human_peer_review is False``. A separate
``human_peer_review`` slot (None by default) is the only place a human
acceptance may be recorded. Confidence is a direct heuristic function of
the three measured pass rates, documented in the payload as uncalibrated.
"""

from __future__ import annotations

import math
from statistics import mean
from typing import Any, Dict, List, Optional, Set

from backend.agents.evidence_policy import field_map
from backend.core.provenance import attest_measurement

from backend.interpretability.discovery import live_measure as lm
from backend.interpretability.discovery.live_discovery import (
    NAMES,
    pair_for_index,
)


REPLICATION_FLOOR = 0.75
RECOVERY_FLOOR = 0.50
MINIMALITY_FLOOR = 0.50


def _wilson_interval(k: int, n: int, z: float = 1.96) -> Dict[str, Any]:
    """Wilson score 95% interval for a binomial proportion.

    Reports uncertainty honestly at small n: 3/3 gives roughly
    [0.44, 1.0], so a deterministic threshold pass on three prompts
    cannot be read as strong evidence of generalization. With no
    trials (n == 0) the interval is maximally uninformative [0, 1].
    """
    if n <= 0:
        return {"lo": 0.0, "hi": 1.0, "n": 0}
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return {"lo": round(max(0.0, center - half), 4),
            "hi": round(min(1.0, center + half), 4),
            "n": n}


def _mean_sd(values: List[float]) -> Dict[str, float]:
    """Mean, population sd, min, and max of a list of floats."""
    if not values:
        return {"mean": 0.0, "sd": 0.0, "min": 0.0, "max": 0.0}
    avg = mean(values)
    var = mean((v - avg) ** 2 for v in values)
    return {"mean": round(avg, 4),
            "sd": round(math.sqrt(var), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4)}


class LiveValidationExecutor:
    """Held-out re-measurement of a live discovery result."""

    method = ("held-out re-measurement: replication of head effects, "
              "injection recovery of the discovered circuit, and "
              "leave-one-out necessity, all on unseen name pairs")

    def validate(self, discovery: Dict[str, Any]) -> Dict[str, Any]:
        heads = discovery.get("heads") or []
        effects = {item.get("head"): item for item in
                   (discovery.get("head_effects") or [])
                   if isinstance(item, dict)}
        if not heads or not effects:
            return self._unavailable(
                "The discovery carries no measured heads to re-measure.")
        # Panel composition is verified before any model access, so these
        # checks are hermetic: they fail closed on malformed records with
        # or without weights loaded. The panels must be recorded in the
        # discovery result. There is no offset fallback: guessing a panel
        # from `n_prompts` (or defaulting it) reintroduces the convention
        # this check exists to replace — a record that used a different
        # count would then validate on overlapping prompts while looking
        # held-out. A missing panel fails closed.
        validation_panel = discovery.get("validation_panel")
        if not isinstance(validation_panel, list) or not validation_panel:
            return self._unavailable(
                "No validation panel recorded in the discovery result; "
                "held-out status cannot be proven without it."
            )
        discovery_panel = discovery.get("discovery_panel", [])
        if not isinstance(discovery_panel, list) or not discovery_panel:
            return self._unavailable(
                "No discovery panel recorded in the discovery result; "
                "disjointness cannot be verified without it."
            )

        # Verify disjointness on recorded pool indices: no index may
        # appear in more than one panel. This covers the validation panel
        # and, when present, the interaction (tuning) panel: edges
        # estimated on discovery prompts would carry the same selection
        # bias the held-out requirement exists to prevent. Entries
        # without an integer pool_index cannot be checked and fail closed
        # rather than passing by default. Indices are additionally bound
        # to content below: each entry's names must match its claimed
        # pool index, so disjoint indices with swapped-in discovery names
        # cannot smuggle overlapping prompts past the set check.
        def _indices(panel: object) -> Optional[Set[int]]:
            indices: Set[int] = set()
            if not isinstance(panel, list):
                return None
            for entry in panel:
                if not isinstance(entry, dict):
                    return None
                idx = entry.get("pool_index")
                if not isinstance(idx, int) or isinstance(idx, bool):
                    return None
                indices.add(idx)
            return indices

        panels = {
            "discovery": _indices(discovery_panel),
            "validation": _indices(validation_panel),
        }
        interaction_panel = discovery.get("interaction_panel")
        if isinstance(interaction_panel, list) and interaction_panel:
            panels["interaction"] = _indices(interaction_panel)
        if any(v is None for v in panels.values()):
            return self._unavailable(
                "Panel entries lack integer pool_index identifiers; "
                "disjointness cannot be verified without them."
            )
        names = sorted(panels)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                overlap = panels[names[i]] & panels[names[j]]
                if overlap:
                    return self._unavailable(
                        f"{names[i].capitalize()} panel overlaps "
                        f"{names[j]} panel at pool indices: "
                        f"{sorted(overlap)}. "
                        f"This violates the held-out requirement."
                    )

        try:
            token_ids = lm.single_token_names(NAMES)
        except lm.LiveUnavailable as exc:
            return self._unavailable(str(exc))
        missing = [n for n, tid in token_ids.items() if tid is None]
        if missing:
            return self._unavailable(
                "IOI comparison tokens are not single vocabulary items: "
                + ", ".join(sorted(missing)))

        prompts = []
        for panel_entry in validation_panel:
            try:
                subj = panel_entry["subject"]
                io_name = panel_entry["io"]
            except (KeyError, TypeError):
                return self._unavailable(
                    "A validation panel entry lacks subject/io names; "
                    "its prompts cannot be reconstructed."
                )
            expected_pair = pair_for_index(panel_entry.get("pool_index"))
            if expected_pair is None or (subj, io_name) != expected_pair:
                return self._unavailable(
                    "A validation panel entry's names do not match its "
                    "claimed pool index; held-out status cannot be proven "
                    "for relabeled prompts."
                )
            clean = lm.clean_prompt(subj, io_name)
            corr = lm.corrupted_prompt(subj, io_name)
            io_id = token_ids[io_name]
            subj_id = token_ids[subj]
            if io_id is None or subj_id is None:
                return self._unavailable(
                    "A validation prompt's names are not single vocabulary "
                    "items; its baseline cannot be measured."
                )
            clean_base = lm.baseline(clean, io_id, subj_id)
            corr_base = lm.baseline(corr, io_id, subj_id)
            prompts.append({
                "subject": subj, "io": io_name, "clean": clean,
                "corrupted": corr, "io_id": io_id, "subj_id": subj_id,
                "clean_diff": clean_base["logit_diff"],
                "corrupted_diff": corr_base["logit_diff"],
                "validation_pool_index": panel_entry["pool_index"],
            })

        circuit = set()
        for label in heads:
            parsed = lm.parse_head(label)
            if parsed is not None:
                circuit.add(parsed)

        # Replication of each head effect.
        replicated = 0
        head_checks = []
        for label in heads:
            parsed = lm.parse_head(label)
            expected = effects.get(label, {}).get("mean_delta", 0.0)
            if parsed is None:
                head_checks.append({"head": label, "replicated": False,
                                    "reason": "unparseable head label"})
                continue
            expected_sign = 1 if expected > 0 else -1
            floor = max(0.10, 0.5 * abs(expected))
            passes = 0
            deltas = []
            for prompt in prompts:
                patched = lm.ablate(prompt["clean"], prompt["io_id"],
                                    prompt["subj_id"], {parsed})
                delta = patched - prompt["clean_diff"]
                deltas.append(round(delta, 4))
                sign = 1 if delta > 0 else -1
                if sign == expected_sign and abs(delta) >= floor:
                    passes += 1
            ok = passes >= 2
            replicated += 1 if ok else 0
            head_checks.append({"head": label, "replicated": ok,
                                "prompts_passed": passes,
                                "per_prompt_delta": deltas})
        replication_rate = replicated / len(heads)

        # Circuit recovery on held-out prompts.
        recoveries = []
        per_prompt_recovery = []
        for prompt in prompts:
            _, caps = lm.capture(prompt["clean"])
            patched = lm.inject(prompt["corrupted"], prompt["io_id"],
                                prompt["subj_id"], caps, circuit)
            denom = prompt["clean_diff"] - prompt["corrupted_diff"]
            faith = ((patched["logit_diff"] - prompt["corrupted_diff"]) / denom
                     if denom > 0.2 else 0.0)
            faith = max(0.0, min(1.0, faith))
            recoveries.append(faith)
            per_prompt_recovery.append({
                "subject": prompt["subject"],
                "io": prompt["io"],
                "pool_index": prompt.get("validation_pool_index"),
                "clean_diff": round(prompt["clean_diff"], 4),
                "corrupted_diff": round(prompt["corrupted_diff"], 4),
                "patched_diff": round(patched["logit_diff"], 4),
                "recovery": round(faith, 4),
            })
        recovery = mean(recoveries) if recoveries else 0.0
        recovery_stats = _mean_sd(recoveries)

        # Necessity (minimality) on held-out prompts.
        necessary = 0
        for label in heads:
            parsed = lm.parse_head(label)
            if parsed is None:
                continue
            hits = 0
            for prompt in prompts:
                if abs(prompt["clean_diff"]) <= 0.2:
                    continue
                patched = lm.ablate(prompt["clean"], prompt["io_id"],
                                    prompt["subj_id"], {parsed})
                if abs(patched - prompt["clean_diff"]) >= 0.10 * abs(prompt["clean_diff"]):
                    hits += 1
            if hits >= 2:
                necessary += 1
        minimality = necessary / len(heads)

        checks = [
            {"check": "replication_rate>=0.75", "value": round(replication_rate, 4),
              "passed": replication_rate >= REPLICATION_FLOOR,
              "n_heads": len(heads),
              "n_prompts_per_head": len(prompts),
              "wilson95": [_wilson_interval(replicated, len(heads))["lo"],
                           _wilson_interval(replicated, len(heads))["hi"]]},
            {"check": "held_out_recovery>=0.50", "value": round(recovery, 4),
              "passed": recovery >= RECOVERY_FLOOR,
              "n_prompts": len(prompts),
              "recovery_sd": recovery_stats["sd"],
              "recovery_range": [recovery_stats["min"], recovery_stats["max"]]},
            {"check": "minimality>=0.50", "value": round(minimality, 4),
              "passed": minimality >= MINIMALITY_FLOOR,
              "n_heads": len(heads),
              "n_prompts_per_head": len(prompts),
              "wilson95": [_wilson_interval(necessary, len(heads))["lo"],
                           _wilson_interval(necessary, len(heads))["hi"]]},
        ]
        confidence = round(0.5 + 0.5 * mean(
            [replication_rate, min(1.0, recovery), minimality]), 4)
        validated = all(check["passed"] for check in checks) and confidence >= 0.85
        replication_interval = _wilson_interval(replicated, len(heads))
        minimality_interval = _wilson_interval(necessary, len(heads))
        scope = (
            f"Held-out re-measurement on {len(prompts)} disjoint name-pair "
            f"prompts against {len(heads)} discovered heads of gpt2-small "
            f"via deterministic forward passes. This verdict covers only "
            f"these prompts, this model, and this run — not broad "
            f"scientific replication."
        )
        limitations = [
            "Small held-out panel: "
            f"{len(prompts)} prompts, so a deterministic threshold pass "
            "is weak evidence of generalization (Wilson 95% intervals "
            "reported in confidence.uncertainty).",
            "Single model (gpt2-small) and deterministic forward passes: "
            "no seed variation, no independent model, no prompt-composition "
            "sensitivity analysis was run.",
            "The automated rule check is not human peer review: see "
            "peer_review.is_human_peer_review (False) and the empty "
            "human_peer_review slot.",
        ]

        # Build the validation result payload (without attested/provenance/measurement_record)
        validation_result = {
            "discovery_id": discovery.get("discovery_id", ""),
            "status": "completed",
            "field_provenance": field_map(
                ("discovery_id", "status", "validated", "confidence",
                  "peer_review", "human_peer_review", "checks", "head_checks",
                  "validation_panel", "per_prompt_recovery", "sample_sizes",
                  "scope", "limitations"),
                "live",
            ),
            "validation_eligible": True,
            "publication_eligible": True,
            "validated": validated,
            "confidence": {
                "confidence_score": confidence,
                "method": ("0.5 + 0.5 * mean(replication_rate, recovery, "
                           "minimality) over held-out prompts"),
                "is_heuristic": True,
                "interpretation": (
                    "Uncalibrated aggregate of three pass rates over "
                    f"{len(prompts)} held-out prompts; not a calibrated "
                    "probability of correctness."),
                "n_held_out_prompts": len(prompts),
                "n_heads": len(heads),
                "uncertainty": {
                    "replication_rate_wilson95": [replication_interval["lo"],
                                                  replication_interval["hi"]],
                    "minimality_wilson95": [minimality_interval["lo"],
                                            minimality_interval["hi"]],
                    "recovery_range": [recovery_stats["min"],
                                       recovery_stats["max"]],
                    "recovery_sd": recovery_stats["sd"],
                },
            },
            "peer_review": {
                "kind": "automated-rule-check",
                "is_human_peer_review": False,
                "reviewer": "automated-live-checks (rule-based, not human peer review)",
                "decision": "Accept" if validated else "Revise",
                "checks": checks,
                "n_held_out_prompts": len(prompts),
                "scope": scope,
            },
            "human_peer_review": None,
            "replication_rate": round(replication_rate, 4),
            "held_out_recovery": round(recovery, 4),
            "minimality": round(minimality, 4),
            "head_checks": head_checks,
            "method": self.method,
            "n_held_out_prompts": len(prompts),
            "per_prompt_recovery": per_prompt_recovery,
            "sample_sizes": {
                "held_out_prompts": len(prompts),
                "heads": len(heads),
                "prompts_per_head_check": len(prompts),
            },
            "scope": scope,
            "limitations": limitations,
            "seed_info": {
                "seeds_varied": False,
                "deterministic": True,
                "note": ("Forward passes are deterministic given weights "
                         "and prompts; no seed variation or repeat-run "
                         "sensitivity analysis was performed."),
            },
            # Panel composition for disjointness verification
            "validation_panel": validation_panel,
        }

        # Attest the validation measurement: binds model, held-out inputs, validation outputs, execution context
        # Fingerprint comes from the engine weights this run measured
        # through (same weights as discovery). Commitments cover retained
        # fields so the policy boundary can recompute them.
        from backend.services import gpt2_engine as _engine

        attested_result = attest_measurement(
            validation_result,
            model_loaded="gpt2-small",
            model_requested="gpt2-small",
            model_content_hash=_engine.weights_fingerprint(),
            input_fields=["validation_panel", "discovery_id"],
            output_fields=["head_checks", "peer_review", "replication_rate",
                           "held_out_recovery", "minimality", "validated",
                           "per_prompt_recovery"],
        )
        return attested_result

    @staticmethod
    def _unavailable(reason: str) -> Dict[str, Any]:
        return {
            "status": "unavailable",
            "provenance": "unavailable",
            "field_provenance": field_map(
                ("status", "validated", "reason"), "unavailable"),
            "validation_eligible": False,
            "publication_eligible": False,
            "validated": False,
            "reason": reason,
        }
