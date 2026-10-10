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

The automated checks object is rule-based, not human peer review, and says
so explicitly.  Confidence is a direct function of the three measured
pass rates, documented in the payload.
"""

from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List, Tuple

from backend.agents.evidence_policy import field_map

from backend.interpretability.discovery import live_measure as lm
from backend.interpretability.discovery.live_discovery import NAMES, prompt_panel


HELD_OUT_OFFSET = 4
HELD_OUT_PROMPTS = 3
REPLICATION_FLOOR = 0.75
RECOVERY_FLOOR = 0.50
MINIMALITY_FLOOR = 0.50


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
        try:
            token_ids = lm.single_token_names(NAMES)
        except lm.LiveUnavailable as exc:
            return self._unavailable(str(exc))
        missing = [n for n, tid in token_ids.items() if tid is None]
        if missing:
            return self._unavailable(
                "IOI comparison tokens are not single vocabulary items: "
                + ", ".join(sorted(missing)))

        pairs = prompt_panel(HELD_OUT_PROMPTS, offset=HELD_OUT_OFFSET)
        prompts = []
        for subj, io_name in pairs:
            clean = lm.clean_prompt(subj, io_name)
            corr = lm.corrupted_prompt(subj, io_name)
            io_id = token_ids[io_name]
            subj_id = token_ids[subj]
            assert io_id is not None and subj_id is not None
            clean_base = lm.baseline(clean, io_id, subj_id)
            corr_base = lm.baseline(corr, io_id, subj_id)
            prompts.append({
                "subject": subj, "io": io_name, "clean": clean,
                "corrupted": corr, "io_id": io_id, "subj_id": subj_id,
                "clean_diff": clean_base["logit_diff"],
                "corrupted_diff": corr_base["logit_diff"],
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
        for prompt in prompts:
            _, caps = lm.capture(prompt["clean"])
            patched = lm.inject(prompt["corrupted"], prompt["io_id"],
                                prompt["subj_id"], caps, circuit)
            denom = prompt["clean_diff"] - prompt["corrupted_diff"]
            faith = ((patched["logit_diff"] - prompt["corrupted_diff"]) / denom
                     if denom > 0.2 else 0.0)
            recoveries.append(max(0.0, min(1.0, faith)))
        recovery = mean(recoveries) if recoveries else 0.0

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
             "passed": replication_rate >= REPLICATION_FLOOR},
            {"check": "held_out_recovery>=0.50", "value": round(recovery, 4),
             "passed": recovery >= RECOVERY_FLOOR},
            {"check": "minimality>=0.50", "value": round(minimality, 4),
             "passed": minimality >= MINIMALITY_FLOOR},
        ]
        confidence = round(0.5 + 0.5 * mean(
            [replication_rate, min(1.0, recovery), minimality]), 4)
        validated = all(check["passed"] for check in checks) and confidence >= 0.85
        return {
            "discovery_id": discovery.get("discovery_id", ""),
            "status": "completed",
            "provenance": "live",
            # Attested here: every check above re-measured on held-out prompts
            # through live forward passes. Wrappers propagate, never invent.
            "attested": True,
            "field_provenance": field_map(
                ("discovery_id", "status", "validated", "confidence",
                 "peer_review", "checks", "head_checks"),
                "live",
            ),
            "validation_eligible": True,
            "publication_eligible": True,
            "validated": validated,
            "confidence": {
                "confidence_score": confidence,
                "method": ("0.5 + 0.5 * mean(replication_rate, recovery, "
                           "minimality) over held-out prompts"),
            },
            "peer_review": {
                "reviewer": "automated-live-checks (rule-based, not human peer review)",
                "decision": "Accept" if validated else "Revise",
                "checks": checks,
            },
            "replication_rate": round(replication_rate, 4),
            "held_out_recovery": round(recovery, 4),
            "minimality": round(minimality, 4),
            "head_checks": head_checks,
            "method": self.method,
        }

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
