"""Live causal discovery executor for the IOI circuit in GPT-2 Small.

Method (all quantities measured from the loaded model, nothing hardcoded):
  1. Baseline clean/corrupted logit differences over a fixed prompt panel.
  2. Full 144-head zero-ablation screen on one prompt, then causal
     verification of the top candidates on every panel prompt.
  3. Circuit = verified heads with consistent sign and mean |effect| floor.
  4. Edges = pairwise zero-ablation interactions
     (|ΔAB − (ΔA + ΔB)| >= edge floor), directed earlier-layer → later-layer.
     Same-layer pairs are reported as unresolved interactions, not edges.
  5. Recovery = clean-to-corrupted activation injection of the circuit.

The result carries explicit live provenance.  Downstream eligibility is
*derived*, not asserted: a measurement being live does not make it scientifically
adequate, and this module no longer claims otherwise.  See `_adequacy`.

When no live model is connected the executor reports itself unavailable and
callers fail closed.
"""

from __future__ import annotations

import hashlib
import json
from statistics import mean
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.agents.evidence_policy import field_map
from backend.core.provenance import set_evidence_level

from . import live_measure as lm

NAMES = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"]
PAIR_POOL: List[Tuple[str, str]] = [
    (a, b) for a in NAMES for b in NAMES if a != b
]

EFFECT_FLOOR = 0.15
EDGE_FLOOR = 0.15
MAX_HEADS = 10
SCREEN_CANDIDATES = 20
EDGE_PAIR_HEADS = 8

#: Minimum prompts before the result can support a validation decision.
#: Matches ``ioi_pipeline.MIN_PROMPTS_FOR_MINIMALITY``: below ten prompts the
#: harness has not established that a surviving head is doing anything rather
#: than fitting one prompt.
MIN_PROMPTS_FOR_VALIDATION = 10

#: Minimum prompts before the *pairwise interaction* stage is treated as
#: measured. It runs on ``prompts[:2]`` by design -- it is ``EDGE_PAIR_HEADS``
#: choose-two ablations, so its cost is quadratic in the pool and quadratic in
#: prompts, and two was chosen to keep a run affordable. Two observations is
#: enough to compute a mean and not enough to support a claim about it.
MIN_PROMPTS_FOR_INTERACTION = 5


def _discovery_id(hypothesis_statement: str, prompts: List[Dict[str, Any]],
                  model_id: str) -> str:
    """A stable identity for a discovery, reproducible across processes.

    This was ``abs(hash((hypothesis_statement, len(prompts), tuple(...))))``.
    Python salts ``hash`` of strings per process unless ``PYTHONHASHSEED`` is
    pinned, so the same discovery of the same hypothesis on the same prompts got
    a different ``discovery_id`` on every interpreter start -- measured:
    0c49e6ce, then 78e1c9a0, then 1cca0669, for byte-identical inputs.

    That defeats the point of an identifier. ``recall(discovery_id)`` cannot find
    a previous run, two runs of the same hypothesis cannot be deduplicated, and a
    record cannot be cited by the ID it was given. SHA-256 over a canonical JSON
    encoding is stable across processes, platforms and Python versions.
    """
    payload = json.dumps(
        {
            "hypothesis": hypothesis_statement,
            "model_id": model_id,
            "prompts": [
                {"clean": p["clean"], "corrupted": p["corrupted"],
                 "io": p["io"], "subject": p["subject"]}
                for p in prompts
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "disc_live_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _adequacy(n_prompts: int, n_interaction_prompts: int,
              faithfulness: Optional[float]) -> Dict[str, Any]:
    """Derive the eligibility ladder from what was actually measured.

    Previously this module returned ``validation_eligible: True`` and
    ``publication_eligible: True`` unconditionally, with defaults of four prompts
    and two prompts for the interaction stage. So a four-prompt run whose edges
    were averaged from two observations declared itself fit for publication. The
    flags were not a judgement about the result; they were constants next to real
    measurements, which is the most confusing place for a constant to sit.

    The ladder, and what each rung means here:

        live                 -- weights were loaded and forward passes ran
        measured             -- a faithfulness figure was computed at all
        statistically_adequate -- enough prompts for the screening stage
        interaction_adequate  -- enough prompts behind the pairwise edges
        validation_eligible  -- statistically_adequate
        publication_eligible -- both of the above

    ``replicated`` is deliberately absent: nothing in this module runs a second
    model or a second seed, so it could only ever be asserted, never measured.
    """
    measured = faithfulness is not None
    statistically_adequate = n_prompts >= MIN_PROMPTS_FOR_VALIDATION
    interaction_adequate = n_interaction_prompts >= MIN_PROMPTS_FOR_INTERACTION

    reasons = []
    if not measured:
        reasons.append("no faithfulness was computed")
    if not statistically_adequate:
        reasons.append(
            f"{n_prompts} prompts, below the {MIN_PROMPTS_FOR_VALIDATION} needed "
            f"for a validation decision"
        )
    if not interaction_adequate:
        reasons.append(
            f"pairwise edges were averaged from {n_interaction_prompts} prompts, "
            f"below the {MIN_PROMPTS_FOR_INTERACTION} needed to support a claim "
            f"about them"
        )

    validation_eligible = measured and statistically_adequate
    publication_eligible = validation_eligible and interaction_adequate

    return {
        "live": True,
        "measured": measured,
        "statistically_adequate": statistically_adequate,
        "interaction_adequate": interaction_adequate,
        "replicated": False,
        "replicated_reason": (
            "not measured: this module runs one model at one seed and performs "
            "no replication"
        ),
        "validation_eligible": validation_eligible,
        "publication_eligible": publication_eligible,
        "min_prompts_for_validation": MIN_PROMPTS_FOR_VALIDATION,
        "min_prompts_for_interaction": MIN_PROMPTS_FOR_INTERACTION,
        "n_prompts": n_prompts,
        "n_interaction_prompts": n_interaction_prompts,
        "ineligible_because": reasons or None,
    }


_registry: Dict[str, Dict[str, Any]] = {}
_REGISTRY_MAX = 128


def remember(result: Dict[str, Any]) -> None:
    disc_id = result.get("discovery_id")
    if isinstance(disc_id, str) and disc_id:
        _registry[disc_id] = result
        while len(_registry) > _REGISTRY_MAX:
            _registry.pop(next(iter(_registry)))


def recall(discovery_id: str) -> Optional[Dict[str, Any]]:
    result = _registry.get(discovery_id)
    return dict(result) if isinstance(result, dict) else None


def prompt_panel(n_prompts: int, offset: int = 0) -> List[Tuple[str, str]]:
    """Deterministic ordered subject/IO name pairs (no randomness)."""
    if n_prompts <= 0:
        return []
    panel = []
    for i in range(n_prompts):
        panel.append(PAIR_POOL[(offset + i) % len(PAIR_POOL)])
    return panel


class LiveIOIDiscovery:
    """Zero-ablation + injection discovery over the live GPT-2 Small weights."""

    method = ("zero-ablation head screening with causal verification, "
              "pairwise ablation-interaction edges, and clean-to-corrupted "
              "injection recovery")

    #: Recorded in the result and mixed into the discovery identity, so that the
    #: same hypothesis screened against a different model is a different
    #: discovery rather than a collision.
    model_id = "gpt2-small"

    @staticmethod
    def available() -> bool:
        try:
            lm._engine()
            return True
        except lm.LiveUnavailable:
            return False

    def run(self, hypothesis_statement: str = "", n_prompts: int = 4,
            max_heads: int = MAX_HEADS,
            n_interaction_prompts: int = 2) -> Dict[str, Any]:
        """Discover the IOI circuit from live weights.

        `n_interaction_prompts` is the number of prompts the pairwise
        interaction stage is averaged over. It defaults to 2 -- that stage is
        choose-two over `EDGE_PAIR_HEADS`, so its cost is quadratic in both the
        pool and the prompt count, and 2 keeps a run affordable.

        2 is enough to compute a mean and not enough to support a claim about it,
        which is why `publication_eligible` comes back False at any
        `n_prompts` until a caller passes at least
        `MIN_PROMPTS_FOR_INTERACTION`. Making it a parameter rather than raising
        the default silently keeps the cost decision with the caller, and leaves
        the honest "cannot support a publication claim" state as the default
        rather than quietly paying for a different experiment.
        """
        try:
            return self._run(hypothesis_statement, n_prompts, max_heads,
                             n_interaction_prompts)
        except lm.LiveUnavailable as exc:
            return {
                "status": "unavailable",
                "provenance": "unavailable",
                "field_provenance": field_map(
                    ("status", "reason"), "unavailable"),
                "measured": False,
                "replicated": False,
                "validation_eligible": False,
                "publication_eligible": False,
                "reason": str(exc),
            }

    def _run(self, hypothesis_statement: str, n_prompts: int,
             max_heads: int, n_interaction_prompts: int = 2) -> Dict[str, Any]:
        token_ids = lm.single_token_names(NAMES)
        missing = sorted(name for name, tid in token_ids.items() if tid is None)
        if missing:
            raise lm.LiveUnavailable(
                "IOI comparison tokens are not single vocabulary items: "
                + ", ".join(missing))
        n_layers, n_heads, _ = lm.dims()
        pairs = prompt_panel(max(1, n_prompts))

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
                "subject": subj, "io": io_name,
                "clean": clean, "corrupted": corr,
                "io_id": io_id, "subj_id": subj_id,
                "clean_top1": clean_base["top1"],
                "clean_diff": round(clean_base["logit_diff"], 4),
                "corrupted_diff": round(corr_base["logit_diff"], 4),
            })

        # Stage A: full-head screen on the first prompt.
        screen = prompts[0]
        screen_effects: Dict[Tuple[int, int], float] = {}
        for layer in range(n_layers):
            for head in range(n_heads):
                patched = lm.ablate(screen["clean"], screen["io_id"],
                                    screen["subj_id"], {(layer, head)})
                screen_effects[(layer, head)] = patched - screen["clean_diff"]
        ranked = sorted(screen_effects, key=lambda k: -abs(screen_effects[k]))
        candidates = ranked[:SCREEN_CANDIDATES]

        # Stage B: verify candidates on every panel prompt.
        verified: Dict[Tuple[int, int], List[float]] = {c: [] for c in candidates}
        for prompt in prompts:
            for cand in candidates:
                patched = lm.ablate(prompt["clean"], prompt["io_id"],
                                    prompt["subj_id"], {cand})
                verified[cand].append(patched - prompt["clean_diff"])

        # Stage C: select the circuit.
        required_signs = max(2, int(0.75 * len(prompts)))
        scored = []
        for cand, deltas in verified.items():
            signs = [1 if d > 0 else -1 for d in deltas]
            consistent = (max(signs.count(1), signs.count(-1))
                          >= required_signs)
            magnitude = mean(abs(d) for d in deltas)
            if consistent and magnitude >= EFFECT_FLOOR:
                scored.append((cand, mean(deltas), magnitude))
        scored.sort(key=lambda item: -item[2])
        circuit = [cand for cand, _, _ in scored[:max(1, max_heads)]]
        head_effects = [
            {"head": lm.head_label(layer, head),
             "mean_delta": round(mean_delta, 4),
             "mean_abs_delta": round(magnitude, 4),
             "per_prompt_delta": [round(d, 4) for d in verified[(layer, head)]]}
            for (layer, head), mean_delta, magnitude in scored[:max(1, max_heads)]
        ]

        # Stage D: pairwise interaction edges (earlier layer -> later layer).
        edge_pool = [cand for cand, _, _ in scored[:EDGE_PAIR_HEADS]]
        pair_prompts = prompts[:max(1, min(n_interaction_prompts, len(prompts)))]
        singles: Dict[Tuple[int, int], List[float]] = {
            cand: verified[cand][:len(pair_prompts)] for cand in edge_pool}
        edges = []
        unresolved = []
        for i, first in enumerate(edge_pool):
            for second in edge_pool[i + 1:]:
                interactions = []
                for idx, prompt in enumerate(pair_prompts):
                    both = lm.ablate(prompt["clean"], prompt["io_id"],
                                     prompt["subj_id"], {first, second})
                    base = prompt["clean_diff"]
                    delta_both = both - base
                    interaction = (delta_both
                                   - singles[first][idx] - singles[second][idx])
                    interactions.append(interaction)
                strength = mean(abs(v) for v in interactions)
                record = {
                    "source": lm.head_label(*first),
                    "target": lm.head_label(*second),
                    "mean_interaction": round(mean(interactions), 4),
                    "mean_abs_interaction": round(strength, 4),
                }
                if strength < EDGE_FLOOR:
                    continue
                if first[0] == second[0]:
                    unresolved.append(record)
                elif first[0] < second[0]:
                    edges.append(record)
                else:
                    edges.append({
                        "source": record["target"],
                        "target": record["source"],
                        "mean_interaction": record["mean_interaction"],
                        "mean_abs_interaction": record["mean_abs_interaction"],
                    })

        # Stage E: injection recovery of the selected circuit.
        faithfulness: List[float] = []
        completeness: List[float] = []
        for prompt in prompts:
            _, caps = lm.capture(prompt["clean"])
            patched = lm.inject(prompt["corrupted"], prompt["io_id"],
                                prompt["subj_id"], caps, set(circuit))
            clean_diff = prompt["clean_diff"]
            corr_diff = prompt["corrupted_diff"]
            rec_diff = patched["logit_diff"]
            denom = clean_diff - corr_diff
            faith = ((rec_diff - corr_diff) / denom
                     if denom > 0.2 else 0.0)
            faithfulness.append(max(0.0, min(1.0, faith)))
            compl = (rec_diff / clean_diff if clean_diff > 0.2 else 0.0)
            completeness.append(max(0.0, min(1.0, compl)))

        disc_id = _discovery_id(hypothesis_statement, prompts, self.model_id)
        mean_faithfulness = round(mean(faithfulness), 4) if faithfulness else 0.0
        adequacy = _adequacy(len(prompts), len(pair_prompts), mean_faithfulness)

        # Evidence level: INTERVENTIONAL (causal intervention via ablation/injection)
        # Upgraded to CAUSALLY_VALIDATED if validation-eligible
        evidence_level = "INTERVENTIONAL"
        if adequacy.get("validation_eligible"):
            evidence_level = "CAUSALLY_VALIDATED"

        result = {
            "discovery_id": disc_id,
            "status": "completed",
            "provenance": "live",
            # Attested here because this module ran the measurements above:
            # per-prompt capture/inject on live weights, not a relabel of
            # someone else's numbers. Wrappers must propagate this, never
            # invent it.
            "attested": True,
            "field_provenance": field_map(
                ("status", "discovery_id", "method", "model_id",
                 "n_prompts", "heads", "head_effects", "edges",
                 "unresolved_interactions", "faithfulness", "completeness",
                 "baselines"),
                "live",
            ),
            **adequacy,
            "method": self.method,
            "model_id": self.model_id,
            "n_prompts": len(prompts),
            "heads": [lm.head_label(layer, head) for layer, head in circuit],
            "head_effects": head_effects,
            "edges": edges,
            "unresolved_interactions": unresolved,
            "faithfulness": mean_faithfulness,
            "completeness": round(mean(completeness), 4) if completeness else 0.0,
            "per_prompt_recovery": [
                {"subject": prompt["subject"], "io": prompt["io"],
                 "faithfulness": round(faith, 4), "completeness": round(comp, 4)}
                for prompt, faith, comp in zip(prompts, faithfulness, completeness)
            ],
            "baselines": prompts,
            "discovery_provenance": "live",
            "synthetic_fields": [],
            "evidence_level": evidence_level,
        }
        remember(result)
        return result
