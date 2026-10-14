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
from typing import Any, Dict, List, Optional, Tuple

from backend.agents.evidence_policy import field_map
from backend.core.provenance import attest_measurement

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

#: Number of held-out prompts used for validation. The validation panel is
#: allocated from pool entries discovery did not use (see
#: `_allocate_validation_panel`), never by fixed offset: with a cyclic
#: pool, an offset that once meant "held out" wraps into the discovery
#: range as soon as the discovery count grows.
HELD_OUT_PROMPTS = 3


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


def prompt_panel(n_prompts: int, offset: int = 0) -> List[Dict[str, Any]]:
    """Deterministic ordered subject/IO name pairs with their pool indices.

    Returns a list of dicts with keys: "subject", "io", "pool_index".
    This makes panel composition auditable and enables disjointness checks.
    """
    if n_prompts <= 0:
        return []
    panel = []
    pool_size = len(PAIR_POOL)
    for i in range(n_prompts):
        pool_index = (offset + i) % pool_size
        subj, io = PAIR_POOL[pool_index]
        panel.append({"subject": subj, "io": io, "pool_index": pool_index})
    return panel


def _allocate_panel(used_indices: object, n: int) -> Optional[List[Dict[str, Any]]]:
    """First consecutive run of `n` pool entries none of which is used.

    `used_indices` is the set of `pool_index` values already taken by
    earlier panels. Scanning pool order keeps allocation deterministic
    across processes. Returns None when no disjoint run of the requested
    size fits — the caller must then fail closed (emit no panel), never
    emit an overlapping one. Non-integer entries (including bools) are
    not valid indices and are ignored.
    """
    if not isinstance(used_indices, (set, frozenset)) or n <= 0:
        return None
    used = {i for i in used_indices
            if isinstance(i, int) and not isinstance(i, bool)}
    pool_size = len(PAIR_POOL)
    for start in range(pool_size):
        candidate = [(start + i) % pool_size for i in range(n)]
        if all(c not in used for c in candidate):
            return [
                {"subject": PAIR_POOL[c][0], "io": PAIR_POOL[c][1],
                 "pool_index": c}
                for c in candidate
            ]
    return None


def _panel_indices(panel: object) -> Optional[set]:
    """Integer `pool_index` set for a panel, or None if unverifiable."""
    if not isinstance(panel, list) or not panel:
        return None
    indices = set()
    for entry in panel:
        if not isinstance(entry, dict):
            return None
        idx = entry.get("pool_index")
        if not isinstance(idx, int) or isinstance(idx, bool):
            return None
        indices.add(idx)
    return indices


def pair_for_index(pool_index: int) -> Optional[Tuple[str, str]]:
    """Canonical (subject, io) pair for a pool index, or None.

    The single authority mapping indices to content. Validators use it
    to bind a panel entry's recorded names to its claimed index, so a
    forged entry cannot borrow a disjoint index while naming an
    overlapping prompt.
    """
    if not isinstance(pool_index, int) or isinstance(pool_index, bool):
        return None
    if 0 <= pool_index < len(PAIR_POOL):
        return PAIR_POOL[pool_index]
    return None


def _allocate_validation_panel(
    discovery_panel: List[Dict[str, Any]],
    n_held_out: int = HELD_OUT_PROMPTS,
) -> Optional[List[Dict[str, Any]]]:
    """Allocate a held-out panel disjoint from discovery, or None.

    Scans the pool in order for the first run of `n_held_out` consecutive
    entries none of which discovery used. Disjointness is decided by the
    recorded `pool_index` sets — not by an offset convention — so a
    discovery of any size gets a genuinely fresh panel or an explicit
    refusal. When discovery already covers the pool (or leaves too few
    consecutive free entries, e.g. 28 of 30 used with 3 required), there
    is no disjoint panel to emit and this returns None instead of a
    self-overlapping one; the validator then fails closed.
    """
    if not isinstance(discovery_panel, list) or n_held_out <= 0:
        return None
    used = _panel_indices(discovery_panel)
    if used is None:
        return None
    return _allocate_panel(used, n_held_out)


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

        # Discovery panel: starts at offset 0, uses n_prompts entries
        discovery_panel = prompt_panel(max(1, n_prompts), offset=0)

        def _measure_baselines(panel: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Clean/corrupted baselines for a panel's entries."""
            measured = []
            for panel_entry in panel:
                subj = panel_entry["subject"]
                io_name = panel_entry["io"]
                clean = lm.clean_prompt(subj, io_name)
                corr = lm.corrupted_prompt(subj, io_name)
                io_id = token_ids[io_name]
                subj_id = token_ids[subj]
                assert io_id is not None and subj_id is not None
                clean_base = lm.baseline(clean, io_id, subj_id)
                corr_base = lm.baseline(corr, io_id, subj_id)
                measured.append({
                    "subject": subj, "io": io_name,
                    "pool_index": panel_entry["pool_index"],
                    "clean": clean, "corrupted": corr,
                    "io_id": io_id, "subj_id": subj_id,
                    "clean_top1": clean_base["top1"],
                    "clean_diff": round(clean_base["logit_diff"], 4),
                    "corrupted_diff": round(corr_base["logit_diff"], 4),
                })
            return measured

        prompts = _measure_baselines(discovery_panel)

        # Interaction (tuning) panel: a disjoint panel for the pairwise
        # edge stage, allocated from pool entries neither discovery nor
        # validation uses. Estimating interactions on discovery prompts
        # would carry the same selection bias the held-out requirement
        # exists to prevent. When no disjoint panel fits (discovery
        # already covers the pool), fall back to the discovery prefix —
        # the pre-existing behavior — and record what was actually used
        # in `interaction_panel` so the manifest stays honest.
        validation_panel = _allocate_validation_panel(discovery_panel)
        _used_for_interaction = set(_panel_indices(discovery_panel) or set())
        if validation_panel is not None:
            _used_for_interaction |= set(_panel_indices(validation_panel) or set())
        interaction_panel = _allocate_panel(
            _used_for_interaction, max(1, n_interaction_prompts))
        if interaction_panel is not None:
            pair_source = _measure_baselines(interaction_panel)
        else:
            # No disjoint tuning panel fits: fall back to the discovery
            # prefix (the pre-existing behavior) and record those
            # identifiers, so the manifest shows what was measured.
            pair_source = prompts[:max(1, min(n_interaction_prompts,
                                              len(prompts)))]
            interaction_panel = [
                {"subject": p["subject"], "io": p["io"],
                 "pool_index": p["pool_index"]}
                for p in pair_source
            ]

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
        # Measured on the interaction panel (`pair_source`), not on the
        # discovery prompts: reusing discovery deltas here would estimate
        # interactions on the same examples that selected the heads.
        edge_pool = [cand for cand, _, _ in scored[:EDGE_PAIR_HEADS]]
        pair_prompts = pair_source
        singles: Dict[Tuple[int, int], List[float]] = {}
        for cand in edge_pool:
            cand_deltas = []
            for prompt in pair_prompts:
                patched = lm.ablate(prompt["clean"], prompt["io_id"],
                                     prompt["subj_id"], {cand})
                cand_deltas.append(patched - prompt["clean_diff"])
            singles[cand] = cand_deltas
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
        if validation_panel is None:
            # No disjoint held-out panel exists for this discovery size,
            # so no validation decision can ever be supported, regardless
            # of prompt counts. Say so explicitly rather than leaving
            # count-derived eligibility flags asserting the opposite.
            adequacy["validation_eligible"] = False
            adequacy["publication_eligible"] = False
            adequacy["ineligible_because"] = (
                (adequacy["ineligible_because"] or []) + [
                    f"no disjoint {HELD_OUT_PROMPTS}-prompt validation "
                    f"panel fits after {len(prompts)} discovery prompts "
                    f"in a pool of {len(PAIR_POOL)}"
                ]
            )

        # Evidence level: INTERVENTIONAL (causal intervention via ablation/injection)
        # Upgraded to CAUSALLY_VALIDATED if validation-eligible
        evidence_level = "INTERVENTIONAL"
        if adequacy.get("validation_eligible"):
            evidence_level = "CAUSALLY_VALIDATED"

        result = {
            "discovery_id": disc_id,
            "status": "completed",
            # Attested here because this module ran the measurements above:
            # per-prompt capture/inject on live weights, not a relabel of
            # someone else's numbers. Wrappers must propagate this, never
            # invent it. attest_measurement will set provenance and attested.
            "field_provenance": field_map(
                ("status", "discovery_id", "method", "model_id",
                 "n_prompts", "heads", "head_effects", "edges",
                 "unresolved_interactions", "faithfulness", "completeness",
                 "baselines", "discovery_panel", "validation_panel",
                 "interaction_panel"),
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
                 "pool_index": prompt["pool_index"],
                 "faithfulness": round(faith, 4), "completeness": round(comp, 4)}
                for prompt, faith, comp in zip(prompts, faithfulness, completeness)
            ],
            "baselines": prompts,
            "discovery_provenance": "live",
            "synthetic_fields": [],
            "evidence_level": evidence_level,
            # Panel composition for disjointness verification. Both the
            # validation and the interaction (tuning) panels are
            # allocated, not offset-derived: when no disjoint panel fits
            # (discovery already covers the pool), validation is None and
            # validation fails closed rather than remeasuring discovery
            # prompts under a held-out label. interaction_panel records
            # the prompts the edge stage actually measured — the
            # allocated panel, or the discovery prefix it fell back to.
            "discovery_panel": discovery_panel,
            "validation_panel": validation_panel,
            "interaction_panel": interaction_panel,
        }

        # Attest the discovery measurement: binds model, inputs, outputs, execution context
        # attest_measurement will set provenance="live" and attested=True.
        #
        # The model fingerprint comes from the engine weights this run
        # actually measured through (lm runs every forward pass on
        # backend.services.gpt2_engine's loaded model), not from a model
        # name and not from a second adapter copy: fingerprinting weights
        # other than the measured ones would bind the wrong artifact.
        # Commitments cover retained fields only, so recomputation at the
        # policy boundary sees exactly what creation hashed.
        from backend.services import gpt2_engine as _engine

        result = attest_measurement(
            result,
            model_loaded=self.model_id,
            model_requested=self.model_id,
            model_content_hash=_engine.weights_fingerprint(),
            input_fields=["baselines", "discovery_panel",
                          "validation_panel", "interaction_panel"],
            output_fields=["heads", "head_effects", "edges",
                           "unresolved_interactions", "faithfulness",
                           "completeness"],
        )
        remember(result)
        return result
