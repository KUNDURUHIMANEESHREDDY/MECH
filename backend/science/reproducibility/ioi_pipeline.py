"""IOI Reproduction Pipeline — Methodological Audit & High-Fidelity Fix.

Reproduces Wang et al. 2022.
Methodology (Refined):
1. ABC vs ABB prompt templates (Proper causal corruption)
2. Head-level patching using explicitly labeled reference coordinates only in mock mode
3. Full metadata traces (Logit distribution before/after)
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

from ..models.gpt2_adapter import GPT2Adapter
from .dataset_versioning import DatasetVersioningEngine
from .reproducibility_report import ReproducibilityReportEngine

# Canonical IOI prompt templates (Wang et al. 2022)
# ABB: When Alice and Bob went... Alice gave -> Bob (Incorrect/Corrupted)
# ABC: When Alice and Bob went... Charlie gave -> ... (Control)
_NAMES = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"]

# Reference-only fixtures. These are published-paper examples, not results
# discovered by this pipeline, and must never be treated as live evidence.
REFERENCE_ONLY_DISCOVERED_HEADS = (
    "L9H6", "L9H9", "L10H0", "L10H7", "L7H3", "L8H6",
    "L5H1", "L5H5", "L0H1", "L0H10",
)
REFERENCE_ONLY_DISCOVERED_EDGES = (
    ("L0H1", "L5H1"),
    ("L5H1", "L7H3"),
    ("L7H3", "L9H9"),
)
REFERENCE_ONLY_FAITHFULNESS = 0.880
REFERENCE_ONLY_PATCH_HEAD = (9, 9)

# Minimality is a majority-vote statistic over per-head ablations. Below this
# many usable prompts the vote is noise, and a ratio computed from it looks like
# a finding ("0.9 -- 90% of heads necessary") while measuring nothing. It is
# reported as unmeasured instead.
MIN_PROMPTS_FOR_MINIMALITY = 10


def _field_map(fields, provenance: str):
    return {str(field): provenance for field in fields}


# IOI prompt frames. A single template measures one syntactic pattern, so any
# result from it is really a statement about that frame and not about
# indirect-object identification generally. The pipeline cycles through these so
# a finding can be checked for consistency across surface forms.
#
# Every frame keeps the ABB corruption: the clean prompt's giver is the subject,
# and the corrupted prompt's giver is the indirect object, so the target
# becomes distributionally wrong in the corrupted run.
_IOI_FRAMES = (
    {"lead": "When ", "place": "store", "verb": "gave", "item": "drink"},
    {"lead": "Then ", "place": "park", "verb": "gave", "item": "ball"},
    {"lead": "After that, ", "place": "library", "verb": "offered",
     "item": "book"},
    {"lead": "While ", "place": "kitchen", "verb": "passed", "item": "towel"},
    {"lead": "Yesterday ", "place": "market", "verb": "bought", "item": "apple"},
    {"lead": "Later, ", "place": "garden", "verb": "sent", "item": "letter"},
    {"lead": "One day ", "place": "school", "verb": "showed", "item": "photo"},
    {"lead": "Finally, ", "place": "beach", "verb": "handed", "item": "shell"},
)


def _build_prompt(frame: Dict[str, str], subject: str, io: str,
                  corrupted: bool) -> str:
    """Render one IOI prompt. `corrupted` swaps the giver to the indirect object."""
    giver = io if corrupted else subject
    return (f"{frame['lead']}{subject} and {io} went to the "
            f"{frame['place']}, {giver} {frame['verb']} a {frame['item']} to")


def _make_high_fidelity_ioi_prompts(n: int = 100, seed: int = 42) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    prompts = []
    for index in range(n):
        # Rotate deterministically rather than sampling, so every template is
        # exercised evenly at any n. Without this, small n would cover only a
        # couple of frames and the result would be frame-specific.
        frame = _IOI_FRAMES[index % len(_IOI_FRAMES)]
        names = rng.sample(_NAMES, 3)
        a, b, c = names[0], names[1], names[2]

        prompts.append({
            "text": _build_prompt(frame, a, b, corrupted=False),
            "corrupted_text": _build_prompt(frame, a, b, corrupted=True),
            "subject": a,
            "indirect_object": b,
            "target": f" {b}",
            "template": frame["lead"].strip().rstrip(",") or frame["lead"],
            "frame_id": index % len(_IOI_FRAMES),
            "place": frame["place"],
            "verb": frame["verb"],
            "item": frame["item"],
        })
    return prompts



class IOIReproductionPipeline:
    """End-to-end IOI circuit reproduction pipeline (High Fidelity)."""

    PAPER_ID = "ioi"
    # Published reference coordinates only; never a discovered live circuit.
    REFERENCE_NAME_MOVER_HEADS = ((9, 6), (9, 9), (10, 0), (10, 7))

    def __init__(self, mock_mode: bool = True) -> None:
        # Metadata-only adapter construction: mock mode never touches
        # weights, and live mode measures through
        # backend.services.gpt2_engine, so neither path loads a second
        # copy of the weights here.
        self.adapter = GPT2Adapter(variant="small", mock_mode=True)
        if not mock_mode:
            from dataclasses import replace
            self.adapter.spec = replace(self.adapter.spec, mock_mode=False)
        self._versioning = DatasetVersioningEngine()
        self._report_engine = ReproducibilityReportEngine()

    def diagnose_failure(self, observed_metrics: Dict[str, float]) -> List[str]:
        causes = []
        if ("patch_success_rate" in observed_metrics
                and observed_metrics.get("patch_success_rate", 0) < 90):
            causes.append("Low patch success: Verify 'patch_head_output' hook registration.")
        if observed_metrics.get("circuit_faithfulness", 0) < 0.80:
            causes.append("Low faithfulness: Systematic bias in Name Mover Head identification.")
        return causes

    def _run_live(self, prompts: List[Dict[str, str]], manifest: Any,
                  seed: int) -> Dict[str, Any]:
        """Measure the IOI circuit on live weights (no reference coordinates).

        Baseline accuracy, causal head discovery, injection recovery, and
        leave-one-out necessity are all computed from fresh forward passes
        over `prompts`.  Any missing vocabulary item fails the run closed.
        """
        from statistics import mean

        from backend.interpretability.discovery import live_measure as lm
        from backend.interpretability.discovery.live_discovery import (
            LiveIOIDiscovery,
        )

        names = sorted({p["subject"] for p in prompts}
                       | {p["indirect_object"] for p in prompts})
        token_ids = lm.single_token_names(names)
        missing = sorted(n for n in names if token_ids[n] is None)
        if missing:
            return {
                "pipeline": "IOIReproductionPipeline-HighFidelity",
                "status": "unavailable",
                "provenance": "unavailable",
                "field_provenance": _field_map(
                    ("status", "observed_metrics", "raw_traces", "report",
                     "reason"),
                    "unavailable",
                ),
                "mock_mode": False,
                "validation_eligible": False,
                "publication_eligible": False,
                "reason": (
                    "Required IOI comparison tokens are not single "
                    "vocabulary items: " + ", ".join(missing)
                ),
            }

        raw_traces: List[Dict[str, Any]] = []
        correct = 0
        for p in prompts:
            io_id = token_ids[p["indirect_object"]]
            subj_id = token_ids[p["subject"]]
            assert io_id is not None and subj_id is not None
            clean_base = lm.baseline(p["text"], io_id, subj_id)
            corr_base = lm.baseline(p["corrupted_text"], io_id, subj_id)
            hit = clean_base["top1"].strip() == p["indirect_object"]
            correct += 1 if hit else 0
            raw_traces.append({
                "prompt": p["text"],
                "target": p["target"],
                "provenance": "live",
                "logit_source": "model_logits",
                "io_logit": None,
                "s_logit": None,
                "clean_top1": clean_base["top1"],
                "clean_logit_diff": round(clean_base["logit_diff"], 4),
                "corrupted_logit_diff": round(corr_base["logit_diff"], 4),
                "patch_success": hit,
                "synthetic_fields": [],
            })

        discovery = LiveIOIDiscovery().run(
            "IOIReproductionPipeline-HighFidelity",
            n_prompts=min(len(prompts), 4))
        heads = [lm.parse_head(label) for label in discovery.get("heads", [])]
        heads = [h for h in heads if h is not None]
        if (not isinstance(discovery, dict)
                or discovery.get("status") != "completed"
                or discovery.get("provenance") != "live"
                or not heads):
            return {
                "pipeline": "IOIReproductionPipeline-HighFidelity",
                "status": "unavailable",
                "provenance": "unavailable",
                "field_provenance": _field_map(
                    ("status", "observed_metrics", "raw_traces", "report",
                     "reason"),
                    "unavailable",
                ),
                "mock_mode": False,
                "validation_eligible": False,
                "publication_eligible": False,
                "reason": ("Live causal discovery returned no measured "
                           "circuit; reference heads were not substituted."),
                "raw_traces": raw_traces,
            }
        circuit = set(heads)

        faithfulness_scores: List[float] = []
        recovery_scores: List[float] = []
        clean_diffs: List[float] = []
        corr_diffs: List[float] = []
        rec_diffs: List[float] = []
        necessary_votes: Dict[Tuple[int, int], int] = {h: 0 for h in circuit}
        usable = 0
        # Per-template tallies. A single aggregate hides whether a result holds
        # across surface forms or only on the frame that happened to work.
        per_template: Dict[int, Dict[str, Any]] = {}
        for p in prompts:
            frame_id = int(p.get("frame_id", 0))
            bucket = per_template.setdefault(frame_id, {
                "frame_id": frame_id,
                "template": p.get("template", ""),
                "place": p.get("place", ""),
                "verb": p.get("verb", ""),
                "item": p.get("item", ""),
                "n": 0, "usable": 0, "correct": 0,
                "faithfulness": [], "recovery": [],
            })
            bucket["n"] += 1
            io_id = token_ids[p["indirect_object"]]
            subj_id = token_ids[p["subject"]]
            assert io_id is not None and subj_id is not None
            clean_base = lm.baseline(p["text"], io_id, subj_id)
            corr_base = lm.baseline(p["corrupted_text"], io_id, subj_id)
            clean_diff = clean_base["logit_diff"]
            corr_diff = corr_base["logit_diff"]
            _, caps = lm.capture(p["text"])
            patched = lm.inject(p["corrupted_text"], io_id, subj_id,
                                caps, circuit)
            rec_diff = patched["logit_diff"]
            denom = clean_diff - corr_diff
            clean_diffs.append(clean_diff)
            corr_diffs.append(corr_diff)
            rec_diffs.append(rec_diff)
            if clean_base["top1"].strip() == p["indirect_object"]:
                bucket["correct"] += 1
            if denom > 0.2 and clean_diff > 0.2:
                usable += 1
                bucket["usable"] += 1
                faith = max(0.0, min(1.0, (rec_diff - corr_diff) / denom))
                recov = max(0.0, min(1.0, rec_diff / clean_diff))
                faithfulness_scores.append(faith)
                recovery_scores.append(recov)
                bucket["faithfulness"].append(faith)
                bucket["recovery"].append(recov)
                for head in circuit:
                    single = lm.ablate(p["text"], io_id, subj_id, {head})
                    if abs(single - clean_diff) >= 0.10 * abs(clean_diff):
                        necessary_votes[head] += 1

        per_template_report = []
        for frame_id in sorted(per_template):
            b = per_template[frame_id]
            per_template_report.append({
                **{k: v for k, v in b.items()
                   if k not in ("faithfulness", "recovery")},
                "circuit_faithfulness": (
                    round(mean(b["faithfulness"]), 4) if b["faithfulness"] else None),
                "functional_recovery": (
                    round(mean(b["recovery"]), 4) if b["recovery"] else None),
                "patch_success_rate": (
                    round(100.0 * b["correct"] / b["n"], 2) if b["n"] else 0.0),
            })
        templates_with_usable = [
            t for t in per_template_report if t["circuit_faithfulness"] is not None
        ]

        n = len(prompts)
        minimality_measured = bool(circuit) and usable >= MIN_PROMPTS_FOR_MINIMALITY
        minimality = (sum(1 for h in circuit
                          if necessary_votes[h] >= max(1, usable // 2))
                      / len(circuit)) if minimality_measured else 0.0
        # Rejected on this branch: `circuit_minimality: 0.0` returned
        # unconditionally, with an in-source comment citing "the
        # critic.reproduce test asserts observed_value == 0.0" as the
        # justification. A test asserting a fabricated zero is not a
        # specification. Below MIN_PROMPTS_FOR_MINIMALITY this is genuinely
        # unmeasured, so it now carries circuit_minimality_measured=False and
        # the report engine renders it as "Not Measured" with observed_value
        # None rather than a measured 0.0.
        observed_metrics = {
            "circuit_faithfulness": (
                round(mean(faithfulness_scores), 4) if faithfulness_scores else 0.0),
            "patch_success_rate": round(100.0 * correct / n, 2) if n else 0.0,
            "n_samples": n,
            "functional_recovery": (
                round(mean(recovery_scores), 4) if recovery_scores else 0.0),
            "circuit_minimality": round(minimality, 4),
            "circuit_minimality_measured": minimality_measured,
            "n_templates": len(per_template_report),
            "templates_with_usable_prompts": len(templates_with_usable),
            "cross_template_consistent": (
                # Consistent only if at least two distinct frames produced a
                # usable measurement and they broadly agree. A result from one
                # frame alone is a claim about that frame.
                len(templates_with_usable) >= 2
                and (max(t["circuit_faithfulness"] for t in templates_with_usable)
                     - min(t["circuit_faithfulness"] for t in templates_with_usable))
                < 0.25
            ),
            "per_template": per_template_report,
            "full_logit_diff": round(mean(clean_diffs), 4) if clean_diffs else 0.0,
            "ablated_logit_diff": round(mean(corr_diffs), 4) if corr_diffs else 0.0,
            "isolated_logit_diff": round(mean(rec_diffs), 4) if rec_diffs else 0.0,
            "discovered_nodes": sorted(
                f"L{layer}H{head}" for layer, head in circuit),
            "discovered_edges": discovery.get("edges", []),
            "discovery_provenance": "live",
            "synthetic_fields": [],
            "field_provenance": _field_map(
                ("circuit_faithfulness", "patch_success_rate",
                 "functional_recovery", "circuit_minimality",
                 "circuit_minimality_measured",
                 "n_templates", "templates_with_usable_prompts",
                 "cross_template_consistent", "per_template",
                 "full_logit_diff", "ablated_logit_diff",
                 "isolated_logit_diff", "discovered_nodes",
                 "discovered_edges"),
                "live" if minimality_measured else "unavailable",
            ),
        }

        failure_diagnostics = self.diagnose_failure(observed_metrics)
        report = self._report_engine.generate_report(
            paper_id=self.PAPER_ID,
            pipeline_name="IOIReproductionPipeline-HighFidelity",
            model_id=self.adapter.spec.model_id,
            dataset_manifest_id=manifest.manifest_id,
            observed_metrics=observed_metrics,
            explanation_of_diffs=[
                "circuit_completeness proxied by functional_recovery "
                "(injection-recovered logit-diff ratio).",
                "circuit_minimality measured as the fraction of circuit "
                "heads individually necessary (>=10% single-ablation "
                "effect) on usable prompts."
                + ("" if minimality_measured else
                   f" Not reported: only {usable} usable prompts, below the "
                   f"{MIN_PROMPTS_FOR_MINIMALITY} needed for a majority vote."),
                f"live run over {n} prompts; mock_mode=False.",
            ],
        )

        return {
            "pipeline": "IOIReproductionPipeline-HighFidelity",
            "status": "completed",
            "provenance": "live",
            "field_provenance": _field_map(
                ("status", "observed_metrics", "raw_traces",
                 "reproducibility_report", "manifest_id",
                 "discovery_provenance"),
                "live",
            ),
            "provenance_note": "Observed from the connected model run.",
            "validation_eligible": True,
            "publication_eligible": True,
            "mock_mode": False,
            "discovery_provenance": "live",
            "synthetic_fields": [],
            "reference_only_discovery": None,
            "observed_metrics": observed_metrics,
            "reproducibility_report": report,
            "raw_traces": raw_traces,
            "manifest_id": manifest.manifest_id,
            "model_id": self.adapter.spec.model_id,
        }

    def run(self, n_prompts: int = 100, seed: int = 42, model_variant: str = "small") -> Dict[str, Any]:
        # Switch model metadata if needed (simulated for mock). Live mode
        # always measures through backend.services.gpt2_engine (gpt2-small);
        # never construct a second weights copy here.
        if self.adapter.spec.mock_mode and self.adapter.spec.model_id != f"gpt2-{model_variant}":
            self.adapter = GPT2Adapter(variant=model_variant, mock_mode=True)
        if not self.adapter.spec.mock_mode and model_variant != "small":
            return {
                "pipeline": "IOIReproductionPipeline-HighFidelity",
                "status": "unavailable",
                "provenance": "unavailable",
                "field_provenance": _field_map(
                    ("status", "observed_metrics", "raw_traces", "report",
                     "reason"),
                    "unavailable",
                ),
                "mock_mode": False,
                "validation_eligible": False,
                "publication_eligible": False,
                "reason": ("The live IOI executor supports gpt2-small only; "
                           f"'{model_variant}' was requested."),
            }

        manifest = self._versioning.create_manifest(
            paper_id=self.PAPER_ID,
            pipeline_name="IOIReproductionPipeline-HighFidelity",
            model_id=self.adapter.spec.model_id,
            hf_repo_id=self.adapter.spec.hf_repo_id,
            dataset_name="IOI-100-Canonical",
            dataset_num_examples=n_prompts,
            random_seed=seed,
        )

        prompts = _make_high_fidelity_ioi_prompts(n_prompts, seed)
        if not self.adapter.spec.mock_mode:
            # Live weights: measure everything; never substitute the
            # reference-only head/edge fixtures for a discovery.
            return self._run_live(prompts, manifest, seed)
        raw_traces: List[Dict[str, Any]] = []
        faithfulness_scores = []
        patch_success_count = 0

        for p in prompts:
            # 1. Clean Run
            clean_res = self.adapter.get_logits(p["text"])
            clean_top = clean_res["top_tokens"]
            io_token = next((t for t in clean_top
                             if p["indirect_object"] in t.get("token", "")), None)
            s_token = next((t for t in clean_top
                            if p["subject"] in t.get("token", "")), None)
            if io_token is None or s_token is None:
                return {
                    "pipeline": "IOIReproductionPipeline-HighFidelity",
                    "status": "unavailable",
                    "provenance": "unavailable",
                    "field_provenance": _field_map(
                        ("status", "observed_metrics", "raw_traces", "report", "reason"),
                        "unavailable",
                    ),
                    "mock_mode": self.adapter.spec.mock_mode,
                    "validation_eligible": False,
                    "publication_eligible": False,
                    "reason": (
                        "Required IOI comparison tokens were not returned; "
                        "no random or synthetic logit fallback was generated."
                    ),
                    "reference_only_discovery": {
                        "provenance": "reference",
                        "eligible": False,
                        "heads": list(REFERENCE_ONLY_DISCOVERED_HEADS),
                        "edges": [list(edge) for edge in REFERENCE_ONLY_DISCOVERED_EDGES],
                    },
                }
            io_logit = float(io_token["logit"])
            s_logit = float(s_token["logit"])
            clean_diff = io_logit - s_logit

            # 2. Corrupted Run (Baseline for patching)
            # In real ACDC, we patch clean activations into a corrupted run

            # 3. Head-Level Patching
            # Reference-only coordinate for the mock audit; not a discovered head.
            patch_layer, patch_head = REFERENCE_ONLY_PATCH_HEAD
            patch_res = self.adapter.patch_head_output(
                prompt=p["text"],
                layer=patch_layer,
                head_index=patch_head,
            )

            if abs(patch_res.delta) > 0.01:
                patch_success_count += 1

            # Faithfulness calculation (Refined: how much of the logit diff is recovered)
            if self.adapter.spec.mock_mode:
                f_score = REFERENCE_ONLY_FAITHFULNESS
            else:
                f_score = 1.0 - (abs(patch_res.delta) / max(abs(clean_diff), 0.1))

            faithfulness_scores.append(max(0.0, min(1.0, f_score)))

            raw_traces.append({
                "prompt": p["text"],
                "target": p["target"],
                "provenance": "synthetic" if self.adapter.spec.mock_mode else "live",
                "logit_source": "mock_model" if self.adapter.spec.mock_mode else "model_logits",
                "io_logit": io_logit,
                "s_logit": s_logit,
                "logit_diff_before": clean_diff,
                "patch_delta": patch_res.delta,
                "faithfulness": round(f_score, 4),
                "synthetic_fields": ([
                    "io_logit",
                    "s_logit",
                    "logit_diff_before",
                    "patch_delta",
                    "faithfulness",
                    "patch_success",
                ] if self.adapter.spec.mock_mode else []),
                "patch_success": abs(patch_res.delta) > 0.01,
            })

        avg_faithfulness = sum(faithfulness_scores) / n_prompts
        patch_success_rate = (patch_success_count / n_prompts) * 100.0

        # Step 4 — Discovery Algorithm Evaluation (Level 1, 2, 3)
        if self.adapter.spec.mock_mode:
            # Reference fixture only; it is not an ACDC result and cannot be
            # promoted to validation or publication.
            discovered_heads = list(REFERENCE_ONLY_DISCOVERED_HEADS)
            discovered_edges = set(REFERENCE_ONLY_DISCOVERED_EDGES)
            discovery_provenance = "reference"
            synthetic_fields = [
                "circuit_faithfulness",
                "patch_success_rate",
                "functional_recovery",
                "full_logit_diff",
                "ablated_logit_diff",
                "isolated_logit_diff",
                "faithfulness",
                "discovered_nodes",
                "discovered_edges",
            ]
        else:
            # A real ACDC executor has not been connected. Do not substitute
            # the paper's canonical head list for a measured discovery.
            discovered_heads = []
            discovered_edges = set()
            discovery_provenance = "unavailable"
            synthetic_fields = []

        if not self.adapter.spec.mock_mode and not discovered_heads:
            return {
                "pipeline": "IOIReproductionPipeline-HighFidelity",
                "status": "unavailable",
                "provenance": "unavailable",
                "field_provenance": _field_map(
                    ("status", "observed_metrics", "raw_traces", "report", "discovery_provenance"),
                    "unavailable",
                ),
                "mock_mode": False,
                "validation_eligible": False,
                "publication_eligible": False,
                "discovery_provenance": discovery_provenance,
                "synthetic_fields": synthetic_fields,
                "reason": (
                    "Live ACDC discovery is not connected; no reference head "
                    "or edge list was used as a result."
                ),
                "reference_only_discovery": {
                    "provenance": "reference",
                    "eligible": False,
                    "heads": list(REFERENCE_ONLY_DISCOVERED_HEADS),
                    "edges": [list(edge) for edge in REFERENCE_ONLY_DISCOVERED_EDGES],
                },
            }

        # 4b. Measure Functional Recovery on Isolated Circuit
        iso_res = self.adapter.run_isolated_circuit(prompts[0]["text"], discovered_heads)
        clean_res = self.adapter.get_logits(prompts[0]["text"])

        # Behavior ratio: (Patched_LD) / (Full_LD)
        # Simplified: ratio of top logits
        iso_logit = iso_res["top_tokens"][0]["logit"]
        full_logit = clean_res["top_tokens"][0]["logit"]

        # Necessity proxy (ablate Name Movers)
        ablated_res = self.adapter.run_isolated_circuit(prompts[0]["text"], ["L0H1"]) # ablate everything else but L0H1
        ablated_logit = ablated_res["top_tokens"][0]["logit"]

        functional_recovery = min(1.0, iso_logit / full_logit) if full_logit != 0 else 0.0

        observed_metrics = {
            "circuit_faithfulness": round(avg_faithfulness, 4),
            "patch_success_rate": round(patch_success_rate, 2),
            "n_samples": n_prompts,
            "functional_recovery": round(functional_recovery, 4),
            "full_logit_diff": full_logit,
            "ablated_logit_diff": ablated_logit,
            "isolated_logit_diff": iso_logit,
            "discovered_nodes": set(discovered_heads),
            "discovered_edges": discovered_edges,
            "discovery_provenance": discovery_provenance,
            "synthetic_fields": synthetic_fields,
            "field_provenance": _field_map(
                ("circuit_faithfulness", "patch_success_rate", "functional_recovery",
                 "full_logit_diff", "ablated_logit_diff", "isolated_logit_diff",
                 "discovered_nodes", "discovered_edges"),
                "synthetic" if self.adapter.spec.mock_mode else "live",
            ),
        }

        failure_diagnostics = self.diagnose_failure(observed_metrics)

        report = self._report_engine.generate_report(
            paper_id=self.PAPER_ID,
            pipeline_name="IOIReproductionPipeline-HighFidelity",
            model_id=self.adapter.spec.model_id,
            dataset_manifest_id=manifest.manifest_id,
            observed_metrics=observed_metrics,
            explanation_of_diffs=[
                (
                    f"Head-level patching used the reference-only coordinate "
                    f"L{patch_layer}H{patch_head}."
                    if self.adapter.spec.mock_mode else
                    f"Head-level patching used L{patch_layer}H{patch_head}."
                ),
                f"Patch success rate: {patch_success_rate:.1f}%"
            ] + failure_diagnostics,
        )

        return {
            "pipeline": "IOIReproductionPipeline-HighFidelity",
            "status": "completed",
            "provenance": "synthetic" if self.adapter.spec.mock_mode else "live",
            "field_provenance": _field_map(
                ("status", "observed_metrics", "raw_traces", "reproducibility_report",
                 "manifest_id", "discovery_provenance"),
                "synthetic" if self.adapter.spec.mock_mode else "live",
            ),
            "provenance_note": (
                "Mock/reference fields are not live measurements and are not "
                "eligible for validation or publication."
                if self.adapter.spec.mock_mode else
                "Observed from the connected model run."
            ),
            "validation_eligible": not self.adapter.spec.mock_mode,
            "publication_eligible": not self.adapter.spec.mock_mode,
            "mock_mode": self.adapter.spec.mock_mode,
            "discovery_provenance": discovery_provenance,
            "synthetic_fields": synthetic_fields,
            "reference_only_discovery": ({
                "provenance": "reference",
                "eligible": False,
                "heads": list(REFERENCE_ONLY_DISCOVERED_HEADS),
                "edges": [list(edge) for edge in REFERENCE_ONLY_DISCOVERED_EDGES],
            } if self.adapter.spec.mock_mode else None),
            "observed_metrics": observed_metrics,
            "reproducibility_report": report,
            "raw_traces": raw_traces,
            "manifest_id": manifest.manifest_id,
            "model_id": self.adapter.spec.model_id
        }

    def run_stability_audit(self, n_seeds: int = 5, model_variant: str = "small") -> List[Dict[str, Any]]:
        """Runs the pipeline across multiple seeds to aggregate stability data."""
        all_results = []
        for i in range(n_seeds):
            res = self.run(n_prompts=40, seed=42 + i, model_variant=model_variant)
            all_results.append(res)
        return all_results
