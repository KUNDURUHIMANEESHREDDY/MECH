"""MECH Standalone CLI for Mechanistic Experimentation.

Allows Agent 1, researchers, and automated tools to execute real mechanistic experiments
from the command line without any UI or frontend dependencies.

Commands:
- model list / model info / model validate
- dataset list / dataset inspect
- run-baseline
- run-intervention
- run-patching
- run-pipeline
- inspect-neuron / inspect-head / logit-lens
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from typing import Any, Dict, List, Optional

from backend.core.experiment_engine import (
    ComponentTarget,
    InterventionSpec,
    MechanisticExperimentEngine,
)
from backend.core.model_registry import get_model_registry
from backend.datasets.dataset_manager import DatasetManager
from backend.storage.database import DesktopStorage
from backend.storage.scientific_entities import (
    EvidenceProvenanceSource,
    EvidenceRecord,
    InterventionType,
    KnowledgeType,
)


def _format_table(rows: List[Dict[str, Any]], headers: List[str]) -> str:
    """Formats rows as a terminal markdown table."""
    if not rows:
        return "(no rows)"
    col_widths = {h: max(len(h), max(len(str(r.get(h, ""))) for r in rows)) for h in headers}
    header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    sep_line = "-+-".join("-" * col_widths[h] for h in headers)
    body = "\n".join(" | ".join(str(r.get(h, "")).ljust(col_widths[h]) for h in headers) for r in rows)
    return f"{header_line}\n{sep_line}\n{body}"


def handle_model(args: argparse.Namespace) -> None:
    storage = DesktopStorage()
    registry = get_model_registry(storage)

    if args.action == "list":
        models = registry.list_models()
        if args.json:
            print(json.dumps(models, indent=2))
        else:
            headers = ["model_id", "model_name", "parameter_count", "precision", "status"]
            print(_format_table(models, headers))

    elif args.action == "info":
        meta = registry.get_metadata(args.model_id)
        if not meta:
            print(f"Error: Model '{args.model_id}' not found.", file=sys.stderr)
            sys.exit(1)
        if args.json:
            print(json.dumps(meta, indent=2))
        else:
            print(f"\n--- Model Info: {args.model_id} ---")
            for k, v in meta.items():
                print(f"  {k:20s}: {v}")

    elif args.action == "validate":
        res = registry.validate_model_health(args.model_id)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(f"\n--- Health Validation: {args.model_id} ---")
            print(f"Status:     {res.get('status')}")
            print(f"Latency:    {res.get('latency_ms')} ms")
            print(f"Device:     {res.get('device')}")
            print(f"Dimensions: layers={res.get('n_layers')}, heads={res.get('n_heads')}, d_model={res.get('d_model')}, d_mlp={res.get('d_mlp')}")


def handle_dataset(args: argparse.Namespace) -> None:
    data_dir = os.environ.get("MECH_DATA_DIR", "./datasets")
    mgr = DatasetManager(data_dir=data_dir)

    if args.action == "list":
        datasets = mgr.list_datasets()
        if args.json:
            print(json.dumps(datasets, indent=2))
        else:
            headers = ["dataset_id", "version", "num_prompts", "is_builtin", "citation"]
            print(_format_table(datasets, headers))

    elif args.action == "inspect":
        prompts = mgr.load(args.dataset_id)
        if args.json:
            print(json.dumps({"dataset_id": args.dataset_id, "num_prompts": len(prompts), "prompts": prompts}, indent=2))
        else:
            print(f"\n--- Dataset: {args.dataset_id} ({len(prompts)} prompts) ---")
            for idx, p in enumerate(prompts[:5]):
                print(f"[{idx+1}] Clean:     {p.get('clean')}")
                print(f"    Corrupted: {p.get('corrupted')}")
                print(f"    Target:    '{p.get('target')}' | Distractor: '{p.get('distractor')}'\n")


def handle_baseline(args: argparse.Namespace) -> None:
    engine = MechanisticExperimentEngine(model_id=args.model)
    res = engine.run_baseline(
        prompt=args.prompt,
        target_token=args.target,
        distractor_token=args.distractor,
        top_k=args.top_k,
    )
    if args.json:
        print(json.dumps(res.to_dict(), indent=2))
    else:
        print(f"\n=== Clean Baseline Forward Pass ===")
        print(f"Prompt:          \"{res.prompt}\"")
        print(f"Execution Time:  {res.execution_time_ms:.1f} ms")
        print(f"Next Predicted:  \"{res.next_token}\"")
        if res.target_token:
            print(f"Target Token:    \"{res.target_token}\" -> logit={res.target_logit:.2f}, prob={res.target_probability:.4f}, rank={res.target_rank}")
        if res.distractor_token:
            print(f"Distractor:      \"{res.distractor_token}\" -> logit={res.distractor_logit:.2f}, prob={res.distractor_probability:.4f}, rank={res.distractor_rank}")
        if res.logit_diff is not None:
            print(f"Logit Diff (LD): {res.logit_diff:.2f}")

        print("\nTop Predicted Tokens:")
        rows = [t.to_dict() for t in res.top_tokens]
        print(_format_table(rows, ["rank", "token", "logit", "probability"]))


def handle_intervention(args: argparse.Namespace) -> None:
    storage = DesktopStorage()
    engine = MechanisticExperimentEngine(model_id=args.model, storage=storage)

    itype_map = {
        "ablation_zero": InterventionType.ABLATION_ZERO,
        "ablation_mean": InterventionType.ABLATION_MEAN,
        "ablation_noise": InterventionType.ABLATION_NOISE,
        "patching": InterventionType.ACTIVATION_PATCHING,
        "activation_patching": InterventionType.ACTIVATION_PATCHING,
        "steering": InterventionType.STEERING,
        "scaling": InterventionType.SCALING,
        "clamping": InterventionType.CLAMPING,
    }
    itype = itype_map.get(args.type.lower(), InterventionType.ABLATION_ZERO)
    comp = ComponentTarget.from_string(args.component)

    spec = InterventionSpec(
        clean_prompt=args.prompt,
        corrupted_prompt=args.corrupted,
        target_token=args.target,
        distractor_token=args.distractor,
        target_components=[comp],
        intervention_type=itype,
        scale_coefficient=args.coeff,
        random_seed=args.seed,
    )

    res = engine.run_intervention_experiment(spec)
    if args.json:
        print(json.dumps(res.to_dict(), indent=2))
    else:
        print(f"\n=== Causal Intervention Experiment: {res.target_component} ===")
        print(f"Intervention:    {res.intervention_type} (coeff={res.scale_coefficient})")
        print(f"Prompt:          \"{res.clean_prompt}\"")
        print(f"Target Token:    \"{res.target_token}\"")
        print(f"Clean Target:    logit={res.clean_target_logit:.2f}, prob={res.clean_target_prob:.4f}, rank={res.clean_target_rank}")
        print(f"Intervened:      logit={res.intervened_target_logit:.2f}, prob={res.intervened_target_prob:.4f}, rank={res.intervened_target_rank}")
        print(f"\n--- Causal Effect Metrics ---")
        print(f"Δlogit:          {res.delta_logit:+.2f}")
        print(f"Δprob:           {res.delta_prob:+.4f} ({res.delta_prob*100:.1f}%)")
        if res.delta_logit_diff is not None:
            print(f"ΔLogit Diff:     {res.delta_logit_diff:+.2f}")
        if res.indirect_effect is not None:
            print(f"Indirect Effect: {res.indirect_effect:.2%}")
        print(f"KL Divergence:   {res.kl_divergence:.6f} nats")
        print(f"Prediction Flip: {res.top_prediction_flipped} ('{res.clean_predicted_token}' -> '{res.intervened_predicted_token}')")
        print(f"Specificity:     {res.specificity_ratio:.1f}x over 4-control battery (Cohen's d={res.cohens_d:.2f})")
        print(f"Evidence Tier:   {res.evidence_tier}")
        print(f"Verdict:         {res.verdict}")
        print(f"Provenance Hash: {res.provenance_hash}")


def handle_patching(args: argparse.Namespace) -> None:
    args.type = "activation_patching"
    args.coeff = 1.0
    handle_intervention(args)


def handle_pipeline(args: argparse.Namespace) -> None:
    storage = DesktopStorage()
    storage.initialize()

    # 1. Register Investigation & Hypothesis
    inv_id = f"inv_{uuid.uuid4().hex[:6]}"
    storage.save_investigation({
        "id": inv_id,
        "title": f"Investigation: {args.component}",
        "research_question": f"What is the causal role of {args.component}?",
        "model_id": args.model,
        "dataset_id": "custom",
    })

    hyp_id = f"hyp_{uuid.uuid4().hex[:6]}"
    storage.save_hypothesis({
        "id": hyp_id,
        "investigation_id": inv_id,
        "title": f"Hypothesis on {args.component}",
        "statement": args.statement or f"Component {args.component} causally drives {args.target}",
        "target_component": args.component,
        "prediction": "Ablation causes delta logit > 0.30",
        "expected_evidence": "Significant drop in target logit",
        "falsification_condition": "Delta logit <= 0.0 or specificity < 1.5",
    })

    # 2. Run Mechanistic Experiment
    engine = MechanisticExperimentEngine(model_id=args.model, storage=storage)
    comp = ComponentTarget.from_string(args.component)

    spec = InterventionSpec(
        clean_prompt=args.prompt,
        corrupted_prompt=args.corrupted,
        target_token=args.target,
        distractor_token=args.distractor,
        target_components=[comp],
        intervention_type=InterventionType.ABLATION_ZERO,
        random_seed=args.seed,
    )

    res = engine.run_intervention_experiment(spec)

    # 3. Create Evidence Record linked to Hypothesis
    supports = res.delta_logit > 0.30 and res.specificity_ratio >= 1.5
    evidence_knowledge_type = (
        KnowledgeType.CAUSAL_EVIDENCE
        if supports
        else (KnowledgeType.OBSERVATION if res.delta_logit > 0 else KnowledgeType.INFERENCE)
    )

    evidence = EvidenceRecord(
        investigation_id=inv_id,
        hypothesis_id=hyp_id,
        experiment_run_id=res.experiment_id,
        source_type=EvidenceProvenanceSource.COMPUTED,
        claim=f"Intervention on {args.component} produced Δlogit={res.delta_logit:.2f} (Δprob={res.delta_prob:.2%}) with specificity {res.specificity_ratio:.1f}x",
        evidence_level="CAUSALLY_VERIFIED" if (supports and res.specificity_ratio >= 2.0) else ("SUPPORTED" if supports else "CONTRADICTED"),
        supports_hypothesis=supports,
        knowledge_type=evidence_knowledge_type,
        metric_name="delta_logit",
        metric_value=res.delta_logit,
        baseline_value=res.clean_target_logit,
        control_value=res.mean_control_delta_logit,
        sample_size=1,
        statistical_details={"specificity_ratio": res.specificity_ratio, "cohens_d": res.cohens_d, "kl_divergence": res.kl_divergence},
        provenance_chain=[inv_id, hyp_id, res.experiment_id, res.provenance_hash],
        methodology=f"Zero-ablation forward hook on {args.component} with 4-control battery",
    )
    storage.save_evidence_record(evidence.model_dump())

    # 4. Evaluate Hypothesis Status
    from backend.science.hypothesis_engine import HypothesisEngine
    hyp_engine = HypothesisEngine(storage=storage)
    eval_res = hyp_engine.evaluate_hypothesis(hyp_id, inv_id)

    output_dict = {
        "investigation_id": inv_id,
        "hypothesis_id": hyp_id,
        "statement": args.statement or f"Component {args.component} causally drives {args.target}",
        "hypothesis_status": eval_res.get("status"),
        "supporting_evidence_count": eval_res.get("supporting_count"),
        "contradicting_evidence_count": eval_res.get("contradicting_count"),
        "experiment_result": res.to_dict(),
    }

    if args.json:
        print(json.dumps(output_dict, indent=2))
    else:
        print(f"\n=== Hypothesis -> Experiment -> Result Pipeline ===")
        print(f"Investigation:   {inv_id}")
        print(f"Hypothesis:      {hyp_id}")
        print(f"Statement:       \"{output_dict['statement']}\"")
        print(f"Evaluation:      {output_dict['hypothesis_status']}")
        print(f"Supporting:      {output_dict['supporting_evidence_count']} | Contradicting: {output_dict['contradicting_evidence_count']}")
        print(f"Δlogit:          {res.delta_logit:.2f} (Specificity {res.specificity_ratio:.1f}x)")
        print(f"Provenance Hash: {res.provenance_hash}")


def handle_inspect_neuron(args: argparse.Namespace) -> None:
    engine = MechanisticExperimentEngine(model_id=args.model)
    res = engine.inspect_neuron(layer=args.layer, neuron_idx=args.neuron, prompt=args.prompt)
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(f"\n=== Neuron Inspection: L{args.layer}.mlp.N{args.neuron} ===")
        for k, v in res.items():
            print(f"  {k:20s}: {v}")


def handle_inspect_head(args: argparse.Namespace) -> None:
    engine = MechanisticExperimentEngine(model_id=args.model)
    res = engine.inspect_head(layer=args.layer, head_idx=args.head, prompt=args.prompt)
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(f"\n=== Attention Head Inspection: L{args.layer}H{args.head} ===")
        for k, v in res.items():
            if k != "top_attended_tokens":
                print(f"  {k:20s}: {v}")
        if res.get("top_attended_tokens"):
            print("\nTop Attended Tokens:")
            print(_format_table(res["top_attended_tokens"], ["index", "token", "weight"]))


def handle_logit_lens(args: argparse.Namespace) -> None:
    engine = MechanisticExperimentEngine(model_id=args.model)
    traj = engine.compute_logit_lens(prompt=args.prompt, top_k=args.top_k)
    if args.json:
        print(json.dumps(traj, indent=2))
    else:
        print(f"\n=== Logit Lens Trajectory for \"{args.prompt}\" ===")
        for step in traj:
            top_strs = ", ".join(f"\"{t['token']}\" ({t['probability']:.2f})" for t in step["top_predictions"][:3])
            print(f"{step['layer_label']:10s} (norm={step['residual_norm']:6.2f}) -> {top_strs}")


def handle_discover(args: argparse.Namespace) -> None:
    from backend.core.discovery_engine import AutomatedDiscoveryEngine
    engine = AutomatedDiscoveryEngine(model_id=args.model)
    report = engine.discover_and_validate_circuit(
        clean_prompt=args.prompt,
        corrupted_prompt=args.corrupted,
        target_token=args.target,
        distractor_token=args.distractor,
        max_candidates_to_validate=args.max_candidates,
        validation_repeats=args.repeats,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"\n=== Automated Mechanistic Circuit Discovery ===")
        print(f"Model:           {report.model_id} (hash={report.model_hash[:10]})")
        print(f"Prompt:          \"{report.clean_prompt}\"")
        print(f"Target Token:    \"{report.target_token}\"")
        print(f"Scanned:         {report.total_heads_scanned} heads across {len(report.layer_scan)} layers")
        print(f"Execution Time:  {report.execution_time_ms:.1f} ms")

        print("\n--- Causal Layer Windows ---")
        causal_layers = [l.to_dict() for l in report.layer_scan if l.is_causal_window]
        if causal_layers:
            print(_format_table(causal_layers, ["layer", "delta_logit", "indirect_effect", "residual_norm"]))

        print("\n--- Discovered & Ranked Candidate Components ---")
        cand_rows = [c.to_dict() for c in report.top_candidates[:args.max_candidates]]
        print(_format_table(cand_rows, ["causal_rank", "component_id", "component_type", "functional_role", "delta_logit", "validation_status"]))


def main() -> None:
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")

    parser = argparse.ArgumentParser(description="MECH Mechanistic Experimentation CLI", parents=[common_parser])
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. Model Subcommand
    p_model = subparsers.add_parser("model", help="Model registry & inspection", parents=[common_parser])
    p_model.add_argument("action", choices=["list", "info", "validate"], help="Action to perform")
    p_model.add_argument("--model-id", default="gpt2", help="Model ID")
    p_model.set_defaults(func=handle_model)

    # 2. Dataset Subcommand
    p_data = subparsers.add_parser("dataset", help="Dataset management", parents=[common_parser])
    p_data.add_argument("action", choices=["list", "inspect"], help="Action to perform")
    p_data.add_argument("--dataset-id", default="ioi", help="Dataset ID")
    p_data.set_defaults(func=handle_dataset)

    # 3. Baseline Subcommand
    p_base = subparsers.add_parser("run-baseline", help="Run clean baseline forward pass", parents=[common_parser])
    p_base.add_argument("--prompt", required=True, help="Prompt text")
    p_base.add_argument("--target", default=None, help="Target token")
    p_base.add_argument("--distractor", default=None, help="Distractor token")
    p_base.add_argument("--model", default="gpt2", help="Model ID")
    p_base.add_argument("--top-k", type=int, default=5, help="Top K tokens")
    p_base.set_defaults(func=handle_baseline)

    # 4. Intervention Subcommand
    p_int = subparsers.add_parser("run-intervention", help="Run causal intervention experiment", parents=[common_parser])
    p_int.add_argument("--prompt", required=True, help="Clean prompt text")
    p_int.add_argument("--target", required=True, help="Target token")
    p_int.add_argument("--component", required=True, help="Target component (e.g. L9H9, L8_N412, L6_MLP)")
    p_int.add_argument("--type", default="ablation_zero", help="Intervention type (ablation_zero, ablation_mean, ablation_noise, steering, scaling)")
    p_int.add_argument("--coeff", type=float, default=0.0, help="Steering / Scaling coefficient")
    p_int.add_argument("--corrupted", default=None, help="Corrupted prompt (optional)")
    p_int.add_argument("--distractor", default=None, help="Distractor token (optional)")
    p_int.add_argument("--model", default="gpt2", help="Model ID")
    p_int.add_argument("--seed", type=int, default=42, help="Random seed")
    p_int.add_argument("--repeats", type=int, default=3, help="Number of repeated trials")
    p_int.set_defaults(func=handle_intervention)

    # 5. Patching Subcommand
    p_patch = subparsers.add_parser("run-patching", help="Run activation patching (clean vs corrupted)", parents=[common_parser])
    p_patch.add_argument("--clean", dest="prompt", required=True, help="Clean prompt text")
    p_patch.add_argument("--corrupted", required=True, help="Corrupted prompt text")
    p_patch.add_argument("--target", required=True, help="Target token")
    p_patch.add_argument("--distractor", default=None, help="Distractor token (optional)")
    p_patch.add_argument("--component", required=True, help="Target component (e.g. L9H9, L8_N412, L6_MLP)")
    p_patch.add_argument("--model", default="gpt2", help="Model ID")
    p_patch.add_argument("--seed", type=int, default=42, help="Random seed")
    p_patch.add_argument("--repeats", type=int, default=3, help="Number of repeated trials")
    p_patch.set_defaults(func=handle_patching)

    # 6. Pipeline Subcommand
    p_pipe = subparsers.add_parser("run-pipeline", help="Run Hypothesis -> Experiment -> Evidence pipeline", parents=[common_parser])
    p_pipe.add_argument("--prompt", required=True, help="Prompt text")
    p_pipe.add_argument("--target", required=True, help="Target token")
    p_pipe.add_argument("--component", required=True, help="Target component (e.g. L9H9)")
    p_pipe.add_argument("--statement", default=None, help="Scientific hypothesis statement")
    p_pipe.add_argument("--corrupted", default=None, help="Corrupted prompt (optional)")
    p_pipe.add_argument("--distractor", default=None, help="Distractor token (optional)")
    p_pipe.add_argument("--model", default="gpt2", help="Model ID")
    p_pipe.add_argument("--seed", type=int, default=42, help="Random seed")
    p_pipe.add_argument("--repeats", type=int, default=3, help="Number of repeated trials")
    p_pipe.set_defaults(func=handle_pipeline)

    # 7. Automated Discovery Subcommand
    p_disc = subparsers.add_parser("discover", help="Automated candidate discovery across layers and heads", parents=[common_parser])
    p_disc.add_argument("--prompt", required=True, help="Clean prompt text")
    p_disc.add_argument("--target", required=True, help="Target token")
    p_disc.add_argument("--corrupted", default=None, help="Corrupted prompt text (optional)")
    p_disc.add_argument("--distractor", default=None, help="Distractor token (optional)")
    p_disc.add_argument("--model", default="gpt2", help="Model ID")
    p_disc.add_argument("--max-candidates", type=int, default=4, help="Max candidates to validate")
    p_disc.add_argument("--repeats", type=int, default=3, help="Validation repeats per candidate")
    p_disc.set_defaults(func=handle_discover)

    # 8. Neuron Inspection
    p_neu = subparsers.add_parser("inspect-neuron", help="Deep inspection of neuron weights and activation", parents=[common_parser])
    p_neu.add_argument("--layer", type=int, required=True, help="Layer index")
    p_neu.add_argument("--neuron", type=int, required=True, help="Neuron index")
    p_neu.add_argument("--prompt", default=None, help="Prompt to evaluate activation on")
    p_neu.add_argument("--model", default="gpt2", help="Model ID")
    p_neu.set_defaults(func=handle_inspect_neuron)

    # 9. Attention Head Inspection
    p_head = subparsers.add_parser("inspect-head", help="Deep inspection of attention head weights and pattern", parents=[common_parser])
    p_head.add_argument("--layer", type=int, required=True, help="Layer index")
    p_head.add_argument("--head", type=int, required=True, help="Head index")
    p_head.add_argument("--prompt", default=None, help="Prompt to evaluate attention matrix on")
    p_head.add_argument("--model", default="gpt2", help="Model ID")
    p_head.set_defaults(func=handle_inspect_head)

    # 10. Logit Lens
    p_lens = subparsers.add_parser("logit-lens", help="Compute layer-by-layer logit lens trajectory", parents=[common_parser])
    p_lens.add_argument("--prompt", required=True, help="Prompt text")
    p_lens.add_argument("--model", default="gpt2", help="Model ID")
    p_lens.add_argument("--top-k", type=int, default=5, help="Top K predictions per layer")
    p_lens.set_defaults(func=handle_logit_lens)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
