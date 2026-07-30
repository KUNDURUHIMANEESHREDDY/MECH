"""
Main entry point for the Neuron Inspector.

Runs a full analysis experiment on the default mock model and
saves results to JSON.

Usage:
    python main.py [--layer LAYER] [--output OUTPUT]

Examples:
    python main.py                          # Run full analysis on layer 0
    python main.py --layer 5                # Run analysis on layer 5
    python main.py --output results.json    # Save results to custom path
"""

import argparse
import json
import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.interpretability.data_generator import MockModelData
from backend.interpretability.mock_runtime import MockRuntime
from backend.interpretability.repository import ActivationRepository
from backend.interpretability.neuron_inspector import NeuronInspector
from backend.interpretability.attention_inspector import AttentionInspector
from backend.interpretability.residual_inspector import ResidualInspector
from backend.interpretability.heatmap import HeatmapGenerator
from analysis.experiment_runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(
        description="Run Neuron Inspector analysis"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="Layer index to analyze (default: 0)",
    )
    parser.add_argument(
        "--output",
        default="experiments/results/analysis_results.json",
        help="Output file path (default: experiments/results/analysis_results.json)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top neurons to inspect (default: 10)",
    )
    parser.add_argument(
        "--token",
        type=int,
        default=0,
        help="Token index to analyze (default: 0)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Neuron Inspector - Analysis Runner")
    print("=" * 60)

    # Initialize model and inspectors
    model = MockModelData()
    _runtime = MockRuntime(
        num_layers=model.num_layers,
        num_heads=model.num_heads,
        hidden_dim=model.hidden_dim,
        seq_len=model.seq_len,
        vocab_size=model.vocab_size,
        seed=model.seed,
    )
    _repository = ActivationRepository(_runtime)
    neuron_inspector = NeuronInspector(repository=_repository)
    attention_inspector = AttentionInspector(repository=_repository)
    residual_inspector = ResidualInspector(repository=_repository)
    heatmap_generator = HeatmapGenerator(repository=_repository)
    runner = ExperimentRunner(model=model)

    print(f"\nModel Configuration:")
    print(f"  Layers: {model.num_layers}")
    print(f"  Heads: {model.num_heads}")
    print(f"  Hidden Dim: {model.hidden_dim}")
    print(f"  Seq Len: {model.seq_len}")

    # Run neuron analysis
    print(f"\n--- Neuron Analysis (Layer {args.layer}) ---")
    neurons = neuron_inspector.inspect_layer(
        args.layer, top_k=args.top_k, token_index=args.token
    )
    print(f"  Top {args.top_k} neurons:")
    for n in neurons:
        print(f"    Neuron {n.neuron_index}: activation={n.activation:.4f}, "
              f"sparsity={n.statistics.sparsity:.2f}")

    # Run attention analysis
    print(f"\n--- Attention Analysis (Layer {args.layer}) ---")
    heads = attention_inspector.inspect_all_heads(args.layer, args.token)
    print(f"  Top 5 heads by importance:")
    for h in heads[:5]:
        print(f"    Head {h.head}: importance={h.importance:.4f}, "
              f"shape={h.shape}")

    # Run residual analysis
    print(f"\n--- Residual Analysis (Token {args.token}) ---")
    residuals = residual_inspector.inspect_all_layers(args.token)
    print(f"  Residual norms across layers:")
    for r in residuals:
        print(f"    {r.layer}: norm={r.norm:.4f}, "
              f"contribution={r.contribution:.4f}")

    # Run cross-layer analysis
    print(f"\n--- Cross-Layer Analysis ---")
    cross_layer = runner.run_cross_layer_analysis()
    trends = cross_layer.results["trends"]
    print(f"  Sparsity increasing: {trends['sparsity_increasing']}")
    print(f"  Mean increasing: {trends['mean_increasing']}")
    print(f"  Max sparsity layer: {trends['max_sparsity_layer']}")
    print(f"  Min sparsity layer: {trends['min_sparsity_layer']}")

    # Generate heatmaps
    print(f"\n--- Heatmap Generation ---")
    heatmaps = heatmap_generator.prepare_all_heatmaps(
        layer_index=args.layer, head_index=0, token_index=args.token
    )
    print(f"  Generated {len(heatmaps)} heatmaps:")
    for name, hm in heatmaps.items():
        print(f"    {name}: {len(hm.data)}x{len(hm.data[0])} matrix")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results = {
        "model_config": {
            "num_layers": model.num_layers,
            "num_heads": model.num_heads,
            "hidden_dim": model.hidden_dim,
            "seq_len": model.seq_len,
        },
        "neuron_analysis": {
            "layer": args.layer,
            "top_k": args.top_k,
            "neurons": [n.model_dump() for n in neurons],
        },
        "attention_analysis": {
            "layer": args.layer,
            "heads": [h.model_dump() for h in heads],
        },
        "residual_analysis": {
            "token_index": args.token,
            "residuals": [r.model_dump() for r in residuals],
        },
        "cross_layer_analysis": cross_layer.to_dict(),
        "heatmaps": {
            name: hm.model_dump() for name, hm in heatmaps.items()
        },
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output}")
    print("=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
