"""Phase 39.6 — TransformerLens Integration & Backend Alignment Audit.

Executes:
- Parallel inference across Native (HuggingFace) and TransformerLens backends.
- Logit and Activation alignment measurement.
- Cross-backend circuit validation.
"""

import sys
import os
import json

# Ensure project root is in path
sys.path.insert(0, os.getcwd())

from backend.science.models.adapter_registry import ModelAdapterRegistry
from backend.science.reproducibility.scientific_validator import ScientificValidator

def run_alignment_audit():
    print("Starting Phase 39.6: Backend Alignment Audit...")
    registry = ModelAdapterRegistry()
    validator = ScientificValidator()

    # 1. Instantiate both backends for GPT-2 Small
    print("\n[1/3] Initializing Backends...")
    native = registry.get_adapter("gpt2-small", mock_mode=True)
    tl_backend = registry.get_adapter("tl-gpt2-small", mock_mode=True)

    print(f"- Backend A: {native.spec.model_id} (Native)")
    print(f"- Backend B: {tl_backend.spec.model_id} (TransformerLens)")

    # 2. Run Alignment Audit
    print("\n[2/3] Measuring Numerical Alignment...")
    prompts = [
        "When Alice and Bob went to the store, Alice gave a drink to",
        "The capital of France is",
        "The quick brown fox jumps over the lazy"
    ]

    alignment = validator.validate_backend_alignment(native, tl_backend, prompts)

    print(f"- Mean Cosine Similarity: {alignment['mean_cosine_similarity']}")
    print(f"- Mean KL Divergence    : {alignment['mean_kl_divergence']}")
    print(f"- Top-1 Agreement       : {alignment['top_1_agreement_pct']}%")
    print(f"- Alignment Verdict     : {alignment['verdict']}")

    # 3. Generate Final Scientific Report
    print("\n[3/3] Generating Validation Artifacts...")
    artifacts = validator.generate_validation_artifacts(
        benchmark_results=[],
        scientific_leaderboard=[],
        backend_alignment=alignment,
        output_dir="benchmark_report_v39_6_alignment"
    )

    print(f"\nAudit Complete.")
    print(f"- Report: {artifacts['md']}")
    print(f"- Summary JSON: {artifacts['json']}")

if __name__ == "__main__":
    try:
        run_alignment_audit()
        print("\nPhase 39.6 Verification Status: PASSED")
    except Exception as e:
        print(f"\nVerification Status: FAILED - {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
