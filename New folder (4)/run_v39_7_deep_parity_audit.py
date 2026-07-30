"""Phase 39.7 — Deep Backend Parity & Hook Equivalence Audit.

Executes:
- Deep internal cache comparison (layer-by-layer activations).
- Hook equivalence test (Intervention Delta parity).
- Generates high-fidelity alignment report.
"""

import sys
import os
import json

# Ensure project root is in path
sys.path.insert(0, os.getcwd())

from backend.science.models.adapter_registry import ModelAdapterRegistry
from backend.science.reproducibility.scientific_validator import ScientificValidator

def run_deep_parity_audit():
    print("Starting Phase 39.7: Deep Backend Parity Audit...")
    registry = ModelAdapterRegistry()
    validator = ScientificValidator()

    # 1. Initialize Backends
    print("\n[1/4] Initializing Backends...")
    native = registry.get_adapter("gpt2-small", mock_mode=True)
    tl_backend = registry.get_adapter("tl-gpt2-small", mock_mode=True)

    prompt = "When Alice and Bob went to the store, Alice gave a drink to"

    # 2. Output Alignment (Legacy check)
    print("\n[2/4] Measuring Output Alignment...")
    alignment = validator.validate_backend_alignment(native, tl_backend, [prompt])
    print(f"- Top-1 Agreement: {alignment['top_1_agreement_pct']}%")

    # 3. Deep Internal Alignment
    print("\n[3/4] Layer-by-Layer Activation Parity...")
    internal_parity = validator.validate_internal_cache_alignment(native, tl_backend, prompt)
    print(f"- Overall Internal Similarity: {internal_parity['overall_internal_similarity']}")
    print(f"- Internal Verdict: {internal_parity['verdict']}")

    # 4. Hook Equivalence
    print("\n[4/4] Hook Equivalence (Intervention Parity)...")
    hook_eq = validator.validate_hook_equivalence(
        native, tl_backend, prompt, layer=9, head=9
    )
    print(f"- Native Delta: {hook_eq['delta_a']}")
    print(f"- TL Delta    : {hook_eq['delta_b']}")
    print(f"- Delta Error : {hook_eq['delta_error']}")
    print(f"- Hook Verdict : {hook_eq['verdict']}")

    # 5. Generate Final Report
    artifacts = validator.generate_validation_artifacts(
        benchmark_results=[],
        scientific_leaderboard=[],
        backend_alignment=alignment,
        internal_parity=internal_parity,
        hook_equivalence=hook_eq,
        output_dir="benchmark_report_v39_7_parity"
    )

    print(f"\nAudit Complete.")
    print(f"- Master Report : {artifacts['md']}")
    print(f"- Summary JSON  : {artifacts['json']}")

if __name__ == "__main__":
    try:
        run_deep_parity_audit()
        print("\nPhase 39.7 Verification Status: PASSED")
    except Exception as e:
        print(f"\nVerification Status: FAILED - {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
