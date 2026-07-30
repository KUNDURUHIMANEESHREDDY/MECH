"""Phase 38 — Peer Review Simulator Verification Script.

Tests the unanimous PASS logic, diagnostic scorecards, and report generation.
"""

import sys
import os
import json

# Ensure project root is in path
sys.path.insert(0, os.getcwd())

from backend.research_platform.autonomous.research_governance import ResearchGovernanceEngine, ClaimState
from backend.research_platform.autonomous.peer_review_panel import ReviewVerdict
from backend.research_platform.autonomous.scientific_consensus_engine import ScientificConsensusEngine

def test_peer_review_pass():
    print("Test 1: High-Rigor Result (Expected PASS)")
    gov = ResearchGovernanceEngine()
    consensus = ScientificConsensusEngine()

    claim_id = "claim_ioi_gpt2_small"
    gov.register_claim(claim_id)

    # Mock High Rigor Evidence (from Phase 37 Ref)
    evidence = {
        "n_samples": 100,
        "ci_low": 0.86,
        "ci_high": 0.89,
        "fidelity_pct": 98.5,
        "patch_success_rate": 94.0,
        "primary_score": 87.4, # faithfulness
        "patch_layer": 9
    }
    metadata = {
        "has_corrupted_baseline": True,
        "ablation_type": "mean",
        "seed": 42,
        "git_sha": "abcd123",
        "dataset_hash": "hash789",
        "model_id": "gpt2-small",
        "torch_version": "2.1.0",
        "transformers_version": "4.35.0",
        "algorithm": "Path Patching",
        "has_controls": True,
        "raw_traces": True
    }

    result = gov.submit_for_review(claim_id, evidence, metadata)
    print(f"- Verdict: {result['review'].final_verdict}")
    print(f"- Overall Score: {result['review'].overall_score_pct}%")
    print(f"- Unanimous: {result['review'].unanimous_pass}")
    print(f"- Governance State: {result['current_state']}")

    # Generate Report
    report_path = consensus.generate_peer_review_report(result['review'], output_dir="review_test_output")
    print(f"- Report generated: {report_path}")

    assert result['review'].final_verdict == ReviewVerdict.PASS
    assert result['current_state'] == ClaimState.VALIDATED

def test_peer_review_revision_required():
    print("\nTest 2: Moderate Rigor (Expected REVISION_REQUIRED)")
    gov = ResearchGovernanceEngine()
    claim_id = "claim_moderate"
    gov.register_claim(claim_id)

    evidence = {
        "n_samples": 50,
        "ci_low": 0.75,
        "ci_high": 0.85,
        "fidelity_pct": 85.0,
        "patch_success_rate": 80.0,
        "primary_score": 75.0,
        "patch_layer": 9
    }
    # Metadata to ensure REVISION_REQUIRED (score between FAIL and PASS thresholds)
    metadata = {
        "seed": 42,
        "algorithm": "patching",
        "has_corrupted_baseline": True,
        "ablation_type": "mean",
        "dataset_hash": "hash123",
        "git_sha": "git456",
        "model_id": "gpt2-small",
        "has_controls": False # missing some items
    }

    result = gov.submit_for_review(claim_id, evidence, metadata)
    print(f"- Verdict: {result['review'].final_verdict}")
    print(f"- Overall Score: {result['review'].overall_score_pct}%")
    print(f"- Governance State: {result['current_state']}")

    assert result['review'].final_verdict == ReviewVerdict.REVISION_REQUIRED
    assert result['current_state'] == ClaimState.REVISION_REQUIRED

if __name__ == "__main__":
    try:
        test_peer_review_pass()
        test_peer_review_revision_required()
        print("\nVerification Status: PASSED")
    except Exception as e:
        print(f"\nVerification Status: FAILED - {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
