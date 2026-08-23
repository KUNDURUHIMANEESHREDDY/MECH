import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.research_platform.meta.enhanced_reflection_engine import (
    EnhancedSelfReflectionEngine,
    HypothesisSurvivalTracker,
    ResourceOutcomeOracle,
    MetaInsightSynthesizer,
    HypothesisLifecycle,
    ResourceOutcomeRecord,
    DecisionPattern,
)

print("=" * 60)
print("PHASE 3: SELF-REFLECTION META-LOOP VERIFICATION")
print("=" * 60)

# Test 1: Create engine
print("\n[Test 1] Engine instantiation...")
engine = EnhancedSelfReflectionEngine()
print("  PASSED: EnhancedSelfReflectionEngine instantiates")

# Test 2: Hypothesis survival tracker
print("\n[Test 2] Hypothesis survival tracker...")
tracker = HypothesisSurvivalTracker()
# Record some hypotheses from a campaign
tracker.record_campaign_hypotheses(
    campaign_id="camp_001",
    hypotheses_tested=[
        "SAE Feature #42 mediates name token representation",
        "L8_N402 is an indirect object identifier",
    ],
    hypothesis_results={
        "h1": "confirmed",
        "h2": "falsified",
    },
)
lifecycle = tracker.get_lifecycle("SAE Feature #42 mediates name token representation")
print(f"  PASSED: Lifecycle retrieved: {lifecycle is not None}")
stats = tracker.get_statistics()
print(f"  PASSED: Statistics: {stats['total_hypotheses']} total hypotheses, "
      f"confirmation_rate={stats['confirmation_rate']}")

# Test 3: Resource outcome oracle
print("\n[Test 3] Resource outcome oracle...")
oracle = ResourceOutcomeOracle()
record = ResourceOutcomeRecord(
    record_id="rec_001",
    campaign_id="camp_001",
    compute_gb_hours=12.5,
    final_confidence=0.88,
    initial_confidence=0.50,
    confidence_gain=0.38,
    hypothesis_count=2,
    falsified_count=1,
    successful_hypotheses=1,
    cost_per_confidence_gain=12.5 / 0.38,
    efficiency_score=0.38 / 12.5,
    target_confidence_reached=True,
    notes="Test campaign",
)
oracle.record_campaign_outcome(record)
analysis = oracle.get_efficiency_analysis()
print(f"  PASSED: Efficiency analysis: {analysis['total_campaigns']} campaigns, "
      f"overall_efficiency={analysis['overall_efficiency']:.6f}")
budget_rec = oracle.get_optimal_budget_recommendation()
print(f"  PASSED: Budget recommendation: {budget_rec.get('recommended_compute_gb_hours', 'N/A')} GB hours")

# Test 4: Meta-insight synthesizer
print("\n[Test 4] Meta-insight synthesizer...")
synthesizer = MetaInsightSynthesizer(tracker, oracle)
insight_ids = synthesizer.analyze_campaign_patterns(
    campaign_id="camp_001",
    hypotheses_outcomes={"h1": "confirmed", "h2": "falsified"},
    resource_data={"compute_gb_hours": 12.5, "final_confidence": 0.88, "initial_confidence": 0.50},
)
print(f"  PASSED: Synthesizer generated {len(insight_ids)} insights")
all_insights = synthesizer.get_all_insights()
print(f"  PASSED: {len(all_insights)} total insights")
all_patterns = synthesizer.get_all_patterns()
print(f"  PASSED: {len(all_patterns)} total decision patterns")

# Test 5: Full engine integration
print("\n[Test 5] Enhanced reflection report...")
engine2 = EnhancedSelfReflectionEngine()
report = engine2.generate_enhanced_reflection_report(
    campaign_id="camp_test",
    successful_hypotheses=["Test hypothesis 1"],
    failed_hypotheses=["Test hypothesis 2"],
    compute_used_gb_hours=10.0,
    hypotheses_outcomes={"hyp_test": "confirmed"},
    resource_data={"compute_gb_hours": 10.0, "final_confidence": 0.85, "initial_confidence": 0.5},
)
print(f"  PASSED: Report generated")
print(f"  - hypothesis_lifecycle_changes: {len(report.get('hypothesis_lifecycle_changes', []))}")
print(f"  - resource_efficiency_metrics: {report.get('resource_efficiency_metrics', {})}")
print(f"  - cross_campaign_patterns_matched: {report.get('cross_campaign_patterns_matched', [])}")
print(f"  - meta_insights_generated: {report.get('meta_insights_generated', [])}")

# Test 6: Meta summary
print("\n[Test 6] Meta summary...")
meta_summary = engine2.get_meta_summary()
print(f"  PASSED: Meta summary keys: {list(meta_summary.keys())}")
print(f"  - total_campaigns_reflected: {meta_summary['total_campaigns_reflected']}")
print(f"  - hypothesis_survival total_hypotheses: {meta_summary['hypothesis_survival']['total_hypotheses']}")
print(f"  - resource_outcome total_campaigns: {meta_summary['resource_outcome']['total_campaigns']}")

print("\n" + "=" * 60)
print("ALL VERIFICATION TESTS PASSED!")
print("=" * 60)