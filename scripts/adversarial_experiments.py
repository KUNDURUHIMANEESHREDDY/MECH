"""
Adversarial Scientific Validity Experiments
Demonstrates whether MECH manufactures significance for irrelevant components.
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from backend.core.model_adapter import get_model_adapter
from backend.core.experiment_engine import (
    MechanisticExperimentEngine, InterventionSpec, ComponentTarget, InterventionType
)

print("=" * 70)
print("ADVERSARIAL EXPERIMENTS — MECH SIGNIFICANCE MANUFACTURING")
print("=" * 70)

adapter = get_model_adapter(model_id="gpt2")
engine = MechanisticExperimentEngine(model_id="gpt2", adapter=adapter, storage=None)

prompt = "The capital of France is"
target_token = " Paris"

print(f"\nPrompt: '{prompt}'")
print(f"Target: '{target_token}'")

# Baseline
base = engine.run_baseline(prompt, target_token=target_token)
print(f"Baseline target logit: {base.target_logit:.4f}")
print(f"Baseline top tokens: {base.top_tokens[:3] if hasattr(base, 'top_tokens') else base.top_token}")

# ---------------------------------------------------------------
# ADVERSARIAL 1: IRRELEVANT COMPONENT
# Ablate layer 0, head 0 — should be irrelevant for factual recall
# ---------------------------------------------------------------
print("\n" + "-" * 70)
print("ADVERSARIAL 1: IRRELEVANT COMPONENT (layer 0, head 0)")
print("-" * 70)

spec_irr = InterventionSpec(
    clean_prompt=prompt,
    target_token=target_token,
    components=[ComponentTarget(type="attention_head", layer=0, index=0)],
    intervention_type=InterventionType.ABLATION_ZERO,
    random_seed=42,
    repeats=1,
)
res_irr = engine.run_intervention_experiment(spec_irr)
print(f"Delta logit: {res_irr.delta_logit:.4f}")
print(f"Evidence tier: {res_irr.evidence_tier}")
print(f"Verdict: {res_irr.verdict}")
print(f"Controls: {[(c.control_type, round(c.delta_logit,4), c.passed) for c in res_irr.controls]}")

# ---------------------------------------------------------------
# ADVERSARIAL 2: RANDOM COMPONENT
# Ablate layer 11, head 11 — arbitrary far component
# ---------------------------------------------------------------
print("\n" + "-" * 70)
print("ADVERSARIAL 2: RANDOM COMPONENT (layer 11, head 11)")
print("-" * 70)

spec_rand = InterventionSpec(
    clean_prompt=prompt,
    target_token=target_token,
    components=[ComponentTarget(type="attention_head", layer=11, index=11)],
    intervention_type=InterventionType.ABLATION_ZERO,
    random_seed=42,
    repeats=1,
)
res_rand = engine.run_intervention_experiment(spec_rand)
print(f"Delta logit: {res_rand.delta_logit:.4f}")
print(f"Evidence tier: {res_rand.evidence_tier}")
print(f"Verdict: {res_rand.verdict}")
print(f"Controls: {[(c.control_type, round(c.delta_logit,4), c.passed) for c in res_rand.controls]}")

# ---------------------------------------------------------------
# ADVERSARIAL 3: SPECIFICITY DENOMINATOR COLLAPSE
# Demonstrate the division-by-near-zero bug in specificity
# ---------------------------------------------------------------
print("\n" + "-" * 70)
print("ADVERSARIAL 3: SPECIFICITY DENOMINATOR COLLAPSE")
print("-" * 70)

def compute_specificity_buggy(mean_dz, neg_deltas):
    eps = 1e-4
    max_neg_dz = max(neg_deltas)
    return max(0.0, mean_dz) / max(eps, max_neg_dz)

# Case A: normal — target effect 0.5, largest control effect 0.3
spec_a = compute_specificity_buggy(0.5, [0.1, 0.2, 0.3])
print(f"Normal:   target=0.5, controls=[0.1,0.2,0.3] -> specificity={spec_a:.2f}")

# Case B: a control happens to nudge target UP (negative delta_logit)
# max_neg_dz becomes negative -> denominator collapses to eps -> specificity explodes
spec_b = compute_specificity_buggy(0.5, [-0.2, -0.1, 0.05])
print(f"Collapse: target=0.5, controls=[-0.2,-0.1,0.05] -> specificity={spec_b:.2f}  <-- FALSE INFLATION")

# Case C: even with TINY target effect, specificity passes threshold 2.0
spec_c = compute_specificity_buggy(0.05, [-0.3, -0.2, -0.1])
print(f"Tiny:     target=0.05, controls=[-0.3,-0.2,-0.1] -> specificity={spec_c:.2f}  <-- PASSES >= 2.0")

# ---------------------------------------------------------------
# ADVERSARIAL 4: "PASSED" HARDCODING ON NEGATIVE CONTROLS
# ---------------------------------------------------------------
print("\n" + "-" * 70)
print("ADVERSARIAL 4: NEGATIVE CONTROLS HARDCODED passed=True")
print("-" * 70)

all_passed = all(c.passed for c in res_irr.controls)
print(f"Irrelevant-component run: all {len(res_irr.controls)} controls report passed={all_passed}")
print("  (Negative controls should be able to FAIL if they also move the target)")
for c in res_irr.controls:
    print(f"    {c.control_type}: delta={c.delta_logit:.4f}, passed={c.passed}")

# ---------------------------------------------------------------
# ADVERSARIAL 5: ANY ABLATION CHANGES OUTPUT
# Sweep multiple heads to show intervention≠causality
# ---------------------------------------------------------------
print("\n" + "-" * 70)
print("ADVERSARIAL 5: SWEEP — does ablating ANY head change output?")
print("-" * 70)

deltas = []
for layer in [0, 6, 11]:
    for head in [0, 6]:
        spec = InterventionSpec(
            clean_prompt=prompt,
            target_token=target_token,
            components=[ComponentTarget(type="attention_head", layer=layer, index=head)],
            intervention_type=InterventionType.ABLATION_ZERO,
            random_seed=42,
            repeats=1,
        )
        r = engine.run_intervention_experiment(spec)
        deltas.append((layer, head, r.delta_logit, r.evidence_tier))
        print(f"  L{layer}H{head}: delta={r.delta_logit:.4f}, tier={r.evidence_tier}")

nonzero = sum(1 for _,_,d,_ in deltas if abs(d) > 0.01)
print(f"\n  Heads with |delta|>0.01: {nonzero}/{len(deltas)} (confirms ablation≠causality)")

# ---------------------------------------------------------------
# ADVERSARIAL 6: REPLICATION — are 5 trials independent?
# ---------------------------------------------------------------
print("\n" + "-" * 70)
print("ADVERSARIAL 6: REPLICATION INDEPENDENCE (5 'trials')")
print("-" * 70)

spec_rep = InterventionSpec(
    clean_prompt=prompt,
    target_token=target_token,
    components=[ComponentTarget(type="attention_head", layer=0, index=0)],
    intervention_type=InterventionType.ABLATION_ZERO,
    random_seed=42,
    repeats=5,
)
res_rep = engine.run_intervention_experiment(spec_rep)
print(f"5 trials requested. Trial deltas: {[round(d,6) for d in res_rep.trial_delta_logits]}")
print(f"All identical? {len(set(round(d,8) for d in res_rep.trial_delta_logits)) == 1}")
print(f"Stability reported: {res_rep.multi_trial_stats.stability if res_rep.multi_trial_stats else 'N/A'}")
print(f"CI95: {res_rep.multi_trial_stats.ci95_low if res_rep.multi_trial_stats else 'N/A'} "
      f"to {res_rep.multi_trial_stats.ci95_high if res_rep.multi_trial_stats else 'N/A'}")
print("  -> '95% CI' is just a point estimate (variance=0 from duplicate trials)")

# ---------------------------------------------------------------
# SUMMARY VERDICT
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("ADVERSARIAL VERDICT")
print("=" * 70)
print(f"[1] Irrelevant L0H0: tier={res_irr.evidence_tier} (should be INSUFFICIENT/REFUTED)")
print(f"[2] Random L11H11:  tier={res_rand.evidence_tier} (should be INSUFFICIENT/REFUTED)")
print(f"[3] Specificity collapse demonstrable: {spec_b:.0f}x and {spec_c:.0f}x (false inflation)")
print(f"[4] Negative controls always passed=True: {all_passed}")
print(f"[5] {nonzero}/{len(deltas)} heads change output (intervention≠causality)")
print(f"[6] 5 'trials' identical: {len(set(round(d,8) for d in res_rep.trial_delta_logits)) == 1}")
