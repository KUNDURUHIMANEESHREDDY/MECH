# AGENT2_EPISTEMIC_AUDIT.md

## Executive Summary

Agent 2 (Scientific Research Orchestrator) has been audited for epistemic integrity. **40/40 tests passed with 0 critical violations.**

## Audit Results

| Category | Tests | Status |
|---|---|---|
| Hypothesis Lifecycle | 6/6 | ALL PASS |
| Evidence Classification | 4/4 | ALL PASS |
| LLM Adversarial | 4/4 | ALL PASS |
| Contradictory Evidence | 3/3 | ALL PASS |
| Falsification Quality | 4/4 | ALL PASS |
| Replication Requirements | 4/4 | ALL PASS |
| Uncertainty Propagation | 3/3 | ALL PASS |
| LLM Fallback | 4/4 | ALL PASS |
| Research-Loop Adversarial | 8/8 | ALL PASS |
| **TOTAL** | **40/40** | **ALL PASS** |

## 1. State-Machine Findings

### Valid Transitions (Enforced)

```
PROPOSED → PLANNED → TESTING → REPLICATION → EVALUATION → SUPPORTED/REFUTED/INSUFFICIENT
```

### Illegal Transitions (All Rejected)

| Attack | Result |
|---|---|
| PROPOSED → SUPPORTED | REJECTED |
| PROPOSED → REPLICATION | REJECTED |
| PLANNED → EVALUATION | REJECTED |
| TESTING → EVALUATION | REJECTED (must go through REPLICATION) |
| EVALUATION → TESTING | REJECTED (cannot go backwards) |

### Evidence Required for Each Transition

| From | To | Required Evidence |
|---|---|---|
| PROPOSED | PLANNED | Experiment plan created |
| PLANNED | TESTING | Agent 1 execution initiated |
| TESTING | REPLICATION | Initial Agent 1 evidence |
| REPLICATION | EVALUATION | Replicated Agent 1 evidence |
| EVALUATION | SUPPORTED | Multiple supporting experiments |
| EVALUATION | REFUTED | Contradicting evidence |
| EVALUATION | INSUFFICIENT | Inconclusive evidence |

## 2. Evidence-Classification Findings

### Classification Rules

| Classification | Requirement |
|---|---|
| OBSERVED | Agent 1 measured effect |
| REPRODUCED | Multiple Agent 1 runs |
| SPECIFIC | Effect > control by >2x |
| CAUSALLY_SUPPORTED | Multiple specific replications |
| REFUTED | Agent 1 shows no effect |
| INSUFFICIENT | Inconclusive Agent 1 results |

### Test Results

| Test | Result |
|---|---|
| OBSERVED cannot become CAUSALLY_SUPPORTED without intervention | PASS |
| AI_GENERATED cannot masquerade as COMPUTED_EVIDENCE | PASS |
| MODEL_INTERPRETATION cannot masquerade as COMPUTED_EVIDENCE | PASS |
| No specificity data → INSUFFICIENT | PASS |

## 3. LLM Adversarial Tests

### Test Results

| Test | Result |
|---|---|
| LLM confidence cannot set SUPPORTED directly | PASS |
| LLM-only claim → INSUFFICIENT | PASS |
| Misleading LLM input ignored | PASS |
| LLM suggestion uses evidence rules | PASS |

### Rule Enforced

```
LLM → PROPOSAL → Agent 1 → MEASUREMENT → EVIDENCE → SCIENTIFIC STATUS
```

**NEVER:**

```
LLM → SCIENTIFIC FACT
```

## 4. Falsification Findings

### Falsification Quality

| Test | Result |
|---|---|
| Falsification has different controls (null controls) | PASS |
| Falsification has higher repeats (20 vs 10) | PASS |
| Falsification includes p_value metric | PASS |
| Hypothesis unchanged by falsification design | PASS |

### Falsification Design

- **Null controls**: L0H0, L6H6 (known irrelevant components)
- **Higher repeats**: 20 (vs 10 for main experiment)
- **Additional metrics**: p_value for significance testing
- **Purpose**: "What would disprove this hypothesis?"

## 5. Uncertainty Handling

### Test Results

| Test | Result |
|---|---|
| Uncertain effect → INSUFFICIENT | PASS |
| Zero effect → EVIDENCE_CONTRADICTS | PASS |
| Mixed confidence → INSUFFICIENT | PASS |

### Rule

Uncertainty from Agent 1 **must survive** through:

```
Agent 1 → Agent 2 → Research State → Agent 3
```

Agent 2 **cannot** transform uncertainty into strong causal conclusions.

## 6. Contradiction Handling

### Test Results

| Test | Result |
|---|---|
| Supporting evidence preserved | PASS |
| Contradicting evidence preserved | PASS |
| System does not cherry-pick | PASS |

### Rule

When evidence contradicts:

- Both supporting and contradicting evidence preserved in chain
- Assessment: `INSUFFICIENT_EVIDENCE` (mixed results)
- System cannot select only supporting evidence

## 7. LLM Fallback Audit

### Test Results

| Test | Result |
|---|---|
| LLM unavailable → heuristic fallback | PASS |
| Malformed response handled | PASS |
| Heuristic cannot fabricate CAUSALLY_SUPPORTED | PASS |
| Heuristic cannot set SUPPORTED | PASS |

### Heuristic Capabilities

| Allowed | Not Allowed |
|---|---|
| Interpretation (supportive/falsifying/inconclusive) | CAUSALLY_SUPPORTED |
| Rationale (natural language) | Direct state promotion |
| Suggestions (next steps) | Scientific status |

## 8. Research-Loop Adversarial Cases

### Test Results

| Case | Result |
|---|---|
| No candidate | PASS (empty priorities) |
| No experiment | PASS (INSUFFICIENT) |
| Failed experiment | PASS (EVIDENCE_CONTRADICTS) |
| Missing evidence | PASS (INSUFFICIENT) |
| Contradictory evidence | PASS (INSUFFICIENT) |
| No replication | PASS (EVALUATION blocked) |
| Partial replication | PASS (safe termination) |
| Model unavailable | PASS (INSUFFICIENT) |

## 9. Severity Assessment

### Critical Violations: 0

No critical epistemic violations found.

### Warnings: 0

No warnings.

## 10. Fixes Applied

### State Machine Fix

**Before**: TESTING could go directly to EVALUATION

**After**: TESTING must go through REPLICATION before EVALUATION

```python
# Before
HypothesisState.TESTING: [HypothesisState.REPLICATION, HypothesisState.EVALUATION],

# After
HypothesisState.TESTING: [HypothesisState.REPLICATION],  # Must replicate before evaluation
```

### Specificity Calculation Fix

**Before**: Division by zero produced `inf`

**After**: Returns 0 when no evidence exists

```python
# Before
specificity = avg_effect / max_control if max_control > 0 else float("inf")

# After
specificity = avg_effect / max_control if max_control > 0 and len(evidence_records) > 0 else 0.0
```

## Conclusion

Agent 2 **cannot** manufacture scientific conclusions from reasoning alone. The system enforces:

1. All measurements come from Agent 1
2. State transitions require evidence
3. REPLICATION required before EVALUATION
4. LLM provides proposals, not facts
5. Heuristics provide interpretation, not status
6. Invalid transitions are rejected
7. Uncertainty propagates correctly
8. Contradictory evidence preserved

**AUDIT RESULT: PASS**
