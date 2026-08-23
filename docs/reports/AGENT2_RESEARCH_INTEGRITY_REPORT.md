# AGENT 2 RESEARCH INTEGRITY REPORT

## Executive Summary

Agent 2 (Scientific Research Orchestrator) has been tested for epistemic integrity. The system **cannot manufacture scientific conclusions from reasoning alone**. All 36 tests passed.

## Acceptance Criterion

Agent 2 must prove:

```
LLM → PROPOSAL → Agent 1 → MEASUREMENT → EVIDENCE → SCIENTIFIC STATUS
```

**NEVER:**

```
LLM → SCIENTIFIC FACT
```

## Test Results

| Category | Tests | Status |
|---|---|---|
| Research Loop | 11/11 | ALL PASS |
| Falsification | 5/5 | ALL PASS |
| Replication | 3/3 | ALL PASS |
| State Machine Attack | 5/5 | ALL PASS |
| Agent 1 Unavailable | 5/5 | ALL PASS |
| LLM Hallucination | 3/3 | ALL PASS |
| Heuristic Fallback | 4/4 | ALL PASS |
| **TOTAL** | **36/36** | **ALL PASS** |

## 1. Full Research Loop

### QUESTION → HYPOTHESIS → PLAN → AGENT 1 → EVIDENCE → FALSIFICATION → REPLICATION → EVALUATION

```
1. QUESTION
   Behavior: Indirect Object Identification
   Observation: GPT-2 predicts Mary in IOI prompt
   Question: Which attention head mediates name resolution?

2. HYPOTHESIS
   H1: L9H9 role in IOI [PROPOSED]

3. PLAN
   Experiment: Test L9H9
   Controls: L9H0, L9H1, L0H9
   Metrics: delta_logit, delta_prob, effect_size, specificity

4. AGENT 1
   delta_logit=1.23, specificity=3.5x

5. EVIDENCE
   State: TESTING

6. FALSIFICATION
   Falsification test with 20 repeats
   Controls: L0H0, L6H6

7. AGENT 1
   delta_logit=0.15, specificity=1.2x

8. REPLICATION
   State: REPLICATION

9. EVALUATION
   Supporting: 1, Contradicting: 1
   Assessment: INSUFFICIENT_EVIDENCE
```

## 2. Falsification Run

Falsification is a **core feature**, not an optional AI prompt.

- Hypothesis → "What would disprove this?" → Falsification experiment → Agent 1 → Real result
- Higher repeats (20 vs 10) for statistical power
- Null controls: L0H0, L6H6 (known irrelevant)
- Metrics include p_value for significance testing

## 3. Replication

Hypothesis lifecycle:

```
PROPOSED → PLANNED → TESTING → REPLICATION → EVALUATION → SUPPORTED/REFUTED/INSUFFICIENT
```

Evidence counts and confidence updated correctly.

## 4. State Machine Attack

All illegal transitions **rejected**:

| Attack | Result |
|---|---|
| PROPOSED → SUPPORTED | REJECTED |
| PROPOSED → REPLICATION | REJECTED |
| PLANNED → EVALUATION | REJECTED |
| PROPOSED → REFUTED | REJECTED |
| TESTING → SUPPORTED | REJECTED |

Required evidence for each transition:

| From | To | Required |
|---|---|---|
| PROPOSED | PLANNED | Experiment plan |
| PLANNED | TESTING | Agent 1 execution |
| TESTING | REPLICATION | Initial evidence |
| REPLICATION | EVALUATION | Replicated evidence |
| EVALUATION | SUPPORTED | Multiple supporting experiments |
| EVALUATION | REFUTED | Contradicting evidence |
| EVALUATION | INSUFFICIENT | Inconclusive evidence |

## 5. Agent 1 Unavailable

When Agent 1 is unavailable:

- No evidence → No conclusion
- Assessment: `INSUFFICIENT_EVIDENCE`
- Effect size: 0
- Specificity: 0
- Replications: 0
- Cannot promote hypothesis to SUPPORTED/REFUTED

## 6. LLM Hallucination Test

LLM claims **cannot** become scientific evidence:

- LLM → PROPOSAL (not measurement)
- LLM cannot set SUPPORTED directly
- Evidence requires Agent 1 execution
- Even hallucinated values are classified as INSUFFICIENT_EVIDENCE

## 7. Heuristic Fallback

Heuristics provide:

- Interpretation (supportive/falsifying/inconclusive)
- Rationale (natural language explanation)
- Suggestions (next steps)

Heuristics do **NOT** provide:

- `CAUSALLY_SUPPORTED` (requires Agent 1 evidence)
- Direct state promotion
- Scientific status

## Evidence Classification Rules

Classification must be based on Agent 1 evidence:

| Classification | Requirement |
|---|---|
| OBSERVED | Agent 1 measured effect |
| REPRODUCED | Multiple Agent 1 runs |
| SPECIFIC | Effect > control by >2x |
| CAUSALLY_SUPPORTED | Multiple specific replications |
| REFUTED | Agent 1 shows no effect |
| INSUFFICIENT | Inconclusive Agent 1 results |

## Architecture

```
Agent 2 (Reasoning)
├── Hypothesis Generator
├── Experiment Planner
├── Falsification Designer
├── Evidence Reasoner
├── Candidate Prioritizer
├── Research Loop Controller
└── LLM Integration (proposals only)

Agent 1 (Execution)
├── Model Layer (GPT-2, HuggingFace)
├── Execution (forward passes, hooks)
├── Interventions (ablation, patching)
├── Measurement (delta_logit, effect_size)
├── Controls (matched, random)
└── Discovery (head/MLP/neuron scan)
```

## Conclusion

Agent 2 **cannot** manufacture scientific conclusions from reasoning alone. The system enforces:

1. All measurements come from Agent 1
2. State transitions require evidence
3. LLM provides proposals, not facts
4. Heuristics provide interpretation, not status
5. Invalid transitions are rejected

**ACCEPTANCE CRITERION: MET**
