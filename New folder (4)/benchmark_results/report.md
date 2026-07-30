# Benchmark Report — 2026-07-28T18:14:15.045761Z

- **Overall Coverage**: 93.3%
- **Overall Fidelity**: 98.86%
- **Models Tested**: 5

## Performance Dashboard
| Model | Throughput (TPS) | Mean Latency (ms) | Peak VRAM (MB) | GPU Util % | FLOPs (Est) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| gpt2_small | 1704498228.5 | 0.00 | 0.0 | 0.0% | 2.48e+13 |
| gpt2_medium | 1295026676.2 | 0.00 | 0.0 | 0.0% | 7.10e+13 |
| gemma_2b | 1539860610.9 | 0.00 | 0.0 | 0.0% | 5.32e+13 |
| llama_3b | 1293941686.9 | 0.00 | 0.0 | 0.0% | 5.32e+13 |
| qwen2_5_05b | 856647155.3 | 0.00 | 0.0 | 0.0% | 5.32e+13 |

## Model: gpt2_small (stub)
- **Execution Mode**: ExecutionMode.MOCK
| Task | Fidelity % | Published | Observed | 95% CI | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ioi | 99.78% | 97.0 | 96.78 | [93.33, 100.24] | ✅ PASS |
| induction_heads | 98.89% | 95.0 | 96.05 | [93.35, 98.75] | ✅ PASS |
| greater_than | 99.44% | 89.0 | 89.5 | [84.59, 94.4] | ✅ PASS |
| logit_lens | 99.79% | 82.0 | 82.17 | [77.84, 86.5] | ✅ PASS |
| sae | 98.92% | 91.0 | 91.98 | [89.6, 94.36] | ✅ PASS |
| copy_task | 99.6% | 93.0 | 93.37 | [89.92, 96.82] | ✅ PASS |
| arithmetic | 99.81% | 84.0 | 83.84 | [79.28, 88.4] | ✅ PASS |
| factual_recall | 99.41% | 79.0 | 78.53 | [72.84, 84.22] | ✅ PASS |
| universality | 99.56% | 82.0 | 81.64 | [74.05, 89.23] | ✅ PASS |

## Model: gpt2_medium (stub)
- **Execution Mode**: ExecutionMode.MOCK
| Task | Fidelity % | Published | Observed | 95% CI | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ioi | 98.79% | 97.0 | 98.17 | [95.55, 100.8] | ✅ PASS |
| induction_heads | 97.61% | 95.0 | 92.73 | [89.13, 96.33] | ✅ PASS |
| greater_than | 98.46% | 89.0 | 87.63 | [82.36, 92.9] | ✅ PASS |
| logit_lens | 97.77% | 82.0 | 80.17 | [75.66, 84.68] | ✅ PASS |
| sae | 99.86% | 91.0 | 90.87 | [88.35, 93.4] | ✅ PASS |
| copy_task | 98.97% | 93.0 | 93.96 | [90.66, 97.26] | ✅ PASS |
| arithmetic | 99.15% | 84.0 | 84.72 | [80.26, 89.18] | ✅ PASS |
| factual_recall | 99.11% | 79.0 | 78.3 | [72.58, 84.01] | ✅ PASS |
| universality | 95.73% | 82.0 | 85.5 | [78.6, 92.4] | ✅ PASS |

## Model: gemma_2b (stub)
- **Execution Mode**: ExecutionMode.MOCK
| Task | Fidelity % | Published | Observed | 95% CI | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ioi | 98.81% | 97.0 | 98.15 | [95.51, 100.79] | ✅ PASS |
| induction_heads | 98.68% | 95.0 | 96.25 | [93.62, 98.89] | ✅ PASS |
| greater_than | 97.6% | 89.0 | 86.87 | [81.46, 92.27] | ✅ PASS |
| logit_lens | 97.36% | 82.0 | 79.83 | [75.29, 84.37] | ✅ PASS |
| copy_task | 99.49% | 93.0 | 92.53 | [88.88, 96.17] | ✅ PASS |
| arithmetic | 95.85% | 84.0 | 87.48 | [83.38, 91.58] | ✅ PASS |
| factual_recall | 98.93% | 79.0 | 78.16 | [72.43, 83.88] | ✅ PASS |
| universality | 98.96% | 82.0 | 81.15 | [73.48, 88.82] | ✅ PASS |

## Model: llama_3b (stub)
- **Execution Mode**: ExecutionMode.MOCK
| Task | Fidelity % | Published | Observed | 95% CI | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ioi | 98.88% | 97.0 | 95.91 | [92.04, 99.79] | ✅ PASS |
| induction_heads | 98.29% | 95.0 | 93.38 | [89.93, 96.82] | ✅ PASS |
| greater_than | 97.99% | 89.0 | 90.79 | [86.16, 95.42] | ✅ PASS |
| logit_lens | 99.69% | 82.0 | 82.25 | [77.93, 86.58] | ✅ PASS |
| copy_task | 99.56% | 93.0 | 93.41 | [89.97, 96.85] | ✅ PASS |
| arithmetic | 99.32% | 84.0 | 84.57 | [80.09, 89.05] | ✅ PASS |
| factual_recall | 96.29% | 79.0 | 81.93 | [76.6, 87.26] | ✅ PASS |
| universality | 99.79% | 82.0 | 82.17 | [74.67, 89.67] | ✅ PASS |

## Model: qwen2_5_05b (stub)
- **Execution Mode**: ExecutionMode.MOCK
| Task | Fidelity % | Published | Observed | 95% CI | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ioi | 99.77% | 97.0 | 96.77 | [93.31, 100.24] | ✅ PASS |
| induction_heads | 98.91% | 95.0 | 96.03 | [93.33, 98.74] | ✅ PASS |
| greater_than | 99.06% | 89.0 | 89.83 | [85.0, 94.67] | ✅ PASS |
| logit_lens | 100.0% | 82.0 | 82.0 | [77.65, 86.34] | ✅ PASS |
| copy_task | 99.31% | 93.0 | 92.36 | [88.68, 96.04] | ✅ PASS |
| arithmetic | 99.35% | 84.0 | 83.45 | [78.85, 88.06] | ✅ PASS |
| factual_recall | 99.58% | 79.0 | 79.33 | [73.72, 84.94] | ✅ PASS |
| universality | 99.99% | 82.0 | 82.0 | [74.48, 89.53] | ✅ PASS |

