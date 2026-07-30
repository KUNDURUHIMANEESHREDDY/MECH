# Benchmark Report — 2026-07-28T11:50:08.010883Z

- **Overall Coverage**: 100.0%
- **Overall Fidelity**: 99.45%
- **Models Tested**: 1

## Model: gpt2_small (huggingface)
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

