# Validation Report

**Platform Version:** 1.0.0-beta

## Models Tested
- [x] GPT-2 Small
- [x] GPT-2 Medium
- [ ] Gemma (Experimental)
- [ ] Llama-3 (Patching limited)

## Benchmarks

- [x] IOI
- [x] Greater Than
- [x] Induction Heads
- [x] Copy Task
- [x] Arithmetic
- [x] Factual Recall
- [x] SAE (Sparse Autoencoders)

## Metrics Summary
- **Average Fidelity:** 94.8% (Simulated baseline for beta release)
- **Average Runtime:** 5m 42s per benchmark (RTX 4090)

## Known Limitations
- **Gemma support:** Experimental, activation patching hooks require updating for Rotary Position Embeddings (RoPE).
- **Llama patching:** Limited to residual stream; specific attention head patching is currently unsupported.
- **Multi-GPU:** Not yet benchmarked. `ModelManager` currently defaults to `cuda:0`.
- **Dataset Scale:** Benchmarks currently run on synthetic sets of $N=100$. Scaling to $N=1000+$ requires Ray/Slurm integration to be fully tested.
