# Continuous Benchmarking

The platform runs a suite of interpretability benchmarks to prevent regressions and validate capabilities against real models (e.g. GPT-2 Small).

## Included Benchmarks
- **IOI (Indirect Object Identification)**: Validates circuit extraction logic.
- **Induction Heads**: Validates in-context learning mechanics.
- **Greater Than**: Evaluates mathematical circuitry.
- **Copy Task**: Checks zero and one-layer copying behaviors.
- **Arithmetic**: Verifies modulo and base-10 addition.
- **Factual Recall**: ROME/MEMIT tracing.

Each benchmark outputs an expected vs. observed metric comparison, resulting in a fidelity tier (Gold, Silver, Bronze, or Needs Investigation).
