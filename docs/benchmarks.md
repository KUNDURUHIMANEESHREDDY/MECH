# Benchmarks

MECH runs interpretability benchmarks against **real GPT-2 small weights** and reports
measured results. Where no live executor is connected, the benchmark returns
`status: "unavailable"` with a reason rather than a fabricated score.

## The live path

```bash
curl -X POST http://127.0.0.1:8000/api/benchmarks/run \
     -H 'Content-Type: application/json' -d '{"name":"IOI"}'
```

```jsonc
{
  "status": "completed",
  "benchmark_name": "IOI",
  "score": 1.0,
  "pass_rate": 1.0,
  "eval_samples": 6,
  "provenance": "live",
  "field_provenance": { "score": "live", "pass_rate": "live", "eval_samples": "live" }
}
```

**Only `IOI` has a live executor today.** Every other benchmark name returns:

```json
{ "status": "unavailable",
  "reason": "No live benchmark executor is connected for '<name>'" }
```

This is deliberate — see `backend/validation/benchmark_runner.py:23`.

## The catalogue

`GET /api/benchmarks` returns `IOI`, `Induction`, `SAE`, `ACDC`, `PathPatching` with
`provenance: "reference"`. The wider catalogue in `backend/benchmarking/benchmark_tasks.py`
carries nine tasks, each with a **published baseline** used for comparison:

| Task | Metric | Published value | Source |
|------|--------|-----------------|--------|
| IOI | circuit recovery fidelity | 0.97 | Wang et al. 2022 |
| Induction Heads | induction score | 0.95 | Olsson et al. 2022 |
| Greater-Than | circuit faithfulness | 0.89 | Hanna et al. 2023 |
| Logit Lens | final layer accuracy | 0.82 | nostalgebraist 2020 |
| SAE | reconstruction R² | 0.91 | Cunningham et al. 2023 |
| Copy Task | copy fidelity | 0.93 | Elhage et al. 2021 |
| Arithmetic | arithmetic faithfulness | 0.84 | Stolfo et al. 2023 |
| Factual Recall | causal effect score | 0.79 | Meng et al. 2022 |
| Universality | alignment fidelity | 0.82 | Chughtai et al. 2023 |

**These are literature baselines, not MECH measurements.** They are the reference column an
observed metric is compared against.

> ⚠️ **Do not read the catalogue scores as results.** `BenchmarkTaskExecutor` currently adds
> Gaussian noise around the published value (`score = ref + rng.gauss(0, 0.015)`) and its
> `_run_tl` / `_run_hf` / `_run_ollama` paths import the library but execute no algorithm.
> Only the `IOI` route through `/api/benchmarks/run` is genuinely live.

## Measured results

Reproduce with `python scripts/capture_results.py` → `docs/results/capture.json`.

| Measurement | Result |
|-------------|--------|
| IOI clean accuracy (3 templates) | **1.0** (3/3) |
| IOI corrupted flip rate (3 templates) | **1.0** (3/3) |
| Baseline clean logit difference | `2.1681` |
| Heads screened by zero-ablation | 144 / 144 |
| Heads whose ablation supports IOI / opposes it | 68 / 76 |
| Strongest supporting head (signed effect) | `L2H0`, `+0.9391` (43 % of baseline) |
| Steering flip (layer 10, α=40) | ` the` → ` Paris` |
| GPT-2 perplexity on held-out prose | 44.9 |

See the [README](../README.md#results) for the full tables and figures.

## Known limitations

- **The live IOI benchmark runs on 6 prompts.** `score: 1.0` on 6 samples is nearly
  meaningless statistically. The paper's panel is 100 templates; port it before quoting the
  number.
- **The confidence interval is closed-form**, `1.96 · √(score(1−score)/n)`, not bootstrapped.
- **No persistence.** `GET /api/benchmarks/run` returns a result; there is no result-history
  endpoint. The Benchmark dashboard labels this boundary in the UI rather than hiding it.
- `backend/benchmarking/benchmark_tasks.py:382,466` call `logger.warning(...)` without
  importing `logging`, raising `NameError` on the fallback path. Fix before relying on the
  catalogue executor.
