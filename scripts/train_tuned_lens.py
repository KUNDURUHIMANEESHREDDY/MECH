"""Fit full-affine and diagonal tuned lenses on a real corpus and compare them.

Corpus: this repository's own prose (README + docs/*.md). That is disclosed in
the report rather than presented as a research corpus -- it is small, in-domain,
and far below what a published tuned lens uses.
"""
from __future__ import annotations
import glob, json, os, re, sys, time

# Run as `python scripts/train_tuned_lens.py`: Python puts scripts/ on sys.path,
# not the repository root, so the package import below would fail.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if os.getcwd() != _REPO_ROOT:
    os.chdir(_REPO_ROOT)

from backend.interpretability.algorithms.tuned_lens import (
    TunedLens, FULL_AFFINE, DIAGONAL_AFFINE,
)

MAX_CHUNKS = 120
MIN_WORDS, MAX_WORDS = 8, 40


def build_corpus() -> list[str]:
    paths = ["README.md"] + sorted(glob.glob("docs/*.md")) + sorted(glob.glob("docs/notes/*.md"))
    chunks: list[str] = []
    for path in paths:
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        # Strip code fences and tables: they tokenise badly and are not prose.
        text = re.sub(r"```.*?```", " ", text, flags=re.S)
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith(("|", "#", "-", "*", ">", "!")):
                continue
            for sentence in re.split(r"(?<=[.!?])\s+", line):
                words = sentence.split()
                if MIN_WORDS <= len(words) <= MAX_WORDS:
                    chunks.append(sentence)
        if len(chunks) >= MAX_CHUNKS * 3:
            break
    # Deterministic spread across the corpus rather than the first N lines.
    step = max(1, len(chunks) // MAX_CHUNKS)
    return chunks[::step][:MAX_CHUNKS]


def main() -> dict:
    prompts = build_corpus()
    print(f"corpus: {len(prompts)} chunks", flush=True)
    if not prompts:
        raise SystemExit("empty corpus")

    results = {}
    for kind, steps, lr in ((FULL_AFFINE, 100, 2e-3), (DIAGONAL_AFFINE, 100, 5e-3)):
        t0 = time.time()
        lens = TunedLens()
        report = lens.train(prompts, steps=steps, learning_rate=lr,
                            kind=kind, identity_decay=1e-3, holdout=True)
        elapsed = time.time() - t0
        print(f"{kind}: {report.get('status')} params={report.get('parameters_fitted')} "
              f"in_sample={report.get('layers_improved')}/{report.get('layers_total')} "
              f"holdout={report.get('layers_holdout_improved')} {elapsed:.1f}s", flush=True)

        out_dir = os.path.join("backend", "interpretability", "algorithms", "translators")
        os.makedirs(out_dir, exist_ok=True)
        saved = lens.save(os.path.join(out_dir, f"gpt2-tuned-lens-{kind}.pt"))
        report["artifact"] = saved
        report["wall_clock_s"] = round(elapsed, 1)
        report["corpus"] = {"source": "README.md + docs/**/*.md", "chunks": len(prompts)}
        results[kind] = report

        # Per-layer held-out reduction, the figure that means anything.
        for row in report.get("per_layer", []):
            if row.get("layer") in (1, 4, 8, 11):
                print(f"    L{row['layer']}: holdout reduction "
                      f"{row.get('holdout_kl_reduction_pct')}% "
                      f"(untuned {row.get('holdout_kl_untuned')} -> "
                      f"tuned {row.get('holdout_kl_tuned')})", flush=True)

    os.makedirs("docs/results", exist_ok=True)
    with open("docs/results/tuned_lens_full_affine.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print("wrote docs/results/tuned_lens_full_affine.json", flush=True)
    for kind, rep in results.items():
        print(f"COMPARE {kind}: artifact={rep['artifact'].get('sha256')}", flush=True)
    return results


if __name__ == "__main__":
    main()
