"""Complete the full-vs-diagonal tuned-lens comparison.

The full-affine half already ran (661.8 s) and was saved to
`backend/interpretability/algorithms/translators/gpt2-tuned-lens-full_affine.pt`.
Re-training it just to produce a comparison file would burn another 11 minutes,
so this loads the saved artifact for its report and fits only the diagonal half.
"""
from __future__ import annotations
import json, os, sys, time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

from backend.interpretability.algorithms.tuned_lens import (
    TunedLens, FULL_AFFINE, DIAGONAL_AFFINE,
)
from scripts.train_tuned_lens import build_corpus

ART = "backend/interpretability/algorithms/translators"

def main():
    prompts = build_corpus()
    full_path = os.path.join(ART, f"gpt2-tuned-lens-{FULL_AFFINE}.pt")
    lens = TunedLens()
    loaded = lens.load(full_path)
    if loaded.get("status") != "loaded":
        raise SystemExit(f"cannot load full-affine artifact: {loaded}")
    full_report = lens.training_report
    print("loaded full_affine report:",
          full_report.get("parameters_fitted"), "params, holdout",
          full_report.get("layers_holdout_improved"), flush=True)

    t0 = time.time()
    diag_lens = TunedLens()
    diag = diag_lens.train(prompts, steps=100, learning_rate=5e-3,
                           kind=DIAGONAL_AFFINE, identity_decay=1e-3, holdout=True)
    elapsed = time.time() - t0
    print(f"diagonal_affine: {diag.get('status')} params={diag.get('parameters_fitted')} "
          f"in_sample={diag.get('layers_improved')}/{diag.get('layers_total')} "
          f"holdout={diag.get('layers_holdout_improved')} {elapsed:.1f}s", flush=True)
    saved = diag_lens.save(os.path.join(ART, f"gpt2-tuned-lens-{DIAGONAL_AFFINE}.pt"))
    diag["artifact"] = saved
    diag["wall_clock_s"] = round(elapsed, 1)
    diag["corpus"] = {"source": "README.md + docs/**/*.md", "chunks": len(prompts)}

    results = {FULL_AFFINE: full_report, DIAGONAL_AFFINE: diag}
    os.makedirs("docs/results", exist_ok=True)
    with open("docs/results/tuned_lens_full_affine.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print("wrote docs/results/tuned_lens_full_affine.json", flush=True)

    def mean_holdout(rep):
        vals = [r["holdout_kl_reduction_pct"] for r in rep.get("per_layer", [])
                if r.get("holdout_kl_reduction_pct") is not None]
        return sum(vals) / len(vals) if vals else None

    for kind, rep in results.items():
        print(f"MEAN_HOLDOUT {kind}: {mean_holdout(rep)}", flush=True)

if __name__ == "__main__":
    main()
