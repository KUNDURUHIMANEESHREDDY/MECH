"""DEMONSTRATION of the reproducibility pipeline. NOT a scientific validation.

Renamed from ``run_reproducibility_audit.py``.

What that file did
------------------
It was named "audit" and described itself as "the high-fidelity validation
pipeline", then:

* loaded the dataset, and on **any** failure set
  ``os.environ["MECH_BYPASS_HASH_CHECK"] = "1"`` and reloaded -- disabling
  integrity checking for every later load in the process, and never restoring
  it;
* forced ``get_adapter("gpt2-small", mock_mode=True)``, so the "model
  fingerprint" bound into the certificate described fixture weights;
* generated its 100 "observations" with the expression
  ``[0.88 + (0.02 * (0.5 - i/100.0)) for i in range(100)]`` -- an arithmetic
  ramp -- and passed them to ``validate_benchmark``, which computes a bootstrap
  CI, Cohen's d and a verdict over them;
* supplied ``published=0.880, registry=0.878, baseline=0.875, parity=0.9999``
  as literals, while the validator separately hardcoded
  ``"published": 0.88  # Mocked lookup`` into the certificate;
* printed ``VERIFIED``, a ``Reproducibility Score``, a ``Digital Signature`` and
  ``Portable Bundle: VERIFIED``.

Every number that came out was an input. A PASS was guaranteed by construction,
and the statistics machinery reported the guarantee as a finding.

What this file does instead
---------------------------
Runs the real benchmark suite against real weights and prints a banner stating
what the run is. There is no fallback that produces numbers: if the model cannot
load, the script raises.

Usage
-----
    python -m backend.science.reproducibility.run_reproducibility_demo

To have the manifest signed, supply a key kept outside the repository:

    set MECH_SIGNING_KEY_PATH=C:\\keys\\mech_signing_ed25519.pem

Without one the manifest is written unsigned with the reason recorded. That is
the intended behaviour: a certificate signed with a key that ships in the source
attests to nothing.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.getcwd())

from backend.science.reproducibility.benchmark_runner import BenchmarkRunner
from backend.science.reproducibility.scientific_validator import ScientificValidator
from backend.research_datasets.dataset_manager import DatasetManager

BANNER = """\
================================================================
  MODE: DEMONSTRATION
  NOT A SCIENTIFIC VALIDATION
================================================================
  Artifacts from this script must not be cited as evidence or used
  to support a claim about any model. Use BenchmarkRunner directly
  for results you intend to stand behind.
================================================================\
"""

#: Left None so the certificate records that no literature comparison was
#: supplied. The old script passed 0.880 as a literal while the validator
#: hardcoded 0.88 internally, so the run always "agreed" with a figure nobody
#: looked up.
PUBLISHED_REFERENCE = None

#: Per-pipeline metric ids that the reference registry knows about, with the
#: pipeline key they come from. A pipeline whose metric has no registry entry
#: cannot be validated, and is reported as such rather than validated against an
#: invented baseline.
VALIDATABLE = {
    "ioi": "ioi_faithfulness",
    "induction_heads": "induction_score",
}


def load_dataset(ds_manager: DatasetManager) -> List[Dict[str, Any]]:
    """Load the dataset, reporting exactly which integrity checks ran.

    No `except` sets a bypass and retries. If verification fails this raises: the
    dataset manager's message names the failing check and the escape hatch, and
    that hatch now requires an explicit environment value rather than being
    flipped on by a script for convenience.
    """
    print("[1/4] Loading Golden Dataset (integrity verification)...")
    # Loaded by folder alias ("ioi"), not by the canonical manifest id
    # ("IOI-Canonical-100"). The manifest registers the dataset under the
    # canonical id but the file lives in datasets/ioi/, and `load` resolves the
    # path from the id it is given -- so the canonical id raises FileNotFoundError
    # while the alias loads. The old script "handled" that FileNotFoundError by
    # setting the hash bypass and retrying with the same id, which raised again.
    prompts = ds_manager.load("ioi")

    status = ds_manager.last_integrity_status
    for name, state in sorted(status.get("checks", {}).items()):
        print(f"  - {name}: {state}")
    print(f"  - integrity_verified: {status.get('integrity_verified')}")

    if not status.get("integrity_verified"):
        raise RuntimeError(
            f"Dataset loaded without full integrity verification ({status}). "
            "Refusing to continue."
        )
    not_recorded = [n for n, s in status.get("checks", {}).items() if s == "not_recorded"]
    if not_recorded:
        print(f"  - no hash recorded in the manifest for: {', '.join(sorted(not_recorded))}")
        print("    (a gap in the manifest, not a verification failure -- but also "
              "not evidence of integrity)")

    print(f"  - ioi: {len(prompts)} prompts loaded")
    return prompts


def run_benchmarks() -> Dict[str, Any]:
    """Run the live benchmark suite. No mock mode, no fallback, no fabrication."""
    print("\n[2/4] Running benchmark suite against real weights...")

    # mock_mode=False is the class default. It is passed explicitly so the line
    # guaranteeing real weights is visible rather than implied.
    report = BenchmarkRunner(mock_mode=False).run_all(model_id="gpt2-small", seed=42)

    measured = not_run = errored = 0
    for name, result in sorted(report["reports"].items()):
        status = result.get("status")
        metrics = result.get("metrics") or {}
        fidelity = metrics.get("circuit_faithfulness", metrics.get("induction_score"))

        if status == "NOT_RUN":
            not_run += 1
            reason = (result.get("error") or "no reason recorded").split(".")[0]
            print(f"  - {name}: NOT_RUN ({reason})")
        elif status == "ERROR":
            errored += 1
            print(f"  - {name}: ERROR")
        else:
            measured += 1
            vram = result.get("peak_vram_gb")
            vram_txt = f", peak VRAM {vram} GB" if vram is not None else ""
            print(f"  - {name}: measured {fidelity}{vram_txt}")

    print(f"  - {measured} measured, {not_run} not run, {errored} errored")

    if measured == 0:
        raise RuntimeError(
            "No benchmark produced a measurement, and this script has no "
            "fabricated fallback. There is nothing to report."
        )
    return report


def validate(validator: ScientificValidator, report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validate the pipelines whose reference and per-sample data both exist.

    `validate_benchmark` requires a registry entry for the metric id and a list
    of per-sample observations. The pipelines report aggregate metrics rather
    than per-sample series, so what is validated is what can honestly be
    validated: for IOI, the per-prompt `clean_logit_diff` traces the pipeline
    recorded, which are real model outputs.

    No arithmetic ramp. No aggregate repeated n times to fake a sample size.
    """
    print("\n[3/4] Statistical validation...")

    results: List[Dict[str, Any]] = []
    for pipeline_id, metric_id in VALIDATABLE.items():
        result = report["reports"].get(pipeline_id)
        if not result or result.get("status") != "PASS":
            print(f"  - {pipeline_id}: skipped (did not run)")
            continue

        observations = _per_sample(result)
        if not observations:
            print(f"  - {pipeline_id}: skipped (no per-sample observations recorded)")
            continue

        # `min_n` is the registry's own 40, not something to lower to make n=10
        # look adequate. At n=10 the CI is wide and the validator is expected to
        # say so.
        stats = validator.validate_benchmark(
            metric_id=metric_id,
            observed_data=observations,
            patch_success_rate=result["metrics"].get("patch_success_rate"),
        )
        # `unmet` lists both False (measured and failed) and "not_assessed" (never
        # measured), because a caller reading the verdict needs to know which of
        # the two stopped it.
        unmet = [k for k, v in (stats.criteria_met or {}).items() if v is not True]
        if unmet:
            detail = ", ".join(
                f"{k}={stats.criteria_met[k]}" for k in sorted(unmet))
            print(f"    unmet: {detail}")
        results.append({
            "id": pipeline_id,
            "published": PUBLISHED_REFERENCE,
            "stats": stats,
        })
        print(f"  - {pipeline_id} ({metric_id}): {stats.verdict}, "
              f"n={stats.n}, mean={stats.observed_mean:.4f}, "
              f"CI=[{stats.bootstrap_ci[0]:.3f}, {stats.bootstrap_ci[1]:.3f}]")

    return results


def _per_sample(result: Dict[str, Any]) -> List[float]:
    """Extract genuinely per-sample measurements from a pipeline result.

    Currently only the IOI pipeline records per-prompt traces. The other
    pipelines return aggregates, and an aggregate is not a sample: repeating one
    measured mean n times would produce a bootstrap CI of zero width and a
    perfect-looking t-statistic out of a single number.
    """
    # `patch_success` is the per-prompt 0/1 outcome of the faithfulness
    # measurement, so its mean is a faithfulness rate in [0, 1] -- the same unit
    # as the registry's `expected_mean`.
    #
    # `clean_logit_diff` is *not*, and an earlier version of this function used it.
    # That fed a logit gap (mean ~2.58) into a validator comparing against a
    # faithfulness rate of 0.880, and the resulting FAIL was a unit mismatch
    # rather than a finding. Mean patch_success here is 0.80 over n=10, which is
    # the pipeline's own `patch_success_rate` of 80.0.
    traces = result.get("raw_traces") or []
    return [
        1.0 if t.get("patch_success") else 0.0
        for t in traces
        if isinstance(t, dict) and "patch_success" in t
    ]


def main() -> None:
    print(BANNER)

    validator = ScientificValidator()
    ds_manager = DatasetManager(data_dir="backend/research_datasets")

    load_dataset(ds_manager)
    report = run_benchmarks()
    results = validate(validator, report)

    if not results:
        print("\nNo pipeline produced validatable per-sample data; "
              "no artifacts written.")
        return

    print("\n[4/4] Generating research manifest & certificate...")
    from backend.science.models.gpt2_adapter import GPT2Adapter

    output_dir = "research_demo_artifacts"
    artifacts = validator.generate_validation_artifacts(
        benchmark_results=results,
        adapter=GPT2Adapter(variant="small", mock_mode=False),
        dataset_id="ioi",
        output_dir=output_dir,
    )

    with open(artifacts["manifest"], "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    print(f"  - Reproducibility Score: {manifest['reproducibility_score']:.1f}/100")
    print(f"  - Merkle Root SHA: {manifest['root_sha256']}")
    print(f"  - Signature: {manifest.get('signature_status')}")
    print(f"  - Location: {os.path.abspath(output_dir)}")

    print("\n" + "=" * 64)
    print("DEMONSTRATION COMPLETE.")
    print("These artifacts describe a real run, but this is a demo entry point.")
    print("Use BenchmarkRunner directly for results you intend to cite.")
    print("=" * 64)


if __name__ == "__main__":
    main()