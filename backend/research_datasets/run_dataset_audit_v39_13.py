"""Dataset Audit — a smoke test of the dataset tooling. NOT a scientific audit.

Exercises the diff engine, the exporter, revalidation and citation metadata.

Two things it does *not* do, which the "Audit" name used to imply:

* It does not audit a real dataset. The semantic diff runs over two hand-written
  dictionaries, so the counts it prints describe those literals only.
* It does not verify anything about the golden datasets. Nothing here checks a
  manifest hash or a signature; `verify_golden_datasets.py` does that.

For real integrity checks run:

    python frontend/scripts/verify_golden_datasets.py
"""

import sys
import os
import json

# Ensure project root is in path
sys.path.insert(0, os.getcwd())

from backend.research_datasets.dataset_manager import DatasetManager
from backend.research_datasets.dataset_diff_engine import DatasetDiffEngine
from backend.research_datasets.dataset_exporter import DatasetExporter

def run_audit():
    print("Starting Phase 39.13: Dataset Evolution & Bundle Audit...")
    manager = DatasetManager(data_dir="backend/research_datasets")
    diff_engine = DatasetDiffEngine()
    exporter = DatasetExporter()

    # 1. Semantic Diffing
    #
    #    These two datasets are hand-written fixtures, not versions loaded from
    #    anywhere, and the diff below is a real diff over them. The printed
    #    counts ("1 added, 1 modified") are therefore a true description of
    #    *these two literals* and of nothing else -- they are not a statement
    #    about any dataset in the store. Labelled as a smoke test so nobody
    #    reads the numbers as a finding about real data.
    print("\n[1/4] Executing Semantic Diffing (SYNTHETIC FIXTURES, not real versions)...")
    data_v1 = {"dataset_id": "IOI-v1.0", "prompts": [{"id": "p1", "clean": "Alice gave a drink to Bob"}]}
    data_v2 = {"dataset_id": "IOI-v1.1", "prompts": [
        {"id": "p1", "clean": "Alice handed a drink to Bob"}, # Modified
        {"id": "p2", "clean": "Charlie gave a book to Diana"} # Added
    ]}

    report = diff_engine.compare_datasets(data_v1, data_v2)
    print(f"  - Prompts: {report.prompts.added} added, {report.prompts.modified} modified.")
    print(f"  - Risk Level: {report.risk_level}")
    print(f"  - Affected Benchmarks: {report.affected_benchmarks}")

    # 2. Bundle Generation
    print("\n[2/4] Generating Reproduction Bundle...")
    zip_path = None
    try:
        zip_path = exporter.export("IOI-Canonical-100", manager, output_dir="exports_audit")
        print(f"  - Bundle Created: {zip_path}")
        print(f"  - Bundle Size: {os.path.getsize(zip_path)} bytes")
    except Exception as e:
        print(f"  - Bundle Export Failed: {e}")

    # 3. Impact & Revalidation
    print("\n[3/4] Triggering Impact Revalidation...")
    manager.trigger_revalidation("IOI-Canonical-100")
    print("  - Audit Log Entry: CREATED")

    # 4. Citation metadata.
    #
    #    This used to print "CITATION.cff & citation.bib: INCLUDED IN BUNDLE"
    #    without opening the bundle. Two problems: nothing was checked, and the
    #    claim was wrong -- the exporter writes CITATION.cff and no citation.bib,
    #    so the bundle did not contain what the audit reported containing.
    print("\n[4/4] Verifying Citation Metadata...")
    citation_files = {}
    if zip_path and os.path.exists(zip_path):
        import zipfile

        with zipfile.ZipFile(zip_path, "r") as archive:
            names = archive.namelist()
        for wanted in ("CITATION.cff", "citation.bib"):
            citation_files[wanted] = wanted in names
            state = "PRESENT" if wanted in names else "ABSENT"
            print(f"  - {wanted}: {state}")

        expected = {"CITATION.cff"}
        missing = [n for n in expected if n not in names]
        if missing:
            print(f"  - WARNING: expected bundle members missing: {', '.join(missing)}")
    else:
        print("  - not checked (bundle export did not produce a file)")

    print("\n" + "="*40)
    print("PHASE 39.13 AUDIT COMPLETE")
    if "citation.bib" in citation_files and not citation_files["citation.bib"]:
        print("NOTE: citation.bib is not produced by DatasetExporter; the bundle "
              "carries CITATION.cff only.")

if __name__ == "__main__":
    run_audit()
