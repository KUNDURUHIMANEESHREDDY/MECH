import os
import json
import shutil
from typing import Any, Dict, List, Optional


class SupplementaryMaterialPackager:
    """
    Packages all research artifacts into a comprehensive supplementary material bundle.
    """

    def __init__(self, experiment_id: str, results: Dict[str, Any],
                 output_root: str = "backend/science/publications/packages"):
        self.experiment_id = experiment_id
        self.results = results if isinstance(results, dict) else {}
        self.package_path = os.path.join(output_root, f"{experiment_id}_supplementary")
        os.makedirs(self.package_path, exist_ok=True)

    def _raw_data_summary(self) -> Dict[str, Any]:
        """Describe the raw data that was actually supplied.

        This used to write ``{"samples": 100, "metric": "logit_diff"}`` -- a
        literal sample count and metric name for every run, including runs with
        no data at all, sitting in a file named ``raw_data``. The honest form is
        a count derived from the traces and the metrics those traces name, with
        an explicit reason when nothing was supplied.
        """
        traces: Optional[List[Any]] = None
        for key in ("raw_traces", "traces", "samples"):
            candidate = self.results.get(key)
            if isinstance(candidate, list):
                traces = candidate
                break

        if traces is None:
            return {
                "measured": False,
                "provenance": "unavailable",
                "reason": ("No raw samples were supplied with this result, so no "
                           "sample count or metric is reported."),
            }

        metrics = sorted({
            str(t.get("metric"))
            for t in traces
            if isinstance(t, dict) and t.get("metric")
        })
        return {
            "measured": True,
            "provenance": "live",
            "samples": len(traces),
            "metrics": metrics,
            "reason": "" if traces else "The supplied trace list was empty.",
        }

    def create_bundle(self):
        """
        Gathers and organizes all required artifacts.
        """
        # 1. Metadata and Reports
        self._save_json("research_manifest.json", self.results.get("manifest", {}))
        self._save_json("statistical_report.json", self.results.get("statistical_results", {}))
        self._save_json("benchmark_certificate.json", self.results.get("certificate", {}))
        self._save_json("environment_snapshot.json", self.results.get("hardware_passport", {}))

        # 2. Figures and Tables
        fig_src = os.path.join("backend/science/publications/figures", self.experiment_id)
        if os.path.exists(fig_src):
            shutil.copytree(fig_src, os.path.join(self.package_path, "figures"),
                            dirs_exist_ok=True)

        # 3. Raw Data provenance (counted, or explicitly absent)
        os.makedirs(os.path.join(self.package_path, "raw_data"), exist_ok=True)
        self._save_json("raw_data/activation_samples_summary.json",
                        self._raw_data_summary())

        return self.package_path

    def _save_json(self, filename: str, data: Any):
        path = os.path.join(self.package_path, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
