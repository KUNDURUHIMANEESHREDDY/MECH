import os
import json
import shutil
from typing import Dict, Any

class SupplementaryMaterialPackager:
    """
    Packages all research artifacts into a comprehensive supplementary material bundle.
    """
    def __init__(self, experiment_id: str, results: Dict[str, Any], output_root: str = "backend/science/publications/packages"):
        self.experiment_id = experiment_id
        self.results = results
        self.package_path = os.path.join(output_root, f"{experiment_id}_supplementary")
        os.makedirs(self.package_path, exist_ok=True)

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
            shutil.copytree(fig_src, os.path.join(self.package_path, "figures"), dirs_exist_ok=True)
            
        # 3. Raw Data (Simplified Mock)
        os.makedirs(os.path.join(self.package_path, "raw_data"), exist_ok=True)
        self._save_json("raw_data/activation_samples_summary.json", {"samples": 100, "metric": "logit_diff"})
        
        return self.package_path

    def _save_json(self, filename: str, data: Any):
        path = os.path.join(self.package_path, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
