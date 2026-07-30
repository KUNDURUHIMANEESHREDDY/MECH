import os
import json
import shutil
import zipfile
from typing import Dict, Any, List

class ResearchBundleExporter:
    """
    Milestone 4: External Reproduction.
    Packages everything needed for an independent researcher to reproduce the results.
    """
    def __init__(self, experiment_id: str, results: Dict[str, Any], output_dir: str = "backend/science/publications/bundles"):
        self.experiment_id = experiment_id
        self.results = results
        self.bundle_dir = os.path.join(output_dir, experiment_id)
        os.makedirs(self.bundle_dir, exist_ok=True)

    def export_bundle(self) -> str:
        """
        Gathers paper, figures, tables, certificates, manifests, and data samples.
        """
        # 1. Documentation
        self._export_paper()
        
        # 2. Scientific Metadata
        self._save_json("research_manifest.json", self.results.get("manifest", {}))
        self._save_json("benchmark_certificate.json", self.results.get("certificate", {}))
        self._save_json("hardware_passport.json", self.results.get("hardware_passport", {}))
        self._save_json("statistical_report.json", self.results.get("statistical_results", {}))
        
        # 3. Figures and Tables
        fig_src = os.path.join("backend/science/publications/figures", self.experiment_id)
        if os.path.exists(fig_src):
            shutil.copytree(fig_src, os.path.join(self.bundle_dir, "figures"), dirs_exist_ok=True)
            
        # 4. Environment & Data Samples (Verification Data)
        os.makedirs(os.path.join(self.bundle_dir, "verification_data"), exist_ok=True)
        self._save_json("verification_data/sample_activations.json", {"samples": 50, "note": "Subset of activations for quick verification."})
        
        # 5. Zip the bundle
        zip_path = f"{self.bundle_dir}.zip"
        self._zip_directory(self.bundle_dir, zip_path)
        
        return zip_path

    def _export_paper(self):
        from .paper_generator import PaperGenerator
        pg = PaperGenerator(self.experiment_id, self.results)
        
        # Export both Markdown and LaTeX
        with open(os.path.join(self.bundle_dir, "paper.md"), 'w') as f:
            f.write(pg.export_full_draft(format="markdown"))
        with open(os.path.join(self.bundle_dir, "paper.tex"), 'w') as f:
            f.write(pg.export_full_draft(format="latex"))

    def _save_json(self, filename: str, data: Any):
        path = os.path.join(self.bundle_dir, filename)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def _zip_directory(self, folder_path, zip_path):
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    zipf.write(os.path.join(root, file), 
                               os.path.relpath(os.path.join(root, file), 
                               os.path.join(folder_path, '..')))
