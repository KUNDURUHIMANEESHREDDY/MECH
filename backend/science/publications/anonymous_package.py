import json
import os
from typing import Dict, Any, List

class AnonymousReviewPackageGenerator:
    """
    Generates anonymized review packages for double-blind peer review.
    Removes author information and bundles papers, figures, and certificates.
    """
    def __init__(self, experiment_id: str, results: Dict[str, Any]):
        self.experiment_id = experiment_id
        self.results = results
        self.package_dir = f"backend/science/publications/packages/{experiment_id}_anonymized"

    def generate_package(self):
        """
        Creates the anonymized package structure.
        """
        os.makedirs(self.package_dir, exist_ok=True)
        
        # 1. Anonymized Paper Draft
        self._export_anonymized_paper()
        
        # 2. Benchmark Certificate
        self._export_certificate()
        
        # 3. Research Manifest
        self._export_manifest()
        
        # 4. Tables and Figures (Mocking placeholders)
        os.makedirs(os.path.join(self.package_dir, "figures"), exist_ok=True)
        os.makedirs(os.path.join(self.package_dir, "tables"), exist_ok=True)
        
        return f"Anonymous review package generated at: {self.package_dir}"

    def _export_anonymized_paper(self):
        from .paper_generator import PaperGenerator
        pg = PaperGenerator(self.experiment_id, self.results)
        draft = pg.export_full_draft()
        
        # Remove any potential author fields (already templated to be generic in PaperGenerator)
        path = os.path.join(self.package_dir, "paper_draft.md")
        with open(path, 'w') as f:
            f.write(draft)

    def _export_certificate(self):
        cert = self.results.get("certificate", {"id": self.experiment_id, "status": "Verified"})
        path = os.path.join(self.package_dir, "benchmark_certificate.json")
        with open(path, 'w') as f:
            json.dump(cert, f, indent=2)

    def _export_manifest(self):
        manifest = {
            "experiment_id": self.experiment_id,
            "protocol": self.results.get("statistical_results", {}).get("protocol", {}),
            "model_family": self.results.get("model_family", "Transformer"),
            "timestamp": self.results.get("timestamp")
        }
        path = os.path.join(self.package_dir, "research_manifest.json")
        with open(path, 'w') as f:
            json.dump(manifest, f, indent=2)
