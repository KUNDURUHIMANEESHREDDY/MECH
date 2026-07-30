import json
import os
from typing import Dict, Any
from .verification_engine import VerificationEngine
from .hardware_rigor import HardwarePassport

class IndependentReproductionValidator:
    """
    Milestone 4: Independent Reproduction.
    Allows external researchers to validate research bundles (papers, certificates, results).
    """

    @staticmethod
    def validate_bundle(bundle_path: str) -> Dict[str, Any]:
        """
        Validates an 'Anonymous Research Bundle' for reproducibility.
        """
        # 1. Load Bundle Components
        manifest_path = os.path.join(bundle_path, "research_manifest.json")
        results_path = os.path.join(bundle_path, "statistical_report.json")
        cert_path = os.path.join(bundle_path, "benchmark_certificate.json")
        
        if not all(os.path.exists(p) for p in [manifest_path, results_path, cert_path]):
            return {"status": "INVALID_BUNDLE", "message": "Missing required bundle components."}
            
        with open(manifest_path, 'r') as f: manifest = json.load(f)
        with open(results_path, 'r') as f: results = json.load(f)
        with open(cert_path, 'r') as f: cert = json.load(f)
        
        # 2. Integrity Verification
        is_authentic = VerificationEngine.verify_integrity(results, cert)
        
        # 3. Environment Check
        # Compare researcher's current hardware to original passport
        current_env = HardwarePassport.capture_passport()
        original_env = manifest.get("hardware_passport", {})
        
        env_match = current_env.get("python") == original_env.get("python")
        
        return {
            "is_authentic": is_authentic,
            "environment_alignment": "MATCH" if env_match else "MISMATCH",
            "current_hardware": current_env,
            "original_hardware": original_env,
            "can_reproduce": is_authentic and env_match
        }
