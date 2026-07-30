import json
import hashlib
from typing import Dict, Any, List, Optional

class VerificationEngine:
    """
    Advanced Verification Engine for research integrity.
    Verifies certificates, manifests, fingerprints, and statistical tolerances.
    """

    @staticmethod
    def verify_integrity(data: Dict[str, Any], certificate: Dict[str, Any]) -> bool:
        """
        Verifies that a data dictionary matches its signed certificate hash.
        """
        if not certificate or "hash" not in certificate:
            return False
            
        # Standardized serialization for hashing
        data_str = json.dumps(data, sort_keys=True)
        calculated_hash = hashlib.sha256(data_str.encode('utf-8')).hexdigest()
        
        return calculated_hash == certificate["hash"]

    @classmethod
    def generate_report(
        cls, 
        current_run: Dict[str, Any], 
        reference_manifest: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Emits a comprehensive verification report.
        """
        report = {
            "checks": {},
            "overall_status": "FAILED"
        }
        
        # 1. Dataset Integrity
        dataset_ok = cls.verify_integrity(
            current_run.get("dataset", {}), 
            current_run.get("dataset_certificate", {})
        )
        report["checks"]["Dataset"] = "PASS" if dataset_ok else "FAIL"

        # 2. Model Fingerprint
        model_match = current_run.get("model_fingerprint") == reference_manifest.get("expected_model_fingerprint")
        report["checks"]["Model Fingerprint"] = "PASS" if model_match else "FAIL"

        # 3. Environment Context
        env_match = current_run.get("env_hash") == reference_manifest.get("expected_env_hash")
        report["checks"]["Environment"] = "PASS" if env_match else "FAIL"

        # 4. Statistical Rigor
        stats = current_run.get("statistical_results", {})
        p_val = stats.get("p_value_raw", 1.0)
        power = stats.get("power", {}).get("observed_power", 0)
        alpha = reference_manifest.get("protocol", {}).get("alpha", 0.05)
        power_thresh = reference_manifest.get("protocol", {}).get("power_threshold", 0.8)
        
        stats_ok = p_val < alpha and power >= power_thresh
        report["checks"]["Statistics"] = "PASS" if stats_ok else "FAIL"

        # 5. Regression Check
        # (This is handled by regression_suite, but we include status here)
        report["checks"]["Regression"] = current_run.get("regression_status", "NOT_RUN")

        # Final Overall Decision
        if all(v == "PASS" for v in report["checks"].values()):
            report["overall_status"] = "VERIFIED"
        
        return report

    @staticmethod
    def check_tolerance(original: float, current: float, tolerance: float) -> bool:
        if original == 0: return abs(current) < tolerance
        return (abs(original - current) / abs(original)) <= tolerance
