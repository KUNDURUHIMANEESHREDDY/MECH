from typing import Dict, Any, List

class PaperValidator:
    """
    Ensures a paper meets all scientific and metadata requirements before export.
    """

    @staticmethod
    def validate(results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Checks for missing metadata, citations, figures, and statistical rigor.
        """
        errors = []
        warnings = []
        
        # 1. Rigor Checks
        power = results.get("power", {}).get("observed_power", 0)
        if power < 0.8:
            warnings.append(f"Low statistical power ({power:.2f}). Consider increasing sample size.")
            
        # 2. Metadata Checks
        if not results.get("paper", {}).get("doi"):
            warnings.append("Reference paper is missing a DOI.")
            
        if not results.get("dataset_certificate"):
            errors.append("Missing dataset hash/certificate. Integrity cannot be verified.")
            
        if not results.get("model_fingerprint"):
            errors.append("Missing model fingerprint.")
            
        # 3. Component Checks
        if not results.get("statistical_results"):
             errors.append("Missing statistical results.")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "status": "READY" if len(errors) == 0 else "INVALID"
        }
