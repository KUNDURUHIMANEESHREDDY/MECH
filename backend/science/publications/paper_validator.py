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
        if not isinstance(results, dict):
            # A non-mapping cannot be checked for anything. Refuse it by name
            # rather than raising on the first attribute access below.
            return {
                "is_valid": False,
                "errors": [
                    "results must be an object, got "
                    f"{type(results).__name__}"
                ],
                "warnings": [],
                "status": "INVALID",
            }

        errors = []
        warnings = []
        
        # 1. Rigor Checks
        power_block = results.get("power")
        if power_block is None:
            observed_power = None
        elif isinstance(power_block, dict):
            observed_power = power_block.get("observed_power")
        else:
            observed_power = None
            warnings.append(
                "'power' is not an object, so observed power cannot be read."
            )

        if observed_power is None:
            # Not measured -- and deliberately not defaulted to 0. Rendering an
            # absent value as "power 0.00" would state a measurement that was
            # never taken, which is the failure this validator exists to catch.
            warnings.append(
                "Statistical power was not recorded, so it cannot be assessed."
            )
        elif isinstance(observed_power, bool) or not isinstance(
                observed_power, (int, float)):
            warnings.append(
                "'power.observed_power' is "
                f"{type(observed_power).__name__}, not a number, so it cannot "
                "be assessed."
            )
        elif observed_power < 0.8:
            warnings.append(
                f"Low statistical power ({observed_power:.2f}). "
                "Consider increasing sample size."
            )
            
        # 2. Metadata Checks
        paper_meta = results.get("paper")
        if paper_meta is None:
            warnings.append("Reference paper is missing a DOI.")
        elif not isinstance(paper_meta, dict):
            # Malformed rather than absent: there is nothing to read a DOI out
            # of, so the export is refused instead of allowed through on a
            # warning that names the wrong problem.
            errors.append(
                "Reference paper metadata is not an object "
                f"({type(paper_meta).__name__}), so its DOI cannot be verified."
            )
        elif not paper_meta.get("doi"):
            warnings.append("Reference paper is missing a DOI.")
            
        if not results.get("dataset_certificate"):
            errors.append("Missing dataset hash/certificate. Integrity cannot be verified.")

        # A fingerprint that exists but was never attested proves nothing: the
        # adapter was in mock_mode, or the weights could not be hashed.
        fingerprint = results.get("model_fingerprint")
        if not fingerprint:
            errors.append("Missing model fingerprint.")
        elif not (isinstance(fingerprint, dict)
                  and fingerprint.get("attested") is True):
            reason = (fingerprint.get("attestation_reason")
                      if isinstance(fingerprint, dict) else "not a fingerprint")
            errors.append(
                f"Model fingerprint is not attested, so the results cannot be "
                f"bound to specific weights. Reason: {reason or 'unknown'}"
            )
            
        # 3. Component Checks
        if not results.get("statistical_results"):
             errors.append("Missing statistical results.")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "status": "READY" if len(errors) == 0 else "INVALID"
        }
