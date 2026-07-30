from typing import Dict, Any

class StatisticalQualityScore:
    """
    Computes a summary metric for the statistical rigor of a benchmark.
    """

    @staticmethod
    def calculate_score(
        power: float,
        has_calibration: bool,
        has_correction: bool,
        has_bootstrap: bool,
        has_permutation: bool,
        has_effect_size: bool
    ) -> float:
        """
        Calculates a score out of 100 based on statistical practices.
        """
        score = 0.0

        # Power (Max 25%)
        # Scale power [0, 1] to [0, 25]
        score += min(power, 1.0) * 25.0

        # Binary checks (15% each)
        if has_calibration: score += 15.0
        if has_correction: score += 15.0
        if has_bootstrap: score += 15.0
        if has_permutation: score += 15.0
        if has_effect_size: score += 15.0

        return min(100.0, score)

    @classmethod
    def evaluate(cls, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expects a results dict containing keys like 'power', 'diagnostics', etc.
        """
        # Extract power. Assume 0 if not present or malformed.
        try:
             power = results.get("power", {}).get("observed_power", 0.0)
        except AttributeError:
             power = 0.0

        has_calibration = "calibration" in results
        has_correction = "multiple_correction" in results or "p_value_adjusted" in results
        has_bootstrap = "bootstrap" in results
        has_permutation = results.get("test_used") == "permutation"
        has_effect_size = "effect_sizes" in results

        score = cls.calculate_score(
            power, has_calibration, has_correction,
            has_bootstrap, has_permutation, has_effect_size
        )

        return {
            "score": float(score),
            "breakdown": {
                "power_contribution": float(min(power, 1.0) * 25.0),
                "calibration": has_calibration,
                "correction": has_correction,
                "bootstrap": has_bootstrap,
                "permutation": has_permutation,
                "effect_size": has_effect_size
            }
        }
