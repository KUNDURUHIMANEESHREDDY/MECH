from typing import Dict, Any, List


class PaperTemplates:
    """
    Provides specific formatting templates for different scientific venues.
    """

    TEMPLATES = {
        "arXiv": {
            "order": ["Abstract", "Introduction", "Methodology", "Results", "Discussion", "Limitations", "Conclusion", "References"],
            "style": "Standard Markdown"
        },
        "NeurIPS": {
            "order": ["Abstract", "Introduction", "Related Work", "Methodology", "Experiments", "Results", "Limitations", "Discussion", "Conclusion", "References"],
            "style": "Latex-inspired Markdown"
        },
        "ICML": {
            "order": ["Abstract", "Introduction", "Methodology", "Results", "Discussion", "Related Work", "Conclusion", "Limitations", "References"],
            "style": "Two-column Markdown"
        }
    }

    @classmethod
    def get_template(cls, venue: str) -> Dict[str, Any]:
        return cls.TEMPLATES.get(venue, cls.TEMPLATES["arXiv"])

    @staticmethod
    def apply_limitations(results: Dict[str, Any]) -> str:
        """Limitations derived from what the run actually recorded.

        This used to say ``Our study primarily focuses on Unknown`` and
        ``The sample size (N=0) is sufficient for the detected effect`` for any
        input -- asserting both the model identity and the *sufficiency* of a
        sample size it did not have. It also asserted SAE findings unconditionally.
        Each limitation is now emitted only when the corresponding input exists.
        """
        results = results if isinstance(results, dict) else {}
        stats = results.get("statistical_results")
        stats = stats if isinstance(stats, dict) else {}

        lims: List[str] = []

        model = results.get("model")
        if isinstance(model, str) and model.strip():
            lims.append(
                f"Our study focuses on {model.strip()}. Results may not "
                "generalize to larger models or different architectures.")
        else:
            lims.append(
                "No model identity was recorded for this run, so the scope of "
                "these results is unknown and they cannot be generalised.")

        n = stats.get("n")
        if n is None:
            n = results.get("n_samples")
        if isinstance(n, (int, float)) and not isinstance(n, bool):
            lims.append(
                f"The reported sample size (N={int(n)}) bounds the statistical "
                "power of this analysis. Adequacy was not assessed here.")
        else:
            lims.append(
                "No sample size was reported, so the statistical power of this "
                "analysis is unknown.")

        # Only claim an SAE limitation when an SAE result is present.
        sae = results.get("sae") or results.get("sae_results")
        if isinstance(sae, dict) and sae:
            lims.append(
                "SAE findings are contingent on the specific checkpoint and "
                "training coefficients used.")

        hardware = results.get("hardware_passport") or results.get("hardware")
        if isinstance(hardware, dict) and hardware:
            lims.append(
                "Runtime metrics are hardware-dependent and may vary across GPU "
                "configurations.")

        return "\n".join(f"* {l}" for l in lims)