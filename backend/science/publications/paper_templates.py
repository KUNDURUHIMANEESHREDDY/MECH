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
        """
        Automatically generates a limitations section based on experiment constraints.
        """
        model = results.get("model", "Unknown")
        n = results.get("n_samples", results.get("statistical_results", {}).get("current_n", 0))
        
        lims = [
            f"Our study primarily focuses on {model}. Results may not generalize to larger models or different architectures.",
            f"The sample size (N={n}) is sufficient for the detected effect but may not capture long-tail phenomena.",
            "SAE findings are contingent on the specific checkpoint and training coefficients used.",
            "Hardware-dependent runtime metrics may vary across different GPU configurations."
        ]
        
        return "## Limitations\n\n" + "\n".join([f"* {l}" for l in lims])
