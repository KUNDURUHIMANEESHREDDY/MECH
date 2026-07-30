from typing import Dict, Any, List

class LatexExporter:
    """
    Exports research papers to professional LaTeX format.
    """

    @staticmethod
    def to_latex(title: str, abstract: str, sections: Dict[str, str], bib_file: str) -> str:
        latex = [
            "\\documentclass{article}",
            "\\usepackage[utf8]{inputenc}",
            "\\usepackage{hyperref}",
            "\\usepackage{booktabs}",
            "\\usepackage{graphicx}",
            f"\\title{{{title}}}",
            "\\author{MECH Autonomous Research Framework}",
            "\\date{\\today}",
            "\\begin{document}",
            "\\maketitle",
            "\\begin{abstract}",
            abstract,
            "\\end{abstract}"
        ]
        
        # Order-sensitive section export
        order = ["Introduction", "Methodology", "Results", "Discussion", "Limitations", "Future Work", "Conclusion"]
        for section in order:
            if section in sections:
                latex.append(f"\\section{{{section}}}")
                latex.append(sections[section].replace("#", "").replace("*", "\\item")) # Simple MD to LaTeX conversion
                
        latex.append(f"\\bibliographystyle{{plain}}")
        latex.append(f"\\bibliography{{{bib_file}}}")
        latex.append("\\end{document}")
        
        return "\n".join(latex)
