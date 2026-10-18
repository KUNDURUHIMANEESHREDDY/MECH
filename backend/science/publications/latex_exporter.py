from typing import Dict, Any, List


def _markdown_to_latex(body: str) -> str:
    r"""Convert the small Markdown subset the sections use into valid LaTeX.

    The previous version did ``body.replace("*", "\\item")``, which emitted
    ``\item`` outside any ``itemize`` environment -- so the produced document
    could not compile. Bullet runs are now wrapped, and inline emphasis is
    escaped rather than turned into list items.
    """
    out: List[str] = []
    in_items = False

    def close_items():
        nonlocal in_items
        if in_items:
            out.append("\\end{itemize}")
            in_items = False

    for raw_line in (body or "").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith(("* ", "- ")):
            if not in_items:
                out.append("\\begin{itemize}")
                in_items = True
            text = stripped[2:].strip()
            # Inline emphasis: **bold** / *italic*.
            while "**" in text:
                text = text.replace("**", "\\textbf{", 1)
                if "**" in text:
                    text = text.replace("**", "}", 1)
                else:
                    text += "}"
            out.append("\\item " + text.replace("_", "\\_"))
            continue
        close_items()
        if not stripped:
            out.append("")
            continue
        if stripped.startswith("### "):
            out.append(f"\\subsubsection{{{stripped[4:].strip()}}}")
        elif stripped.startswith("## "):
            out.append(f"\\subsection{{{stripped[3:].strip()}}}")
        elif stripped.startswith("# "):
            out.append(f"\\section{{{stripped[2:].strip()}}}")
        else:
            out.append(stripped.replace("_", "\\_"))

    close_items()
    return "\n".join(out)


class LatexExporter:
    """
    Exports research papers to professional LaTeX format.
    """

    @staticmethod
    def to_latex(title: str, abstract: str, sections: Dict[str, str], bib_file: str) -> str:
        def esc(value: str) -> str:
            return (value or "").replace("\\", "\\textbackslash{}")

        latex = [
            "\\documentclass{article}",
            "\\usepackage[utf8]{inputenc}",
            "\\usepackage{hyperref}",
            "\\usepackage{booktabs}",
            "\\usepackage{graphicx}",
            f"\\title{{{esc(title)}}}",
            "\\author{MECH Autonomous Research Framework}",
            "\\date{\\today}",
            "\\begin{document}",
            "\\maketitle",
            "\\begin{abstract}",
            _markdown_to_latex(abstract),
            "\\end{abstract}",
        ]

        # Order-sensitive section export. `Abstract` is already emitted above;
        # `References` is produced by \bibliography, not as a section body.
        order = ["Introduction", "Methodology", "Results", "Discussion",
                 "Limitations", "Future Work", "Conclusion"]
        for section in order:
            if section in sections:
                latex.append(f"\\section{{{section}}}")
                latex.append(_markdown_to_latex(sections[section]))

        latex.append("\\bibliographystyle{plain}")
        latex.append(f"\\bibliography{{{esc(bib_file)}}}")
        latex.append("\\end{document}")

        return "\n".join(latex)
