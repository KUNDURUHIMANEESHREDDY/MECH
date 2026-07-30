from typing import Dict, Any, List

class CitationManager:
    """
    Manages scientific citations and generates BibTeX and formatted references.
    """
    
    @staticmethod
    def generate_bibtex(paper_meta: Dict[str, str]) -> str:
        """
        Generates a BibTeX entry from paper metadata.
        """
        key = paper_meta.get("id", "paper_id")
        bib = [
            f"@article{{{key},",
            f"  title = {{{paper_meta.get('title')}}},",
            f"  author = {{{paper_meta.get('authors')}}},",
            f"  year = {{{paper_meta.get('year')}}},",
            f"  journal = {{{paper_meta.get('journal', 'arXiv preprint')}}},"
        ]
        if "url" in paper_meta:
            bib.append(f"  url = {{{paper_meta['url']}}},")
        if "doi" in paper_meta:
            bib.append(f"  doi = {{{paper_meta['doi']}}}")
        
        return "\n".join(bib) + "\n}"

    @staticmethod
    def format_apa(paper_meta: Dict[str, str]) -> str:
        authors = paper_meta.get("authors", "Unknown")
        year = paper_meta.get("year", "n.d.")
        title = paper_meta.get("title", "Unknown Title")
        journal = paper_meta.get("journal", "arXiv preprint")
        return f"{authors} ({year}). {title}. *{journal}*."

    @staticmethod
    def format_ieee(paper_meta: Dict[str, str]) -> str:
        authors = paper_meta.get("authors", "Unknown")
        title = paper_meta.get("title", "Unknown Title")
        year = paper_meta.get("year", "Unknown Year")
        return f"{authors}, \"{title},\" {year}."

    def generate_references_section(self, papers: List[Dict[str, str]], style: str = "APA") -> str:
        """
        Generates a formatted list of references.
        """
        section = ["## References\n"]
        for paper in papers:
            if style == "APA":
                section.append(self.format_apa(paper))
            elif style == "IEEE":
                section.append(self.format_ieee(paper))
            else:
                section.append(self.format_apa(paper))
        return "\n".join(section)

    def export_bib_file(self, papers: List[Dict[str, str]], filepath: str):
        with open(filepath, 'w') as f:
            for paper in papers:
                f.write(self.generate_bibtex(paper) + "\n\n")
