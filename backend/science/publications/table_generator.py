from typing import List, Dict, Any

class TableGenerator:
    """
    Generates publication-quality Markdown/LaTeX tables automatically.
    """

    @staticmethod
    def generate_benchmark_summary(results: List[Dict[str, Any]]) -> str:
        """
        Generates the Benchmark Summary table.
        Columns: Benchmark, Published, Observed, Delta, CI, Verdict
        """
        header = "| Benchmark | Published | Observed | Δ | CI | Verdict |\n"
        separator = "|---|---|---|---|---|---|\n"
        rows = []
        
        for res in results:
            benchmark = res.get("name", "Unknown")
            pub = res.get("published", 0.0)
            obs = res.get("observed", 0.0)
            delta = obs - pub
            ci = "✓" if res.get("ci_match", False) else "✗"
            verdict = res.get("verdict", "FAIL")
            
            rows.append(f"| {benchmark} | {pub:.3f} | {obs:.4f} | {delta:.4f} | {ci} | {verdict} |")
            
        return header + separator + "\n".join(rows)

    @staticmethod
    def generate_circuit_metrics(metrics: List[Dict[str, Any]]) -> str:
        """
        Generates the Circuit Metrics table.
        | Algorithm | Node F1 | Edge F1 | Necessity | Sufficiency |
        """
        header = "| Algorithm | Node F1 | Edge F1 | Necessity | Sufficiency |\n"
        separator = "|---|---|---|---|---|\n"
        rows = []
        
        for m in metrics:
            algo = m.get("algorithm", "ACDC")
            nf1 = f"{m.get('node_f1', 0):.2f}"
            ef1 = f"{m.get('edge_f1', 0):.2f}"
            nec = f"{m.get('necessity', 0):.2f}"
            suf = f"{m.get('sufficiency', 0):.2f}"
            
            rows.append(f"| {algo} | {nf1} | {ef1} | {nec} | {suf} |")
            
        return header + separator + "\n".join(rows)
