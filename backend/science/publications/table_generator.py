from typing import List, Dict, Any, Optional


def _fmt(value: Any, places: int = 4) -> str:
    """Format a number that may be absent.

    ``res.get("published", 0.0)`` used to invent a zero for a missing value, so
    a benchmark with no published baseline rendered as a measured ``0.000`` and
    the delta column read as a real difference. An absent value is stated as
    absent instead.
    """
    if value is None or isinstance(value, bool):
        return "not reported"
    try:
        return f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return "not reported"


def _ci_cell(ci_match: Optional[bool]) -> str:
    """Tri-state: a match, a non-match, or no interval to judge.

    This used to be ``"yes" if res.get("ci_match", False)``, and callers passed
    the literal ``True``, so the column always rendered as a tick.
    """
    if ci_match is True:
        return "yes"
    if ci_match is False:
        return "no"
    return "no interval"


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
        header = "| Benchmark | Published | Observed | Delta | CI | Verdict |\n"
        separator = "|---|---|---|---|---|---|\n"
        rows = []

        for res in results:
            benchmark = res.get("name") or "Unknown"
            published = res.get("published")
            observed = res.get("observed")
            try:
                delta = f"{float(observed) - float(published):.4f}"
            except (TypeError, ValueError):
                delta = "not reported"
            rows.append(
                f"| {benchmark} | {_fmt(published, 3)} | {_fmt(observed)} | "
                f"{delta} | {_ci_cell(res.get('ci_match'))} | "
                f"{res.get('verdict') or 'NOT REPORTED'} |"
            )

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
            rows.append(
                f"| {m.get('algorithm') or 'Unknown'} | "
                f"{_fmt(m.get('node_f1'), 2)} | {_fmt(m.get('edge_f1'), 2)} | "
                f"{_fmt(m.get('necessity'), 2)} | {_fmt(m.get('sufficiency'), 2)} |"
            )

        return header + separator + "\n".join(rows)
