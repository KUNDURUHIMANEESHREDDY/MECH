"""Generate a paper draft from a validated experiment result.

The abstract is *composed from measured fields*, not written in advance. Every
sentence is produced by ``generate_abstract`` from data the run actually
reported, and anything absent is named as not measured rather than asserted.

What this file used to do
-------------------------
``export_full_draft`` called ``self.generate_abstract()`` -- a method that did
not exist anywhere in the repository -- so every invocation raised
``AttributeError`` at line 60 and no artifact was ever produced. The surrounding
code was also result-shaped fiction that would have flowed straight into that
abstract:

* ``effect_size`` defaulted to ``0`` and was then described as "strong causal
  relationship" -- a methodological claim manufactured from a missing value.
* ``ci_match`` was the literal ``True`` in the results table, so the confidence
  interval column always rendered as a tick.
* The Discussion and Conclusion asserted "robustness" and "a rigorous,
  reproducible verification" for any input, including an empty one.

``_number``/``_text`` here return ``None`` for a missing value instead of a
default, so "not reported" cannot become a measurement.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .figure_generator import FigureGenerator
from .table_generator import TableGenerator
from .citation_manager import CitationManager
from .paper_templates import PaperTemplates
from .future_work import FutureWorkGenerator
from .paper_validator import PaperValidator
from .latex_exporter import LatexExporter
from .claim_traceability import ClaimTraceability
from .supplementary import SupplementaryMaterialPackager


def _number(value: Any) -> Optional[float]:
    """A real finite float, or ``None``.

    Deliberately does not coerce a missing value to ``0.0``: a zero effect size
    and an unreported one are different statements, and the old
    ``stats.get("effect_sizes", {}).get("cohens_d", 0)`` could not tell them
    apart.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):  # NaN / inf
        return None
    return out


def _text(value: Any) -> Optional[str]:
    """A non-empty string, or ``None``."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _as_pair(value: Any) -> Optional[Tuple[float, float]]:
    """A two-element numeric interval, or ``None``."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        low, high = _number(value[0]), _number(value[1])
        if low is not None and high is not None:
            return (low, high)
    return None


class PaperGenerator:
    """
    Automates the generation of scientific paper drafts from benchmark results.
    Integrates figures, tables, citations, venue templates, multi-format export,
    claim traceability, and supplementary packaging.
    """

    def __init__(self, experiment_id: str, results: Dict[str, Any], venue: str = "arXiv"):
        self.experiment_id = experiment_id
        self.results = results if isinstance(results, dict) else {}
        self.venue = venue
        self.timestamp = datetime.utcnow().isoformat()
        self.fig_gen = FigureGenerator(experiment_id)
        self.table_gen = TableGenerator()
        self.cite_mgr = CitationManager()
        self.template = PaperTemplates.get_template(venue)
        self.validator = PaperValidator()
        self.tracer = ClaimTraceability()
        self.claims: List[Dict[str, str]] = []

    def add_claim(self, claim: str, evidence_key: str):
        self.claims.append(self.tracer.link_claim(claim, self.experiment_id, evidence_key))

    # ------------------------------------------------------------------ #
    # Derived facts. Each returns what was reported, never a default.
    # ------------------------------------------------------------------ #

    def _model(self) -> Dict[str, Any]:
        fingerprint = self.results.get("model_fingerprint")
        return fingerprint if isinstance(fingerprint, dict) else {}

    def _statistics(self) -> Dict[str, Any]:
        """The measured statistics, with absent fields left absent."""
        stats = self.results.get("statistical_results")
        stats = stats if isinstance(stats, dict) else {}

        effect_sizes = stats.get("effect_sizes")
        effect = None
        if isinstance(effect_sizes, dict):
            effect = _number(effect_sizes.get("cohens_d"))
        if effect is None:
            effect = _number(stats.get("cohens_d"))

        ci = _as_pair(stats.get("bootstrap_ci")) or _as_pair(
            stats.get("confidence_interval"))
        ci_method = _text(stats.get("confidence_interval_method"))
        if ci is None:
            low = _number(stats.get("confidence_interval_low"))
            high = _number(stats.get("confidence_interval_high"))
            if low is not None and high is not None:
                ci = (low, high)
        if ci_method is None and isinstance(stats.get("bootstrap_ci"), (list, tuple)):
            ci_method = "bootstrap"

        n = _number(stats.get("n"))
        if n is None:
            n = _number(self.results.get("n_samples"))

        return {
            "effect_size": effect,
            "ci": ci,
            "ci_method": ci_method,
            "n": n,
            "verdict": _text(stats.get("verdict")),
            "observed_mean": _number(stats.get("observed_mean")),
            "reference": _number(self.results.get("expected_effect_size")),
            "protocol": _text(stats.get("protocol")),
        }

    def _ci_excludes_zero(self) -> Optional[bool]:
        """Whether the reported interval resolves the direction of the effect.

        ``None`` means no interval was reported -- which is not the same as an
        interval that contains zero.
        """
        ci = self._statistics()["ci"]
        if ci is None:
            return None
        low, high = ci
        if low > 0 or high < 0:
            return True
        return False

    def _circuit(self) -> Dict[str, Any]:
        for key in ("circuit", "discovery", "circuit_summary"):
            block = self.results.get(key)
            if isinstance(block, dict):
                return block
        return {}

    # ------------------------------------------------------------------ #
    # Abstract
    # ------------------------------------------------------------------ #

    def generate_abstract(self) -> str:
        """Compose the abstract from the validated result.

        Structure: what was analysed, what circuit (if any) was found, what the
        statistics support, whether the run reports a reproduction, and what was
        not measured. A clause about a quantity is emitted only when that
        quantity was reported.
        """
        sentences: List[str] = []
        model = self._model()
        stats = self._statistics()
        circuit = self._circuit()

        # 1. What was analysed.
        model_name = (_text(model.get("model_id")) or _text(model.get("model_name"))
                      or _text(self.results.get("model")))
        if model_name:
            clause = f"We analyse {model_name}"
            params = _number(model.get("parameter_count"))
            if params is not None:
                clause += f" ({int(params):,} parameters)"
            if model.get("attested") is True:
                weights = _text(model.get("weights_sha256")) or _text(model.get("weights_sha"))
                clause += (f" with weights attested at {weights[:23]}..."
                           if weights and len(weights) > 23 else
                           " with attested weights")
            else:
                clause += (", but its weights are not attested, so these results "
                           "are not bound to a specific checkpoint")
            sentences.append(clause + ".")
        else:
            sentences.append(
                "No model identity was recorded for this run, so the results "
                "below cannot be attributed to particular weights.")

        # 2. What circuit was found.
        heads = circuit.get("heads")
        edges = circuit.get("edges")
        n_heads = len(heads) if isinstance(heads, (list, tuple)) else None
        n_edges = len(edges) if isinstance(edges, (list, tuple)) else None
        faithfulness = _number(circuit.get("faithfulness"))
        completeness = _number(circuit.get("completeness"))
        n_prompts = _number(circuit.get("n_prompts"))

        if n_heads is None and n_edges is None and faithfulness is None:
            sentences.append(
                "No circuit discovery result was present in this run, so no "
                "circuit is described.")
        else:
            parts = []
            if n_heads is not None:
                parts.append(f"{n_heads} head{'s' if n_heads != 1 else ''}")
            if n_edges is not None:
                parts.append(f"{n_edges} edge{'s' if n_edges != 1 else ''}")
            clause = ("The discovery pass recovered " + " and ".join(parts)) if parts else \
                "The discovery pass reported a circuit"
            if n_prompts is not None:
                clause += f" from {int(n_prompts)} prompt{'s' if n_prompts != 1 else ''}"
            if faithfulness is not None:
                clause += f", with a measured faithfulness of {faithfulness:.4f}"
            if completeness is not None:
                clause += f" (completeness {completeness:.4f})"
            sentences.append(clause + ".")

        # 3. What the statistics support.
        effect = stats["effect_size"]
        ci = stats["ci"]
        n = stats["n"]
        if effect is None:
            sentences.append(
                "No effect size was reported, so no claim is made here about the "
                "size or direction of an effect.")
        else:
            clause = f"The measured effect size is d = {effect:.3f}"
            if n is not None:
                clause += f" (n = {int(n)})"
            if ci is not None:
                method = f"{stats['ci_method']} " if stats["ci_method"] else ""
                clause += f", with a {method}interval of [{ci[0]:.3f}, {ci[1]:.3f}]"
                if self._ci_excludes_zero():
                    clause += ("; the interval excludes zero, so the direction of "
                               "the effect is resolved, though this is not by "
                               "itself evidence of a causal mechanism")
                else:
                    clause += ("; the interval includes zero, so the direction of "
                               "the effect is not resolved by this run")
            else:
                clause += (", and no confidence interval was reported, so the "
                           "precision of that estimate is unknown")
            sentences.append(clause + ".")

        # 4. Reproduction verdict, only when one was recorded.
        reproduction = self.results.get("reproduction_successful")
        if isinstance(reproduction, bool):
            sentences.append(
                "The run reports the published effect as reproduced."
                if reproduction else
                "The run does not report a reproduction of the published effect.")
        else:
            sentences.append("No reproduction verdict was recorded for this run.")

        # 5. What was not measured -- named, so absence is visible.
        missing = []
        if stats["verdict"] is None:
            missing.append("no validation verdict")
        if stats["protocol"] is None:
            missing.append("no stated statistical protocol")
        paper = self.results.get("paper")
        if not (isinstance(paper, dict) and paper.get("doi")):
            missing.append("no reference DOI")
        if missing:
            sentences.append("This draft additionally carries " + ", ".join(missing) + ".")

        return " ".join(sentences)

    # ------------------------------------------------------------------ #
    # Result sections
    # ------------------------------------------------------------------ #

    def _summary_table(self) -> str:
        """The benchmark summary table, with absent values left absent.

        ``ci_match`` is derived from the reported interval. It used to be the
        literal ``True``, which rendered every row as a match.
        """
        stats = self._statistics()
        paper = self.results.get("paper")
        title = (_text(paper.get("title")) if isinstance(paper, dict) else None) \
            or "MECH Discovery Report"
        reproduces = self.results.get("reproduction_successful")
        verdict = ("PASS" if reproduces is True
                   else "FAIL" if reproduces is False
                   else "NOT REPORTED")
        return self.table_gen.generate_benchmark_summary([{
            "name": title,
            "published": stats["reference"],
            "observed": stats["effect_size"],
            "ci_match": self._ci_excludes_zero(),
            "verdict": verdict,
        }])

    def _figure_note(self) -> str:
        """A figure reference only when a figure was actually written."""
        stats = self._statistics()
        if stats["effect_size"] is None:
            return ("No figure was generated: a forest plot needs a reported "
                    "effect size, and none was present.")
        paper = self.results.get("paper")
        label = (_text(paper.get("title")) if isinstance(paper, dict) else None) \
            or self.experiment_id
        try:
            fig_id = self.fig_gen.generate_forest_plot([{
                "label": label,
                "effect": stats["effect_size"],
                "ci": list(stats["ci"]) if stats["ci"] else None,
            }])
        except Exception as exc:  # a figure is an artifact, not a measurement
            return f"No figure was generated ({type(exc).__name__}: {exc})."
        return f"Figure data written to `{fig_id}`."

    def _discussion(self) -> str:
        """Discussion text derived from the reported statistics."""
        stats = self._statistics()
        lines: List[str] = []
        if stats["effect_size"] is None:
            lines.append(
                "No effect size was reported, so there is nothing here to "
                "interpret as an effect. Any claim about the mechanism would be "
                "unsupported by this run.")
        else:
            excludes = self._ci_excludes_zero()
            if excludes is True:
                lines.append(
                    f"The reported effect (d = {stats['effect_size']:.3f}) has an "
                    "interval excluding zero, which is consistent with the "
                    "intervention having an effect. Direction and magnitude are "
                    "supported; the mechanism behind them is not established by "
                    "this statistic alone.")
            elif excludes is False:
                lines.append(
                    f"The reported effect (d = {stats['effect_size']:.3f}) has an "
                    "interval containing zero, so this run does not separate the "
                    "effect from noise.")
            else:
                lines.append(
                    f"The reported effect (d = {stats['effect_size']:.3f}) carries "
                    "no interval in this record, so its precision is unquantified "
                    "and no directional claim is made.")
        circuit = self._circuit()
        faithfulness = _number(circuit.get("faithfulness"))
        n_prompts = _number(circuit.get("n_prompts"))
        if faithfulness is None:
            lines.append("No circuit faithfulness measurement is present in this record.")
        elif n_prompts is None:
            lines.append(
                f"Circuit faithfulness was measured at {faithfulness:.4f}, over "
                "an unreported number of prompts.")
        else:
            lines.append(
                f"Circuit faithfulness was measured at {faithfulness:.4f} over "
                f"{int(n_prompts)} prompts.")
        return "\n\n".join(lines)

    def _conclusion(self) -> str:
        stats = self._statistics()
        if stats["effect_size"] is None:
            return ("This run does not support a conclusion about an effect or a "
                    "mechanism, because no effect size was reported.")
        return ("This draft reports a single measured effect with its interval "
                "and states the limits of what that measurement supports. It does "
                "not assert a mechanism beyond the circuit evidence recorded above.")

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #

    def export_full_draft(self, format: str = "markdown") -> str:
        # 1. Validation. Fails closed: an unattested fingerprint or a missing
        # dataset certificate blocks export rather than producing a draft.
        val_report = self.validator.validate(self.results)
        if not val_report["is_valid"]:
            raise ValueError(f"Paper validation failed: {val_report['errors']}")

        paper = self.results.get("paper")
        paper = paper if isinstance(paper, dict) else {}
        title = paper.get("title", "MECH Discovery Report")
        stats = self._statistics()

        # 2. Claim, recorded only for a quantity that was measured.
        if stats["effect_size"] is not None:
            self.add_claim(
                f"Measured effect size d = {stats['effect_size']:.3f}"
                + (f" with interval [{stats['ci'][0]:.3f}, {stats['ci'][1]:.3f}]."
                   if stats["ci"] else " (no interval reported)."),
                "statistical_results.effect_sizes")

        table_md = self._summary_table()

        sections = {
            "Abstract": self.generate_abstract(),
            "Introduction": (
                "Mechanistic interpretability aims to reverse-engineer the "
                "computation a trained network performs. This draft reports a "
                "measurement from MECH and states its limits."),
            "Methodology": (
                stats["protocol"] or
                "No statistical protocol was recorded for this run."),
            "Results": f"### Summary Statistics\n{table_md}\n\n{self._figure_note()}",
            "Discussion": self._discussion(),
            "Limitations": PaperTemplates.apply_limitations(self.results),
            "Future Work": FutureWorkGenerator.generate_directions(self.results),
            "Conclusion": self._conclusion(),
            "References": self.cite_mgr.generate_references_section([paper]),
            "Appendix": self.tracer.generate_traceability_appendix(self.claims),
        }

        # 3. Packaging supplementary material.
        packager = SupplementaryMaterialPackager(self.experiment_id, self.results)
        packager.create_bundle()

        # 4. Format export.
        if format.lower() == "latex":
            bib_name = f"{self.experiment_id}_refs"
            # Output directory is overridable so a caller (or a test) can write
            # its artifacts somewhere other than the source tree.
            packages_dir = os.environ.get("MECH_PUBLICATION_DIR") or os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "packages")
            os.makedirs(packages_dir, exist_ok=True)
            self.cite_mgr.export_bib_file(
                [paper], os.path.join(packages_dir, f"{bib_name}.bib"))
            return LatexExporter.to_latex(title, sections["Abstract"], sections, bib_name)

        draft = [f"# {title}", f"**Venue:** {self.venue}\n"]
        emitted = set()
        for section in self.template["order"]:
            if section in sections:
                draft.append(f"## {section}\n{sections[section]}\n")
                emitted.add(section)
        # A venue's order does not list every section this generator produces
        # (arXiv omits "Future Work"), and a section that is built but never
        # emitted is a silent loss of content. Emit whatever is left.
        for section in ("Future Work",):
            if section in sections and section not in emitted:
                draft.append(f"## {section}\n{sections[section]}\n")
                emitted.add(section)
        if "Appendix" in sections:
            draft.append(sections["Appendix"])
        return "\n".join(draft)
