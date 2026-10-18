"""The paper generator runs end to end, and claims only what was measured.

The defect this pins
--------------------
``PaperGenerator.export_full_draft`` called ``self.generate_abstract()``, which
was defined nowhere in the repository, so *every* invocation raised
``AttributeError`` and no paper artifact was ever produced. The surrounding code
also manufactured result-shaped content that would have gone straight into that
abstract:

* ``effect_size`` defaulted to ``0`` and was then described as a "strong causal
  relationship".
* ``ci_match`` was the literal ``True``, so the interval column always ticked.
* Discussion and Conclusion asserted "robustness" and "a rigorous, reproducible
  verification" for any input.
* ``supplementary.py`` wrote ``{"samples": 100, "metric": "logit_diff"}`` into
  ``raw_data/`` for every run.
"""

from __future__ import annotations

import ast
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.science.publications.paper_generator import PaperGenerator  # noqa: E402
from backend.science.publications.table_generator import TableGenerator  # noqa: E402

PUBLICATIONS_DIR = os.path.join(REPO_ROOT, "backend", "science", "publications")


def _validated(**overrides):
    """A result that clears the validator, with overrides applied."""
    results = {
        "paper": {"title": "IOI Circuit Reproduction", "doi": "10.48550/arXiv.2211.00593"},
        "dataset_certificate": {"sha256": "a" * 64},
        "model_fingerprint": {
            "attested": True, "model_id": "gpt2", "parameter_count": 124439808,
            "weights_sha256": "b" * 64,
        },
        "statistical_results": {
            "effect_sizes": {"cohens_d": 0.8035},
            "bootstrap_ci": [0.61, 0.94],
            "n": 144,
            "verdict": "PASS",
            "protocol": "Head-level zero-ablation over 144 heads.",
        },
        "circuit": {"heads": ["L0H10"], "edges": [1, 2], "faithfulness": 0.8035,
                    "n_prompts": 100},
        "expected_effect_size": 0.86,
        "reproduction_successful": True,
    }
    results.update(overrides)
    return results


def _abstract(**overrides):
    return PaperGenerator("exp_1", _validated(**overrides)).generate_abstract()


# ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ the crash regression ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬


def test_generate_abstract_exists_and_is_callable():
    """The named defect: the method did not exist, so export raised."""
    assert callable(getattr(PaperGenerator, "generate_abstract", None))


def test_export_full_draft_returns_markdown_and_latex(tmp_path, monkeypatch):
    """End to end: the draft is produced in both formats, not raised."""
    monkeypatch.chdir(tmp_path)
    # Keep the bib file out of the source tree.
    monkeypatch.setenv("MECH_PUBLICATION_DIR", str(tmp_path / "pub"))
    md = PaperGenerator("exp_ok", _validated()).export_full_draft("markdown")
    assert md.startswith("# IOI Circuit Reproduction"), md[:120]
    for section in ("Abstract", "Methodology", "Results", "Discussion", "Conclusion"):
        assert f"## {section}" in md, section

    tex = PaperGenerator("exp_ok", _validated()).export_full_draft("latex")
    assert "\\documentclass{article}" in tex
    assert "\\begin{abstract}" in tex and "\\end{abstract}" in tex
    assert tex.count("\\begin{itemize}") == tex.count("\\end{itemize}")


def test_latex_export_does_not_emit_stray_items():
    """`\\item` outside `itemize` was invalid LaTeX for any bulleted body."""
    from backend.science.publications.latex_exporter import _markdown_to_latex

    rendered = _markdown_to_latex("## Head\n\n* one\n* two\n\nplain")
    assert rendered.count("\\begin{itemize}") == 1
    assert rendered.count("\\end{itemize}") == 1
    assert rendered.count("\\item") == 2


# ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ no claim without a measurement ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬


def test_missing_effect_size_makes_no_causal_claim():
    text = _abstract(statistical_results={"verdict": "PASS"})
    assert "strong causal relationship" not in text
    assert "No effect size was reported" in text
    assert "d = 0.000" not in text, "a missing effect size became a zero"


def test_interval_containing_zero_is_not_read_as_an_effect():
    text = _abstract(statistical_results={
        "cohens_d": 0.31, "bootstrap_ci": [-0.12, 0.74], "n": 20, "verdict": "PASS"})
    assert "includes zero" in text
    assert "direction of the effect is not resolved" in text
    assert "excludes zero" not in text


def test_interval_excluding_zero_still_hedges_mechanism():
    text = _abstract()  # CI [0.61, 0.94]
    assert "excludes zero" in text
    assert "not by itself evidence of a causal mechanism" in text


def test_missing_interval_is_named_rather_than_assumed():
    text = _abstract(statistical_results={"cohens_d": 0.5, "n": 8, "verdict": "PASS"})
    assert "no confidence interval was reported" in text
    assert "precision" in text


def test_absent_circuit_is_stated_not_invented():
    text = _abstract(circuit={})
    assert "No circuit discovery result was present" in text


def test_unattested_fingerprint_still_blocks_export():
    """The validator's fail-closed behaviour must survive this change."""
    results = _validated()
    results["model_fingerprint"] = {"attested": False, "attestation_reason": "mock_mode"}
    with pytest.raises(ValueError) as excinfo:
        PaperGenerator("exp_bad", results).export_full_draft()
    assert "not attested" in str(excinfo.value)


def test_missing_dataset_certificate_still_blocks_export():
    results = _validated()
    results.pop("dataset_certificate")
    with pytest.raises(ValueError) as excinfo:
        PaperGenerator("exp_bad", results).export_full_draft()
    assert "dataset hash" in str(excinfo.value).lower()


# ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ ci_match is derived, not asserted ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬


def test_ci_match_column_is_tri_state():
    """The literal `True` rendered every row as a match."""
    yes = TableGenerator.generate_benchmark_summary(
        [{"name": "x", "published": 1.0, "observed": 2.0, "ci_match": True}])
    no = TableGenerator.generate_benchmark_summary(
        [{"name": "x", "published": 1.0, "observed": 2.0, "ci_match": False}])
    absent = TableGenerator.generate_benchmark_summary(
        [{"name": "x", "published": 1.0, "observed": 2.0, "ci_match": None}])
    assert " yes " in yes
    assert " no " in no
    assert "no interval" in absent


def test_missing_numbers_are_not_rendered_as_zeros():
    """`res.get("published", 0.0)` invented a measured-looking 0.000."""
    table = TableGenerator.generate_benchmark_summary(
        [{"name": "x", "observed": 1.5, "ci_match": None, "verdict": "PASS"}])
    assert "0.000" not in table
    assert "not reported" in table


def test_paper_generator_table_derives_ci_from_the_interval(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    md = PaperGenerator("exp_ok", _validated()).export_full_draft("markdown")
    # CI [0.61, 0.94] excludes zero -> the row must say so.
    assert "| yes |" in md, md


# ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ supplementary raw data is counted, not invented ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬


def test_supplementary_raw_data_is_not_a_fabricated_sample_count(tmp_path, monkeypatch):
    import json
    from backend.science.publications.supplementary import SupplementaryMaterialPackager

    monkeypatch.chdir(tmp_path)
    results = _validated(raw_traces=[{"metric": "logit_diff"}] * 3)
    path = SupplementaryMaterialPackager("exp_sup", results).create_bundle()
    payload = json.load(open(os.path.join(path, "raw_data",
                                          "activation_samples_summary.json")))
    assert payload["measured"] is True
    assert payload["samples"] == 3
    assert payload["metrics"] == ["logit_diff"]


def test_supplementary_raw_data_reports_absence_when_nothing_was_supplied(tmp_path, monkeypatch):
    import json
    from backend.science.publications.supplementary import SupplementaryMaterialPackager

    monkeypatch.chdir(tmp_path)
    path = SupplementaryMaterialPackager("exp_sup2", _validated()).create_bundle()
    payload = json.load(open(os.path.join(path, "raw_data",
                                          "activation_samples_summary.json")))
    assert payload["measured"] is False
    assert payload["provenance"] == "unavailable"
    assert payload["reason"]
    assert "samples" not in payload, "a sample count was invented"


# ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ source guards ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬


def test_no_canned_result_prose_survives():
    """The literals that asserted results regardless of the data."""
    banned = (
        "strong causal relationship",
        "rigorous, reproducible verification",
        "demonstrate the robustness of the MECH platform",
    )
    source = open(os.path.join(PUBLICATIONS_DIR, "paper_generator.py"),
                  encoding="utf-8").read()
    for literal in banned:
        assert literal not in source, literal


def test_generate_abstract_is_defined_on_the_class():
    """AST check: the method exists, rather than only being referenced."""
    tree = ast.parse(open(os.path.join(PUBLICATIONS_DIR, "paper_generator.py"),
                          encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "PaperGenerator":
            methods = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
            assert "generate_abstract" in methods, sorted(methods)
            return
    pytest.fail("PaperGenerator class not found")


# ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ section structure: emitted once, and nothing built is dropped ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬


def test_sections_are_not_duplicated(tmp_path, monkeypatch):
    """Leaf generators used to carry their own `## Heading`, so the template
    produced `## Limitations` twice (and `## References` twice)."""
    monkeypatch.chdir(tmp_path)
    md = PaperGenerator("exp_dup", _validated(model="gpt2")).export_full_draft()
    headings = [l for l in md.splitlines() if l.startswith("## ")]
    duplicated = [h for h in set(headings) if headings.count(h) > 1]
    assert not duplicated, f"duplicated sections: {duplicated}"


def test_a_built_section_is_not_silently_dropped(tmp_path, monkeypatch):
    """arXiv's template order omits "Future Work"; it is still emitted."""
    monkeypatch.chdir(tmp_path)
    md = PaperGenerator("exp_fw", _validated(model="gpt2")).export_full_draft()
    assert "## Future Work" in md
    assert "## Appendix" in md


def test_limitations_do_not_assert_an_absent_sample_size():
    """"The sample size (N=0) is sufficient for the detected effect" was a claim
    about a value the run did not have."""
    from backend.science.publications.paper_templates import PaperTemplates

    # `_validated()` carries n=144, so build a copy with the count removed.
    without_n = _validated()
    without_n.pop("n_samples", None)
    without_n["statistical_results"].pop("n", None)
    text = PaperTemplates.apply_limitations(without_n)
    assert "is sufficient for the detected effect" not in text
    assert "N=0" not in text
    assert "No sample size was reported" in text

    sized = PaperTemplates.apply_limitations(_validated(model="gpt2", n_samples=144))
    assert "The reported sample size (N=144) bounds the statistical power" in sized
    assert "is sufficient" not in sized


def test_limitations_name_the_model_only_when_recorded():
    from backend.science.publications.paper_templates import PaperTemplates

    recorded = PaperTemplates.apply_limitations(_validated(model="gpt2", n_samples=144))
    assert "focuses on gpt2" in recorded

    absent = PaperTemplates.apply_limitations(_validated())
    assert "No model identity was recorded" in absent
    assert "focuses on Unknown" not in absent


def test_limitations_mention_sae_only_when_an_sae_result_exists():
    from backend.science.publications.paper_templates import PaperTemplates

    assert "SAE findings" not in PaperTemplates.apply_limitations(_validated())
    assert "SAE findings" in PaperTemplates.apply_limitations(
        _validated(sae={"checkpoint": "x"}))
