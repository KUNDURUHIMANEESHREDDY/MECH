"""P0 #3 -- the paper generator must not crash and must not fabricate.

Claim under test: ``backend/science/publications/paper_generator.py`` no longer
crashes and does not fabricate.

``test_paper_generator_no_fabrication.py`` already pins the original defect
(``export_full_draft`` calling a ``generate_abstract`` that did not exist), the
``effect_size -> 0.0`` substitution, the literal ``ci_match = True``, and the
banned prose.  This module attacks the *crash surface* it does not cover:
adversarial-but-validator-passing result payloads.  For each input the contract
is one of exactly two honest outcomes --

* a draft is produced and rendered, or
* a ``ValueError`` names the missing evidence.

An ``AttributeError`` / ``TypeError`` / ``KeyError`` / ``IndexError`` escaping
is a crash, and this module fails on it.  A missing value appearing as a number
is the other failure, so the number checks from the sibling module are re-run on
the adversarial inputs rather than assumed.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.science.publications.paper_generator import (  # noqa: E402
    PaperGenerator,
    _number,
    _text,
)

#: An exception type that means "the generator has a bug", as opposed to the
#: two honest outcomes.
CRASH_TYPES = (AttributeError, TypeError, KeyError, IndexError, ZeroDivisionError)


def _validated(**overrides):
    """A result that clears ``PaperValidator``, with overrides applied."""
    results = {
        "paper": {"title": "T", "doi": "10.48550/arXiv.2211.00593"},
        "dataset_certificate": {"sha256": "a" * 64},
        "model_fingerprint": {
            "attested": True, "model_id": "gpt2", "parameter_count": 124439808,
            "weights_sha256": "b" * 64,
        },
        "statistical_results": {
            "effect_sizes": {"cohens_d": 0.8},
            "bootstrap_ci": [0.6, 0.9],
            "n": 144,
            "verdict": "PASS",
            "protocol": "Head-level zero-ablation.",
        },
        "circuit": {"heads": ["L0H10"], "edges": [1], "faithfulness": 0.8,
                    "n_prompts": 100},
        "expected_effect_size": 0.86,
        "reproduction_successful": True,
    }
    results.update(overrides)
    return results


#: Each entry is a result payload the *validator accepts* (it only requires a
#: truthy certificate, an attested fingerprint, and truthy statistical results).
#: The generator must still produce a draft or raise ValueError -- never crash.
ADVERSARIAL_RESULTS = [
    ("effect-size-nan", {"statistical_results": {"effect_sizes": {"cohens_d": float("nan")}, "verdict": "PASS"}}),
    ("effect-size-inf", {"statistical_results": {"effect_sizes": {"cohens_d": float("inf")}, "verdict": "PASS"}}),
    ("effect-size-neginf", {"statistical_results": {"effect_sizes": {"cohens_d": float("-inf")}, "verdict": "PASS"}}),
    ("effect-size-string-number", {"statistical_results": {"cohens_d": "0.42", "verdict": "PASS"}}),
    ("effect-size-bool", {"statistical_results": {"cohens_d": True, "verdict": "PASS"}}),
    ("effect-size-none", {"statistical_results": {"effect_sizes": {"cohens_d": None}, "verdict": "PASS"}}),
    ("ci-one-element", {"statistical_results": {"cohens_d": 0.4, "bootstrap_ci": [0.1], "verdict": "PASS"}}),
    ("ci-three-elements", {"statistical_results": {"cohens_d": 0.4, "bootstrap_ci": [0.1, 0.2, 0.3], "verdict": "PASS"}}),
    ("ci-of-strings", {"statistical_results": {"cohens_d": 0.4, "bootstrap_ci": ["0.1", "0.2"], "verdict": "PASS"}}),
    ("ci-of-none", {"statistical_results": {"cohens_d": 0.4, "bootstrap_ci": [None, None], "verdict": "PASS"}}),
    ("ci-inverted", {"statistical_results": {"cohens_d": 0.4, "bootstrap_ci": [0.9, 0.1], "verdict": "PASS"}}),
    ("ci-nan", {"statistical_results": {"cohens_d": 0.4, "bootstrap_ci": [float("nan"), 0.2], "verdict": "PASS"}}),
    ("n-zero", {"statistical_results": {"cohens_d": 0.4, "n": 0, "verdict": "PASS"}}),
    ("n-negative", {"statistical_results": {"cohens_d": 0.4, "n": -5, "verdict": "PASS"}}),
    ("n-float", {"statistical_results": {"cohens_d": 0.4, "n": 12.7, "verdict": "PASS"}}),
    ("n-string", {"statistical_results": {"cohens_d": 0.4, "n": "many", "verdict": "PASS"}}),
    ("statistical-results-empty-dict", {"statistical_results": {"verdict": "PASS"}}),
    ("statistical-results-string", {"statistical_results": "PASS"}),
    ("statistical-results-list", {"statistical_results": [1, 2, 3]}),
    ("circuit-empty", {"circuit": {}}),
    ("circuit-heads-string", {"circuit": {"heads": "L0H10"}}),
    ("circuit-heads-none", {"circuit": {"heads": None}}),
    ("circuit-edges-dict", {"circuit": {"edges": {"a": 1}}}),
    ("circuit-faithfulness-nan", {"circuit": {"heads": [], "faithfulness": float("nan")}}),
    ("circuit-n-prompts-none", {"circuit": {"heads": [], "faithfulness": 0.5, "n_prompts": None}}),
    ("reproduction-non-bool", {"reproduction_successful": "yes"}),
    ("paper-not-a-dict", {"paper": "not a dict", "dataset_certificate": {"sha256": "a" * 64}}),
    ("paper-doi-empty", {"paper": {"title": "T", "doi": ""}}),
    ("expected-effect-string", {"expected_effect_size": "0.86"}),
    ("model-fingerprint-no-weights-hash", {"model_fingerprint": {"attested": True, "model_id": "gpt2"}}),
    ("model-fingerprint-huge-params", {"model_fingerprint": {"attested": True, "model_id": "gpt2", "parameter_count": 10 ** 15}}),
    ("model-fingerprint-params-string", {"model_fingerprint": {"attested": True, "model_id": "gpt2", "parameter_count": "lots"}}),
    ("unicode-title", {"paper": {"title": "\u2014 \u6d4b\u8bd5 \u0000", "doi": "10.1/x"}}),
    ("n_samples-string", {"n_samples": "144"}),
    ("empty-dict", {}),
]


@pytest.mark.parametrize("label,overrides", ADVERSARIAL_RESULTS,
                         ids=[c[0] for c in ADVERSARIAL_RESULTS])
def test_export_never_crashes_on_adversarial_but_valid_inputs(
        label, overrides, tmp_path, monkeypatch):
    """Produce a draft, or refuse with ValueError. Never an unhandled crash."""
    monkeypatch.setenv("MECH_PUBLICATION_DIR", str(tmp_path / "pub"))
    monkeypatch.chdir(tmp_path)

    gen = PaperGenerator("exp_adv", _validated(**overrides))
    try:
        draft = gen.export_full_draft("markdown")
    except ValueError:
        return  # the honest refusal
    except CRASH_TYPES as exc:
        pytest.fail(
            f"paper_generator crashed on {label!r}: {type(exc).__name__}: {exc}"
        )
    assert isinstance(draft, str) and draft.strip(), label


@pytest.mark.parametrize("label,overrides", ADVERSARIAL_RESULTS,
                         ids=[c[0] for c in ADVERSARIAL_RESULTS])
def test_adversarial_inputs_never_mint_a_number(label, overrides):
    """A missing/NaN/inf value must not appear as a rendered measurement."""
    gen = PaperGenerator("exp_adv", _validated(**overrides))
    text = "\n".join([
        gen.generate_abstract(),
        gen._discussion(),
        gen._conclusion(),
        gen._summary_table(),
    ])
    for bad in ("nan", "inf", "-inf"):
        assert bad not in text.lower().replace("information", ""), (
            f"{label!r} rendered {bad!r}: {text[:300]}"
        )
    assert "d = nan" not in text.lower()
    # A missing effect size must be named, not silently omitted.
    eff = gen._statistics()["effect_size"]
    if eff is None:
        assert "no effect size was reported" in text.lower(), (label, text[:400])


def test_zero_is_preserved_but_a_missing_value_is_not_zero():
    """The distinction the module docstring exists for."""
    assert _number(0) == 0.0
    assert _number(0.0) == 0.0
    for missing in (None, "", "  ", True, False, "abc", float("nan"),
                    float("inf"), float("-inf"), [], {}, object()):
        assert _number(missing) is None, missing
    assert _text(None) is None
    assert _text("") is None
    assert _text("   ") is None
    assert _text("x") == "x"


def test_a_real_zero_effect_is_reported_as_measured_zero():
    """Zero is a measurement, so it must survive into the abstract."""
    gen = PaperGenerator("exp_zero", _validated(
        statistical_results={"effect_sizes": {"cohens_d": 0.0},
                             "bootstrap_ci": [-0.1, 0.1], "n": 10,
                             "verdict": "PASS"}))
    text = gen.generate_abstract()
    assert "d = 0.000" in text, text
    assert "includes zero" in text, text


def test_nan_effect_is_treated_as_absent_not_as_zero():
    """NaN is not a number; it must read as 'not reported'."""
    gen = PaperGenerator("exp_nan", _validated(
        statistical_results={"effect_sizes": {"cohens_d": float("nan")},
                             "verdict": "PASS"}))
    assert gen._statistics()["effect_size"] is None
    text = gen.generate_abstract()
    assert "No effect size was reported" in text, text
    assert "d = 0.000" not in text, text


def test_latex_export_survives_adversarial_inputs(tmp_path, monkeypatch):
    """The LaTeX path has its own escaping; it must not crash either."""
    monkeypatch.setenv("MECH_PUBLICATION_DIR", str(tmp_path / "pub"))
    monkeypatch.chdir(tmp_path)
    gen = PaperGenerator("exp_tex", _validated(
        paper={"title": "100% & _under_score_ #hash $var", "doi": "10.1/x"},
        statistical_results={"cohens_d": 0.4, "verdict": "PASS"}))
    try:
        tex = gen.export_full_draft("latex")
    except ValueError:
        return
    except CRASH_TYPES as exc:  # pragma: no cover - the defect path
        pytest.fail(f"latex export crashed: {type(exc).__name__}: {exc}")
    assert "\\documentclass" in tex
    assert tex.count("\\begin{itemize}") == tex.count("\\end{itemize}"), (
        "unbalanced itemize after adversarial LaTeX escaping"
    )


def test_validator_refusal_is_a_message_not_a_crash():
    """Fail-closed must be a ValueError naming the problem."""
    for overrides in ({"model_fingerprint": {"attested": False}},
                      {"model_fingerprint": {}},
                      {"dataset_certificate": None},
                      {"dataset_certificate": {}},
                      {"statistical_results": {}}):
        gen = PaperGenerator("exp_bad", _validated(**overrides))
        with pytest.raises(ValueError) as excinfo:
            gen.export_full_draft()
        assert str(excinfo.value).strip(), overrides
        assert "Paper validation failed" in str(excinfo.value), overrides


def test_validation_failure_is_raised_before_any_artifact_is_written(
        tmp_path, monkeypatch):
    """A refused export must not leave a half-written package behind."""
    monkeypatch.setenv("MECH_PUBLICATION_DIR", str(tmp_path / "pub"))
    monkeypatch.chdir(tmp_path)
    gen = PaperGenerator("exp_refuse", _validated(
        model_fingerprint={"attested": False}))
    with pytest.raises(ValueError):
        gen.export_full_draft()
    pub = tmp_path / "pub"
    written = list(pub.rglob("*")) if pub.exists() else []
    assert not [p for p in written if p.is_file()], (
        f"a refused export wrote artifacts: {written}"
    )


def test_result_payload_is_coerced_to_a_dict_not_crashed_on():
    """A non-dict `results` is normalised in the constructor."""
    for value in (None, [], "text", 42):
        gen = PaperGenerator("exp_coerce", value)  # type: ignore[arg-type]
        assert gen.results == {}
        assert gen._statistics()["effect_size"] is None


def test_abstract_is_never_empty():
    """An empty abstract is a silent failure: nothing to read, no error."""
    for label, overrides in ADVERSARIAL_RESULTS:
        text = PaperGenerator("exp_empty", _validated(**overrides)).generate_abstract()
        assert isinstance(text, str) and text.strip(), label
        assert not math.isnan(float(len(text)))


def test_no_default_zero_returns_remain_in_the_derived_facts():
    """Source guard: `get(..., 0)` in the statistics reader is the old bug."""
    import ast
    text = (REPO_ROOT / "backend" / "science" / "publications" /
            "paper_generator.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef, ast.ClassDef, ast.Module)):
            doc = ast.get_docstring(node, clean=False)
            if doc and doc in text:
                text = text.replace(doc, "")
    # The specific old expression, which could not tell 0 from unreported.
    assert 'get("cohens_d", 0)' not in text
    assert "get('cohens_d', 0)" not in text
    assert 'get("effect_sizes", {}).get("cohens_d", 0)' not in text
