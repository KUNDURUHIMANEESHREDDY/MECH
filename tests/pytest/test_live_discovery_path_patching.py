"""The discovery run must carry a measured path-patching number.

Roadmap 2.1 left two items open that both needed a measurement rather than
a number copied from somewhere else:

* "Add a path-patching pass. `live_discovery.py` already has injection
  recovery -- build on it." The result now carries a measured path-patching
  number for the discovered circuit next to the injection-recovery
  faithfulness, and the published reference circuit is re-measured through
  the *same* harness on the *same* prompts, so the two circuits are compared
  like for like.

* "Add corrupted-prompt patching so the *IOI-specific* signal separates from
  generic token-identity effects." The screen already ranked heads by the
  difference-in-differences; the result now publishes *both* rankings -- the
  raw clean-prompt effect and the IOI-specific one -- so the effect of the
  corrupted control on the selection is visible rather than assumed.

What these tests pin is the wiring: that a measurement which was taken is
reported as measured, and that a measurement which could not be taken stays
unmeasured. Nothing here asserts a number the run did not produce, and the
negative tests are negative on purpose: they pass when the harness refuses
or reports "not measured", not when it manufactures a value.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _weight_guard import skip_reason, weights_available  # noqa: E402

from backend.interpretability.discovery import live_discovery as ld  # noqa: E402
from backend.interpretability.discovery import path_comparison as pc  # noqa: E402

needs_weights = pytest.mark.skipif(
    not weights_available(), reason=skip_reason())


# ── a mocked model, with one generic head and two IOI-specific heads ──────

#: Effect of zero-ablating each head on the clean prompt, and on the ABB
#: corrupted prompt. L0H1 and L1H0 hurt both conditions by the same amount,
#: so their clean-prompt effect is a generic name-token effect: the
#: difference-in-differences cancels it and the raw metric does not.
_CLEAN_DROP = {(0, 0): 0.6, (0, 1): 0.5, (1, 0): 0.3, (1, 1): 0.4}
_CORRUPTED_DROP = {(0, 0): 0.0, (0, 1): 0.5, (1, 0): 0.3, (1, 1): 0.0}

#: The pair (0, 0) + (1, 1) is super-additive on the clean prompt, so the
#: pairwise interaction stage finds an edge between them.
_PAIR_BONUS = 0.4


def _mocked_live(monkeypatch, path_effect: float = 0.42,
                 path_error: Exception | None = None) -> None:
    """Replace the live measurement primitives with a deterministic model."""

    def render(subject: str, io: str, frame_id: int):
        return f"CLEAN::{subject}::{io}", f"CORRUPTED::{subject}::{io}"

    monkeypatch.setattr(ld, "render_prompt_pair", render)
    monkeypatch.setattr(ld.lm, "single_token_names",
                        lambda names: {name: i for i, name in enumerate(names)})
    monkeypatch.setattr(ld.lm, "dims", lambda: (2, 2, 1))

    def baseline(prompt: str, io_id: int, subj_id: int) -> Dict[str, Any]:
        clean = prompt.startswith("CLEAN::")
        return {"top1": "Bob", "logit_diff": 2.0 if clean else 0.0}

    def ablate(prompt: str, io_id: int, subj_id: int, heads: set) -> float:
        clean = prompt.startswith("CLEAN::")
        drops = _CLEAN_DROP if clean else _CORRUPTED_DROP
        value = 2.0 if clean else 0.0
        for head in heads:
            value -= drops.get(head, 0.0)
        if clean and {(0, 0), (1, 1)}.issubset(heads):
            value -= _PAIR_BONUS
        return value

    def path_patch(clean: str, corrupted: str, io_id: int, subj_id: int,
                  sender, receivers):
        if path_error is not None:
            raise path_error
        return {
            "method": "path_patching",
            "sender": ld.lm.head_label(*sender),
            "receivers": [
                {"receiver": ld.lm.head_label(*receiver),
                 "patched_logit_diff": 0.84,
                 "path_effect": path_effect,
                 "top1": "Bob"}
                for receiver in receivers
            ],
            "clean_logit_diff": 2.0,
            "corrupted_logit_diff": 0.0,
            "denominator": 2.0,
        }

    monkeypatch.setattr(ld.lm, "baseline", baseline)
    monkeypatch.setattr(ld.lm, "ablate", ablate)
    monkeypatch.setattr(ld.lm, "capture", lambda prompt: (None, {}))
    monkeypatch.setattr(ld.lm, "inject",
                        lambda *args, **kwargs: {"logit_diff": 1.0})
    monkeypatch.setattr(ld.lm, "path_patch", path_patch)
    monkeypatch.setattr(
        ld, "attest_measurement",
        lambda result, **kwargs: {**result, "provenance": "live",
                                  "attested": True})


def _run_discovery(**kwargs) -> Dict[str, Any]:
    params = {"n_prompts": 2, "max_heads": 2, "n_interaction_prompts": 1,
              "seed": 42}
    params.update(kwargs)
    return ld.LiveIOIDiscovery()._run("find the IOI heads", **params)


# ── the screen's two rankings ──────────────────────────────────────────────

def test_screen_publishes_both_rankings_from_the_same_two_ablations(monkeypatch):
    """The raw metric and the IOI-specific one, side by side.

    This is the item the roadmap left open: "Add corrupted-prompt patching
    so the IOI-specific signal separates from generic token-identity
    effects". The screen already separated it; what was missing was the
    evidence that it did, so both rankings are now in the record.
    """
    _mocked_live(monkeypatch)
    result = _run_discovery()

    screen = result["screen"]
    assert screen["ranking_metric"] == "ioi_specific_ablation_difference"
    assert screen["n_heads_evaluated"] == 4
    assert len(screen["ioi_specific_top"]) == 4
    assert len(screen["raw_clean_effect_top"]) == 4
    assert screen["ranking_changed"] is True
    assert screen["n_heads_ranked_differently"] > 0
    assert screen["top_k_overlap"] == 4

    # The generic head (equal effect in both conditions) is ranked higher by
    # the raw metric than by the difference-in-differences.
    raw = screen["raw_clean_effect_top"]
    specific = screen["ioi_specific_top"]
    assert raw.index("L0H1") < specific.index("L0H1")

    # And it does not reach the circuit, because the floor applies to the
    # IOI-specific effect.
    assert "L0H1" not in result["heads"]


def test_screen_reports_the_measured_effect_of_the_heads_it_ranked(monkeypatch):
    _mocked_live(monkeypatch)
    result = _run_discovery()

    effects = result["screen"]["ioi_specific_top_effects"]
    raw_effects = result["screen"]["raw_clean_effect_top_effects"]
    assert effects["L0H0"] == pytest.approx(0.6)
    assert effects["L0H1"] == pytest.approx(0.0)
    assert raw_effects["L0H1"] == pytest.approx(0.5)


# ── the path-patching pass ────────────────────────────────────────────────

def test_discovery_carries_a_measured_path_patching_number(monkeypatch):
    """A measured path-patching number, next to the injection recovery."""
    _mocked_live(monkeypatch, path_effect=0.42)
    result = _run_discovery()

    path_patching = result["path_patching"]
    assert path_patching["measured"] is True
    assert path_patching["n_measurements"] > 0
    assert path_patching["n_edges"] >= 1
    assert path_patching["mean_abs_path_effect"] == pytest.approx(0.42)
    assert path_patching["max_abs_path_effect"] == pytest.approx(0.42)
    assert "two-hook path patching" in path_patching["method"]
    assert path_patching["significant_paths"], (
        "a measured path above the floor must be reported")
    edge = path_patching["edges"][0]
    assert edge["source"] == "L0H0"
    assert edge["target"] == "L1H1"
    assert edge["mean_abs_path_effect"] == pytest.approx(0.42)


def test_the_path_patching_pass_is_bounded_and_says_how_far(monkeypatch):
    _mocked_live(monkeypatch)
    result = _run_discovery()

    path_patching = result["path_patching"]
    assert path_patching["n_prompts"] <= pc.PATH_PATCHING_MAX_PROMPTS
    assert path_patching["n_prompts"] == result["n_prompts"]
    assert path_patching["effect_floor"] == pc.PATH_PATCHING_EFFECT_FLOOR


def test_the_reference_circuit_is_re_measured_through_the_same_harness(monkeypatch):
    """The published circuit, measured here -- not cited from elsewhere."""
    _mocked_live(monkeypatch)
    result = _run_discovery()

    reference = result["reference_circuit"]
    assert reference["measured"] is True
    assert reference["faithfulness_same_harness"] is not None
    assert reference["n_prompts"] == result["n_prompts"]
    assert reference["heads"], "the reference head list must be reported"
    # It went through the same procedure as the discovered circuit.
    assert result["faithfulness"] == pytest.approx(
        reference["faithfulness_same_harness"])
    assert result["faithfulness_vs_reference_circuit"] == pytest.approx(1.0)
    assert result["path_patching_comparison"]["like_for_like"] is True


def test_the_published_baseline_stays_marked_not_comparable(monkeypatch):
    """The registry's 0.86 is a different measurement; say so."""
    _mocked_live(monkeypatch)
    result = _run_discovery()

    assert result["published_baseline_is_comparable"] is False
    assert "path-patching procedure" in result["published_baseline_note"]


# ── negative cases: a refused or unmeasured result stays unmeasured ───────

def test_an_unmeasurable_path_stays_unmeasured(monkeypatch):
    """A path that could not be measured is reported as not measured.

    Not as zero: "the path was measured and it is small" and "the path was
    not measured" are different findings, and a zero would read as the
    first.
    """
    _mocked_live(monkeypatch,
                 path_error=RuntimeError("path patching: sender not captured"))
    result = _run_discovery()

    path_patching = result["path_patching"]
    assert path_patching["measured"] is False
    assert path_patching["mean_abs_path_effect"] is None
    assert path_patching["max_abs_path_effect"] is None
    assert path_patching["n_measurements"] == 0
    assert path_patching["reason"]
    assert path_patching["edges"][0]["n_measured"] == 0
    assert result["path_patching_comparison"]["like_for_like"] is False


def test_no_edges_means_no_path_measurement_not_a_zero(monkeypatch):
    _mocked_live(monkeypatch)
    empty = pc.measure_paths([], [{"clean": "CLEAN::a::b",
                                   "corrupted": "CORRUPTED::a::b"}])
    assert empty["measured"] is False
    assert empty["mean_abs_path_effect"] is None
    assert empty["max_abs_path_effect"] is None
    assert "no edges" in empty["reason"]


def test_an_unreadable_reference_circuit_stays_unmeasured(monkeypatch):
    """The reference list is an input; if it cannot be read, nothing is made up."""
    _mocked_live(monkeypatch)
    monkeypatch.setitem(sys.modules,
                         "backend.science.reproducibility.ioi_pipeline", None)

    reference = pc.reference_circuit([
        {"clean": "CLEAN::a::b", "corrupted": "CORRUPTED::a::b",
         "io_id": 0, "subj_id": 1, "clean_diff": 2.0, "corrupted_diff": 0.0},
    ])
    assert reference["measured"] is False
    assert reference["faithfulness_same_harness"] is None
    assert reference["reason"]


def test_a_ratio_against_an_unmeasured_reference_is_not_reported(monkeypatch):
    _mocked_live(monkeypatch)
    monkeypatch.setitem(sys.modules,
                         "backend.science.reproducibility.ioi_pipeline", None)
    result = _run_discovery()

    assert result["reference_circuit"]["measured"] is False
    assert result["faithfulness_vs_reference_circuit"] is None
    assert result["path_patching_comparison"]["like_for_like"] is False


# ── the real thing ────────────────────────────────────────────────────────

@needs_weights
def test_live_discovery_measures_path_patching_on_real_weights():
    """The whole thing on the real model, not on the mock above.

    The numbers are printed so the run can be read as a measurement report
    rather than a pass/fail flag.
    """
    result = ld.LiveIOIDiscovery().run(
        "find the IOI circuit", n_prompts=10, max_heads=10,
        n_interaction_prompts=5, seed=42)

    assert result["status"] == "completed"
    screen = result["screen"]
    assert screen["n_heads_evaluated"] == 144
    assert screen["ranking_changed"] is True

    path_patching = result["path_patching"]
    assert path_patching["measured"] is True, path_patching.get("reason")
    assert isinstance(path_patching["mean_abs_path_effect"], float)

    reference = result["reference_circuit"]
    assert reference["measured"] is True, reference.get("reason")
    assert isinstance(reference["faithfulness_same_harness"], float)
    assert isinstance(result["faithfulness"], float)
    assert result["faithfulness_vs_reference_circuit"] is not None
    assert result["published_baseline_is_comparable"] is False

    print("\n[measured discovery]")
    print(json.dumps({
        "heads": result["heads"],
        "faithfulness": result["faithfulness"],
        "n_prompts": result["n_prompts"],
        "screen_ioi_specific_top": screen["ioi_specific_top"],
        "screen_raw_clean_effect_top": screen["raw_clean_effect_top"],
        "screen_top_k_overlap": screen["top_k_overlap"],
        "screen_n_ranked_differently": screen["n_heads_ranked_differently"],
        "screen_only_in_raw": screen["top_k_only_in_raw_clean_effect"],
        "screen_ioi_specific_top_effects": screen["ioi_specific_top_effects"],
        "screen_raw_top_effects": screen["raw_clean_effect_top_effects"],
        "path_patching": {k: v for k, v in path_patching.items()
                          if k != "edges"},
        "path_patching_edges": path_patching["edges"],
        "reference_circuit": {k: v for k, v in reference.items()
                              if k != "path_patching"},
        "reference_path_patching": {k: v for k, v in
                                    (reference.get("path_patching") or {}).items()
                                    if k != "edges"},
        "faithfulness_vs_reference_circuit":
            result["faithfulness_vs_reference_circuit"],
        "path_patching_comparison": result["path_patching_comparison"],
    }, indent=2, sort_keys=True))
