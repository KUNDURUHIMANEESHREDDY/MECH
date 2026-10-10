"""Head importance must be a signed causal effect, not a magnitude.

The rule under test
-------------------
`scripts/capture_results.py` zero-ablates every attention head and publishes the
ranking in `docs/results/capture.json`, which the README's *Head ablation — which
heads the IOI behaviour depends on* table is built from.

It ranked by magnitude:

    ranked = sorted(heads, key=lambda h: abs(h["delta"] or 0), reverse=True)

The engine's `delta` is `patched_ld - clean_ld`. So a head whose removal made the
IOI logit difference *larger* scored the same as one whose removal destroyed it.
On the measured GPT-2 small sweep that is not a subtlety, it is the whole list:
of 144 heads, 68 support the behaviour and 76 oppose it, and the top three
entries of a table titled "which heads the IOI behaviour depends on" --

    L0H7  delta +1.5011   ablation IMPROVED IOI
    L0H0  delta +1.3284   ablation IMPROVED IOI
    L2H3  delta +1.2097   ablation IMPROVED IOI

-- were the three strongest *suppressors* of the behaviour, not members of its
circuit.

What is asserted here
---------------------
* A destructive ablation and an improving one never share a rank.
* Ranking is descending on the signed effect, so protective heads sort last.
* Supporting and opposing heads are counted and listed separately.
* Both absolute and normalised effect size are reported, and the normalised one
  is signed too.
* The engine's sign convention is checked, not assumed -- if `delta` ever flips,
  the run fails rather than silently re-ranking every head.
* Ablations are isolated: the same head measures the same either side of a sweep.
* The figure plots the signed effect, so a supporting head cannot be rendered
  identically to an opposing one.

Negative controls: restoring `abs(delta)` as the sort key, making the sign check
truthiness-based, and dropping the isolation guard were each caught.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "capture_results.py"

from scripts import capture_results as cap  # noqa: E402

ROLE_SUPPORTS = cap.ROLE_SUPPORTS
ROLE_OPPOSES = cap.ROLE_OPPOSES
ROLE_NEUTRAL = cap.ROLE_NEUTRAL


# ── Fakes ───────────────────────────────────────────────────────────────

GPT2_SMALL = {"n_layers": 12, "n_heads": 12, "d_model": 768, "d_head": 64,
              "d_mlp": 3072, "vocab_size": 50257, "n_positions": 1024}


def _patch_result(clean: float, patched: float, status: str = "ok",
                  error: str | None = None) -> Dict[str, Any]:
    return {
        "status": status, "clean_ld": clean, "patched_ld": patched,
        "delta": round(patched - clean, 4), "error": error,
    }


class SweepEngine:
    """An engine whose every head ablations have a scripted effect."""

    def __init__(self, deltas: Dict[tuple, float], clean: float = 2.1681,
                 dims=None, default: float | None = None):
        self.deltas = deltas
        self.clean = clean
        self.dims = dict(GPT2_SMALL if dims is None else dims)
        self.default = default
        self.calls: List[tuple] = []

    def info(self):
        return {"status": "loaded", **self.dims}

    def run_prompt(self, prompt):
        return {"top5": []}

    def patch_head(self, layer, head, pos, neg):
        self.calls.append((layer, head))
        if (layer, head) not in self.deltas and self.default is None:
            return {"status": "error", "error": "no scripted delta"}
        effect = self.deltas.get((layer, head), self.default)
        return _patch_result(self.clean, round(self.clean - effect, 4))


def _sweep(engine, monkeypatch):
    monkeypatch.setattr(cap, "engine", engine)
    return cap.capture_head_sweep()


# ── Sign, not magnitude ─────────────────────────────────────────────────

def test_a_destructive_and_an_improving_ablation_do_not_share_a_rank(monkeypatch):
    """The defect, at its source.

    Two heads, same clean baseline. One's removal halves the IOI difference; the
    other's removal doubles it. The magnitudes differ, so the old code ordered
    them; the question is whether either is "more important".
    """
    engine = SweepEngine({
        (0, 0): 0.05,    # barely matters
        (0, 1): 2.10,    # ablation destroys nearly the whole effect
        (0, 2): -3.00,   # ablation improves it, and by the most
    })
    monkeypatch.setattr(cap, "CAPTURES",
                        (("head_sweep", "sweep", "capture_head_sweep"),))
    monkeypatch.setattr(cap, "EXPECTED_SECTIONS", ("head_sweep",))
    # A three-head model, so all three are swept.
    engine.dims = {**GPT2_SMALL, "n_layers": 1, "n_heads": 3}

    payload = _sweep(engine, monkeypatch)
    order = [h["label"] for h in payload["all_heads"]]

    assert order[0] == "L0H1", (
        "the head whose ablation destroys the IOI signal must rank first, got "
        f"{order}")
    assert order[-1] == "L0H2", (
        "the head whose ablation most improves IOI must rank last, got "
        f"{order}")
    assert order == ["L0H1", "L0H0", "L0H2"]


def test_protective_heads_are_never_at_the_top(monkeypatch):
    """Even when a protective head has the largest magnitude anywhere."""
    engine = SweepEngine({
        (0, 0): -9.99,   # enormous magnitude, but ablation HELPS
        (0, 1): 0.50,    # small magnitude, ablation DESTROYS
    })
    engine.dims = {**GPT2_SMALL, "n_layers": 1, "n_heads": 2}

    payload = _sweep(engine, monkeypatch)

    assert payload["top_heads"][0]["label"] == "L0H1"
    assert payload["top_heads"][0]["role"] == ROLE_SUPPORTS
    assert payload["all_heads"][-1]["label"] == "L0H0"
    assert payload["all_heads"][-1]["role"] == ROLE_OPPOSES


def test_the_measured_gpt2_order_is_reproduced(monkeypatch):
    """The three heads that headed the README table were all suppressors.

    Values taken from the committed artifact, so this pins the concrete
    consequence rather than a synthetic case. `L2H0` was rank 4 of the old
    magnitude table with `delta -0.9391` -- a real circuit member, one place
    below three suppressors that outranked it purely on magnitude.
    """
    deltas = {
        (0, 7): -1.5011,   # old rank 1: ablation improved IOI
        (0, 0): -1.3284,   # old rank 2: ablation improved IOI
        (2, 3): -1.2097,   # old rank 3: ablation improved IOI
        (2, 0): 0.9391,    # old rank 4: a genuine circuit member
        (8, 10): 0.9010,   # old rank 5: also a circuit member
    }
    engine = SweepEngine(deltas, dims={**GPT2_SMALL, "n_layers": 9, "n_heads": 11},
                        default=0.01)

    payload = _sweep(engine, monkeypatch)

    # Under the old magnitude rule the order was L0H7, L0H0, L2H3, L2H0, L8H10.
    top = [h["label"] for h in payload["top_heads"]]
    assert top[:2] == ["L2H0", "L8H10"], (
        f"the two real circuit members must lead, got {top[:2]}")
    assert not {"L0H7", "L0H0", "L2H3"} & set(top), (
        "a head whose ablation improves IOI reached the top of the circuit list")
    assert [h["label"] for h in payload["opposing_heads"]] == [
        "L0H7", "L0H0", "L2H3"], (
        "the three former top entries are the three strongest suppressors")
    assert payload["roles"][ROLE_OPPOSES] == 3
    assert payload["roles"][ROLE_SUPPORTS] == payload["n_heads_swept"] - 3


def test_the_opposing_list_leads_with_the_strongest_opposition(monkeypatch):
    """It is read as "these work against the behaviour", so the strongest first.

    Slicing it out of the descending ranking would put the *least* opposing head
    at the top of a list whose only job is to say which opposition is strongest.
    """
    deltas = {(0, 0): -0.25, (0, 1): -2.0, (0, 2): -1.0, (0, 3): 1.0}
    engine = SweepEngine(deltas)
    engine.dims = {**GPT2_SMALL, "n_layers": 1, "n_heads": 4}

    payload = _sweep(engine, monkeypatch)

    assert [h["label"] for h in payload["opposing_heads"]] == [
        "L0H1", "L0H2", "L0H0"]
    assert [h["label"] for h in payload["supporting_heads"]] == ["L0H3"]


# ── Classification ──────────────────────────────────────────────────────

@pytest.mark.parametrize("effect,role", [
    (5.0, ROLE_SUPPORTS),
    (0.0001, ROLE_SUPPORTS),
    (-5.0, ROLE_OPPOSES),
    (-0.0001, ROLE_OPPOSES),
    (0.0, ROLE_NEUTRAL),
])
def test_role_is_decided_by_sign(effect, role):
    assert cap.classify_head_effect(effect) == role


def test_role_needs_no_arbitrary_threshold():
    """There is no magic epsilon to argue about.

    The engine rounds to four decimals, so a head below that resolution arrives
    as exactly 0.0. Strict sign keeps the classification a fact about the
    measurement instead of a boundary somebody picked.
    """
    assert cap.classify_head_effect(1e-12) == ROLE_SUPPORTS
    assert cap.classify_head_effect(-1e-12) == ROLE_OPPOSES


def test_supporting_and_opposing_heads_are_counted_separately(monkeypatch):
    engine = SweepEngine({
        (0, 0): 1.0, (0, 1): 0.5, (0, 2): -1.0, (0, 3): -0.25, (0, 4): 0.0,
    })
    engine.dims = {**GPT2_SMALL, "n_layers": 1, "n_heads": 5}

    payload = _sweep(engine, monkeypatch)

    assert payload["roles"] == {ROLE_SUPPORTS: 2, ROLE_OPPOSES: 2, ROLE_NEUTRAL: 1}
    assert [h["label"] for h in payload["supporting_heads"]] == ["L0H0", "L0H1"]
    assert [h["label"] for h in payload["opposing_heads"]] == ["L0H2", "L0H3"]


def test_the_two_lists_partition_the_sweep(monkeypatch):
    engine = SweepEngine({(l, h): (0.1 if (l + h) % 2 else -0.1)
                         for l in range(2) for h in range(2)})
    engine.dims = {**GPT2_SMALL, "n_layers": 2, "n_heads": 2}

    payload = _sweep(engine, monkeypatch)

    assert len(payload["supporting_heads"]) + len(payload["opposing_heads"]) \
        + payload["roles"][ROLE_NEUTRAL] == payload["n_heads_swept"]


def test_the_ranking_says_what_it_ranked_on(monkeypatch):
    """A reader who disagrees has to be able to see the choice."""
    engine = SweepEngine({(0, 0): 1.0})
    engine.dims = {**GPT2_SMALL, "n_layers": 1, "n_heads": 1}

    ranking = _sweep(engine, monkeypatch)["ranking"]

    assert ranking["key"] == "effect"
    assert "clean_ld - patched_ld" in ranking["definition"]
    assert "magnitude" in ranking["rationale"]


# ── Reported effect sizes ───────────────────────────────────────────────

def test_both_absolute_and_normalised_sizes_are_reported(monkeypatch):
    engine = SweepEngine({(0, 0): 1.0841, (0, 1): -2.1681})
    engine.dims = {**GPT2_SMALL, "n_layers": 1, "n_heads": 2}

    payload = _sweep(engine, monkeypatch)
    by_label = {h["label"]: h for h in payload["all_heads"]}

    assert by_label["L0H0"]["effect"] == pytest.approx(1.0841, abs=1e-4)
    assert by_label["L0H0"]["effect_abs"] == pytest.approx(1.0841, abs=1e-4)
    assert by_label["L0H0"]["effect_fraction"] == pytest.approx(0.5, abs=1e-3)
    # Signed: an opposing head keeps its sign after normalising.
    assert by_label["L0H1"]["effect_fraction"] == pytest.approx(-1.0, abs=1e-3)
    assert by_label["L0H1"]["effect_abs"] == pytest.approx(2.1681, abs=1e-4)


def test_the_normalised_effect_is_undefined_at_a_zero_baseline(monkeypatch):
    """A ratio against zero is undefined, not infinite, and must not be invented.

    The fake takes `patched = clean - effect`, so a clean baseline of 0 with an
    effect of +0.5 means the ablated difference is -0.5.
    """
    engine = SweepEngine({(0, 0): 0.5})
    engine.clean = 0.0
    engine.dims = {**GPT2_SMALL, "n_layers": 1, "n_heads": 1}

    head = _sweep(engine, monkeypatch)["all_heads"][0]

    assert head["clean_ld"] == 0.0
    assert head["patched_ld"] == pytest.approx(-0.5, abs=1e-4)
    assert head["effect"] == pytest.approx(0.5, abs=1e-4)
    assert head["effect_fraction"] is None


def test_effect_equals_the_negated_engine_delta(monkeypatch):
    """`effect` and `-delta` must agree; they are the same measurement."""
    entry = cap._head_entry(3, 7, _patch_result(2.0, 0.5))

    assert entry["delta"] == -1.5
    assert entry["effect"] == 1.5
    assert entry["effect"] == -entry["delta"]


def test_a_flipped_engine_sign_convention_fails_the_run():
    """The sign convention is the whole point, so it is checked not assumed.

    If the engine ever reported `delta = clean - patched`, every head would
    silently reverse role and the ranking would be confidently backwards.
    """
    flipped = _patch_result(2.0, 0.5)
    flipped["delta"] = 1.5  # the opposite convention

    with pytest.raises(ValueError) as excinfo:
        cap._head_entry(3, 7, flipped)

    message = str(excinfo.value)
    assert "does not match -delta" in message
    assert "convention has changed" in message


# ── Determinism ─────────────────────────────────────────────────────────

def test_ties_are_ordered_deterministically(monkeypatch):
    """Two identical effects must not swap places between runs.

    An unstable ordering cannot be diffed against the previous artifact, and a
    re-run appearing to change results is its own kind of dishonesty.
    """
    deltas = {(0, i): 1.0 for i in range(6)}
    deltas.update({(1, i): 1.0 for i in range(6)})
    engine = SweepEngine(deltas)
    engine.dims = {**GPT2_SMALL, "n_layers": 2, "n_heads": 6}

    first = [h["label"] for h in _sweep(engine, monkeypatch)["all_heads"]]
    second = [h["label"] for h in _sweep(engine, monkeypatch)["all_heads"]]

    assert first == second == sorted(first)


# ── Sweep completeness ──────────────────────────────────────────────────

def test_an_unswept_head_still_fails_the_run(monkeypatch):
    """Retained: a partial sweep cannot be ranked as a complete one."""
    engine = SweepEngine({(0, 0): 1.0})
    engine.dims = {**GPT2_SMALL, "n_layers": 1, "n_heads": 2}

    with pytest.raises(RuntimeError) as excinfo:
        _sweep(engine, monkeypatch)

    assert "1 of 2" in str(excinfo.value)
    assert "cannot be ranked as a complete one" in str(excinfo.value)


def test_every_head_is_actually_called(monkeypatch):
    engine = SweepEngine({(l, h): 0.1
                         for l in range(2) for h in range(3)})
    engine.dims = {**GPT2_SMALL, "n_layers": 2, "n_heads": 3}

    _sweep(engine, monkeypatch)

    assert engine.calls == [(l, h) for l in range(2) for h in range(3)]


# ── Isolation ───────────────────────────────────────────────────────────

def test_a_head_measures_the_same_either_side_of_a_sweep(monkeypatch):
    """No ablation may leak into the next one.

    Checked live against the real engine, sweeping a set of heads forwards and
    then backwards: if `patch_head` left a hook installed or mutated a weight,
    the second pass would differ. It does not -- the hook is removed in a
    `finally` and the activation is cloned before zeroing -- and this guards it.
    """
    from backend.services import gpt2_engine as engine_module

    if not engine_module.is_available():
        pytest.skip("torch/transformers not installed")
    load = engine_module.load()
    if load.get("status") != "loaded":
        pytest.skip(f"weights unavailable: {load.get('error')}")

    prompt, correct, incorrect = cap.IOI_TEMPLATES[0]
    engine_module.run_prompt(prompt)
    pos, neg = correct.strip(), incorrect.strip()

    heads = [(0, 7), (2, 0), (8, 10), (5, 9)]
    forward = {h: engine_module.patch_head(*h, pos, neg) for h in heads}
    reverse = {h: engine_module.patch_head(*h, pos, neg) for h in reversed(heads)}

    for head in heads:
        assert (forward[head]["clean_ld"], forward[head]["patched_ld"]) == (
            reverse[head]["clean_ld"], reverse[head]["patched_ld"]), (
            f"{head} measured differently after the sweep ran in reverse order")

    hooks = getattr(engine_module._model.transformer.h[0].attn.c_proj,
                    "_forward_pre_hooks", {})
    assert len(hooks) == 0, "a forward hook was left installed on the model"


# ── Structural ──────────────────────────────────────────────────────────

def test_the_sort_key_does_not_take_an_absolute_value():
    """Located rather than grepped, because `abs` appears in the figure's
    colour limits and in `effect_abs`, where it is legitimate."""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    node = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef)
                and n.name == "_by_effect_desc")

    abs_calls = [
        n for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name) and n.func.id == "abs"
    ]
    assert not abs_calls, (
        "_by_effect_desc takes an absolute value, so a head that opposes the "
        "behaviour ranks alongside one that supports it")


def test_the_ranking_is_ascending_in_signed_effect(monkeypatch):
    """Read off the output rather than trusting the sort implementation."""
    deltas = {(l, h): ((l * 7 + h) % 5 - 2) * 0.37
              for l in range(3) for h in range(3)}
    engine = SweepEngine(deltas)
    engine.dims = {**GPT2_SMALL, "n_layers": 3, "n_heads": 3}

    effects = [h["effect"] for h in _sweep(engine, monkeypatch)["all_heads"]]

    assert effects == sorted(effects, reverse=True), effects


def test_the_abs_based_ratio_field_is_gone():
    """`drop_ratio` was `abs(delta)/abs(clean_ld)` -- the conflation, named.

    Leaving it in place would preserve the trap under a field nobody re-reads.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)

    literals = {
        n.value for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    assert "drop_ratio" not in literals
    assert "drop_ratio" not in cap.REQUIRED_KEYS["head_sweep"]


def test_the_figure_plots_the_signed_effect(monkeypatch, tmp_path):
    """`abs(delta)` in a diverging map would still hide the sign from the eye
    in any other encoding -- but more concretely, the old figure put a
    suppressing head and a circuit member on the same colour."""
    monkeypatch.setattr(cap, "IMG_DIR", str(tmp_path))
    engine = SweepEngine({(0, 0): 1.0, (0, 1): -1.0})
    engine.dims = {**GPT2_SMALL, "n_layers": 1, "n_heads": 2}

    payload = _sweep(engine, monkeypatch)
    written = cap.render_figures({"head_sweep": payload})

    assert any("ioi-head-sweep" in path for path in written)

    source = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    render = next(n for n in source.body
                  if isinstance(n, ast.FunctionDef) and n.name == "render_figures")
    grid_assignments = [
        n for n in ast.walk(render)
        if isinstance(n, ast.Subscript)
        and isinstance(n.ctx, ast.Store)
        and isinstance(n.slice, ast.Subscript)
        and isinstance(n.slice.value, ast.Name)
        and n.slice.value.id == "grid"
    ]
    for assignment in grid_assignments:
        assert "abs" not in ast.dump(assignment.value), (
            "the head-sweep grid stores a magnitude; opposing heads must land on "
            "the other side of the colour map")


def test_the_figure_labels_both_directions(monkeypatch, tmp_path):
    """A diverging map with no sign convention is a puzzle, not a plot."""
    monkeypatch.setattr(cap, "IMG_DIR", str(tmp_path))
    engine = SweepEngine({(0, 0): 1.0})
    engine.dims = {**GPT2_SMALL, "n_layers": 1, "n_heads": 1}

    cap.render_figures({"head_sweep": _sweep(engine, monkeypatch)})

    source = SCRIPT.read_text(encoding="utf-8")
    assert "clean − ablated" in source
    assert "warm = removal destroys IOI" in source
