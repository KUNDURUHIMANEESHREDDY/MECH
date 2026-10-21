"""IOI discovery must isolate task-specific effects and sample across seeds."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.interpretability.discovery import live_discovery as ld
from backend.science.reproducibility.ioi_pipeline import IOIReproductionPipeline


def test_generic_name_token_effect_cancels_in_the_clean_corrupted_contrast():
    from backend.interpretability.discovery.live_discovery import _ioi_specific_effect

    # Both conditions lose the same 0.5 logit units when the head is ablated.
    assert _ioi_specific_effect(
        clean_before=2.0,
        clean_after=1.5,
        corrupted_before=0.0,
        corrupted_after=-0.5,
    ) == pytest.approx(0.0)


def test_clean_only_causal_effect_survives_the_corruption_contrast():
    from backend.interpretability.discovery.live_discovery import _ioi_specific_effect

    # The head changes the clean IOI answer but has no effect on the ABB control.
    assert _ioi_specific_effect(
        clean_before=2.0,
        clean_after=1.4,
        corrupted_before=0.0,
        corrupted_after=0.0,
    ) == pytest.approx(0.6)


def test_prompt_panel_seed_is_reproducible_but_changes_the_prompt_set():
    first = ld.prompt_panel(12, seed=42)
    repeat = ld.prompt_panel(12, seed=42)
    other = ld.prompt_panel(12, seed=43)

    assert first == repeat
    assert {entry["pool_index"] for entry in first} != {
        entry["pool_index"] for entry in other
    }
    assert all(len({entry["pool_index"] for entry in panel}) == 12
               for panel in (first, other))


def test_prompt_pair_keeps_the_abb_corruption_across_frames():
    from backend.interpretability.discovery.live_discovery import render_prompt_pair

    for frame_id in range(8):
        clean, corrupted = render_prompt_pair("Alice", "Bob", frame_id)
        assert clean != corrupted
        assert "Alice" in clean and "Bob" in clean
        assert "Alice" in corrupted and "Bob" in corrupted
        assert clean.endswith("Alice") is False
        assert corrupted.endswith("Alice") is False
        assert ", Alice " in clean
        assert ", Bob " in corrupted


def test_head_screen_ranks_ioi_specific_effect_not_generic_identity_effect(
        monkeypatch):
    def render(subject: str, io: str, frame_id: int):
        return f"CLEAN::{subject}::{io}", f"CORRUPTED::{subject}::{io}"

    monkeypatch.setattr(ld, "render_prompt_pair", render)
    monkeypatch.setattr(
        ld.lm, "single_token_names", lambda names: {name: i for i, name in enumerate(names)})
    monkeypatch.setattr(ld.lm, "dims", lambda: (1, 2, 1))

    def baseline(prompt: str, io_id: int, subj_id: int) -> Dict[str, Any]:
        clean = prompt.startswith("CLEAN::")
        return {"top1": "Bob", "logit_diff": 2.0 if clean else 0.0}

    def ablate(prompt: str, io_id: int, subj_id: int, heads: set) -> float:
        clean = prompt.startswith("CLEAN::")
        baseline_value = 2.0 if clean else 0.0
        drop = 0.6 if (0, 0) in heads and clean else 0.0
        if (0, 1) in heads:
            drop += 0.5
        return baseline_value - drop

    monkeypatch.setattr(ld.lm, "baseline", baseline)
    monkeypatch.setattr(ld.lm, "ablate", ablate)
    monkeypatch.setattr(ld.lm, "capture", lambda prompt: (None, {}))
    monkeypatch.setattr(
        ld.lm, "inject", lambda *args, **kwargs: {"logit_diff": 1.0})
    monkeypatch.setattr(
        ld, "attest_measurement",
        lambda result, **kwargs: {**result, "provenance": "live", "attested": True},
    )

    result = ld.LiveIOIDiscovery()._run(
        "find IOI heads", n_prompts=2, max_heads=2,
        n_interaction_prompts=1, seed=42,
    )

    assert result["heads"] == ["L0H0"]
    selected = result["head_effects"][0]
    assert selected["effect_kind"] == "ioi_specific_ablation_difference"
    assert selected["mean_ioi_specific_delta"] == pytest.approx(0.6)
    assert "corrupted_prompt" in result["methodology"]


def test_pipeline_runs_discovery_on_the_seeded_prompt_sample(monkeypatch):
    seen = {}
    from backend.interpretability.discovery.live_discovery import LiveIOIDiscovery

    def fake_run(self, hypothesis_statement="", **kwargs):
        seen.update(kwargs)
        return {"status": "completed", "provenance": "live", "heads": ["L0H0"]}

    monkeypatch.setattr(LiveIOIDiscovery, "run", fake_run)

    result = IOIReproductionPipeline(mock_mode=True)._discover_live_circuit(
        n_prompts=40, seed=73)

    assert result["heads"] == ["L0H0"]
    assert seen["seed"] == 73
    assert seen["n_prompts"] >= 10
    assert seen["n_interaction_prompts"] >= 5


def test_seed_audit_reports_head_overlap_in_addition_to_metric_pooling(monkeypatch):
    results = {
        42: ["L0H0", "L1H1"],
        43: ["L0H0", "L2H2"],
        44: ["L0H0", "L1H1"],
    }

    def fake_run(self, n_prompts=100, seed=42, model_variant="small"):
        return {
            "status": "completed",
            "provenance": "live",
            "observed_metrics": {
                "circuit_faithfulness": 0.60 + 0.01 * (seed - 42),
                "discovered_nodes": results[seed],
                "cross_template_consistency_measured": True,
                "cross_template_consistent": True,
                "per_template": [{"frame_id": i, "usable": 3}
                                  for i in range(2)],
            },
        }

    monkeypatch.setattr(IOIReproductionPipeline, "run", fake_run)
    audit = IOIReproductionPipeline(mock_mode=True).run_stability_audit(
        n_seeds=3, n_prompts=20)

    stability = audit["head_selection_stability"]
    assert stability["measured"] is True
    assert stability["n_seed_sets"] == 3
    assert stability["n_pairs"] == 3
    assert stability["mean_pairwise_jaccard"] == pytest.approx(5 / 9)
    assert [entry["seed"] for entry in audit["per_seed"]] == [42, 43, 44]
    assert all(entry["cross_template_consistency_measured"]
               for entry in audit["per_seed"])
