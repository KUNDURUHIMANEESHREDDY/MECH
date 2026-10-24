from types import SimpleNamespace

import pytest


def test_injected_logits_refuses_unaligned_token_positions(monkeypatch):
    """Mismatched clean/corrupted sequence shapes fail closed before hook installation."""
    torch = pytest.importorskip("torch")
    from backend.interpretability.discovery import live_measure

    projection = torch.nn.Identity()
    fake_model = SimpleNamespace(
        transformer=SimpleNamespace(
            h=[SimpleNamespace(attn=SimpleNamespace(c_proj=projection))]
        )
    )
    engine = SimpleNamespace(_model=fake_model)
    monkeypatch.setattr(live_measure, "dims", lambda: (1, 1, 2))
    monkeypatch.setattr(
        live_measure,
        "_inputs",
        lambda _engine, _prompt: {
            "input_ids": torch.zeros((1, 128), dtype=torch.long)
        },
    )

    clean_capture = torch.zeros((1, 14, 2))
    with pytest.raises(
        live_measure.LiveUnavailable,
        match="token positions cannot be aligned",
    ):
        live_measure._injected_logits(
            engine, "corrupted prompt", {0: clean_capture}, {(0, 0)}
        )

    assert not projection._forward_pre_hooks, (
        "an invalid intervention installed a model hook before refusing"
    )
