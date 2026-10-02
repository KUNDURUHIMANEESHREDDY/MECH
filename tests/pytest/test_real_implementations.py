"""The real implementations: SAE encoding, ACDC fidelity, true path patching.

These replace three stubs that previously returned convincing numbers without
measuring anything. The SAE loader used to hand back a config-only SAE reported
as "loaded"; ACDC's fidelity was a formula floored at 0.90; path patching
patched the receiver alone and reported the sender's total effect as a path.

Tests that need real weights skip when GPT-2 is not available -- but the
fail-closed assertions run unconditionally, because those are the property
that matters most.
"""
import os
import tempfile

import pytest


# ── SAE: fails closed without weights ────────────────────────────────────────

def test_sae_without_weights_refuses_to_encode():
    from backend.interpretability.sae.loader import SAE, SAEConfig

    sae = SAE(SAEConfig(d_in=8, d_sae=16, model_name="m", layer=0,
                        repo_id="r", version="v"))
    assert sae.loaded is False
    out = sae.activate([0.1] * 8)
    assert out["status"] == "unavailable"
    assert out["measured"] is False
    assert out["publication_eligible"] is False
    # Crucially: no fabricated feature indices.
    assert "feature_indices" not in out
    assert out["reason"]


def test_sae_integrity_check_is_not_string_equality():
    """Previously config.checkpoint_sha was built from the checkpoint's name."""
    from backend.interpretability.sae.loader import SAE, SAEConfig

    sae = SAE(SAEConfig(d_in=8, d_sae=16, model_name="m", layer=0,
                        repo_id="r", version="v"))
    # A weightless SAE must never "verify" against any string.
    assert sae.verify_integrity("unknown") is False
    assert sae.verify_integrity("sha256:" + "0" * 64) is False
    assert sae.weights_sha256() == ""


def test_sae_loader_reports_why_it_could_not_load(tmp_path):
    from backend.interpretability.sae.loader import SAELoader

    missing = str(tmp_path / "nope.pt")
    res = SAELoader().load_checkpoint(missing)
    assert res["status"] == "unavailable"
    assert res["weights_loaded"] is False
    assert res["validation_eligible"] is False
    assert res["reason"]


def test_sae_provider_without_a_checkpoint_raises(tmp_path):
    from backend.interpretability.sae.loader import LocalCheckpointProvider, SAELoadError

    with pytest.raises(SAELoadError):
        LocalCheckpointProvider().load(str(tmp_path / "absent.pt"), "latest")


def test_sae_env_provider_requires_the_variable():
    from backend.interpretability.sae.loader import (
        LocalCheckpointProvider, SAELoadError,
    )

    name = "MECH_TEST_SAE_PATH_ABSENT"
    os.environ.pop(name, None)
    with pytest.raises(SAELoadError):
        LocalCheckpointProvider().load(f"env:{name}", "latest")


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_torch(), reason="torch not installed")
def test_sae_encodes_with_real_weights(tmp_path):
    """With real encoder weights, activate() performs the actual encoding."""
    import torch

    from backend.interpretability.sae.loader import (
        LocalCheckpointProvider, SAEConfig, SAE,
    )

    d_in, d_sae = 6, 10
    torch.manual_seed(0)
    w_enc = torch.randn(d_in, d_sae)
    path = tmp_path / "sae.pt"
    torch.save({"W_enc": w_enc, "b_pre": torch.zeros(d_in),
                "b_enc": torch.zeros(d_sae)}, path)

    sae = LocalCheckpointProvider().load(str(path), "test")
    assert sae.loaded is True
    assert sae.config.d_in == d_in
    assert sae.config.d_sae == d_sae

    out = sae.activate(torch.ones(d_in))
    assert out["status"] == "completed"
    assert out["measured"] is True
    assert len(out["feature_indices"]) == len(out["activations"]) == min(32, d_sae)
    # ReLU means non-negative firings, and the top-k are genuinely sorted.
    assert all(a >= 0 for a in out["activations"])
    assert out["activations"] == sorted(out["activations"], reverse=True)

    # The hash is over real weights, and verifying against it succeeds.
    digest = sae.weights_sha256()
    assert digest.startswith("sha256:") and len(digest.split(":", 1)[1]) == 64
    assert sae.verify_integrity(digest) is True
    assert sae.verify_integrity("sha256:" + "f" * 64) is False


@pytest.mark.skipif(not _has_torch(), reason="torch not installed")
def test_sae_distinct_inputs_give_distinct_encodings(tmp_path):
    """A real encoder must not return the same features for every input."""
    import torch

    from backend.interpretability.sae.loader import LocalCheckpointProvider

    torch.manual_seed(1)
    path = tmp_path / "sae.pt"
    torch.save({"W_enc": torch.randn(6, 10)}, path)
    sae = LocalCheckpointProvider().load(str(path), "test")

    a = sae.activate([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    b = sae.activate([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    assert a["feature_indices"] != b["feature_indices"]


# ── ACDC fidelity is measured, not assumed ─────────────────────────────────

def test_circuit_fidelity_reports_undefined_rather_than_guessing():
    """No retained circuit, or a zero gap, means undefined -- not a number."""
    from backend.interpretability.discovery.live_measure import circuit_fidelity

    out = circuit_fidelity("clean text", "corrupted text", 1, 2, set())
    assert out["measured"] is False
    assert out["fidelity"] is None
    assert out["reason"]


def test_acdc_no_longer_reports_an_unreachable_fidelity():
    """The 0.90-floored formula must not reappear in the algorithm."""
    import inspect

    from backend.interpretability.discovery.algorithms import acdc

    source = inspect.getsource(acdc)
    assert "0.90 +" not in source
    assert "0.09 *" not in source
    # It now delegates to a real measurement.
    assert "circuit_fidelity" in source


# ── Path patching requires the ids it measures against ─────────────────────

def test_path_patching_refuses_without_token_ids():
    """No io_id/subject_id means no logit difference to isolate a path on."""
    from backend.interpretability.discovery.algorithms.path_patching import (
        PathPatchingAlgorithm,
    )
    from backend.science.models.gpt2_adapter import GPT2Adapter

    algorithm = PathPatchingAlgorithm(GPT2Adapter(variant="small", mock_mode=True))
    report = algorithm.run({"clean": "When Alice and Bob went, Alice gave to",
                            "corrupted": "When Alice and Bob went, Bob gave to"})

    assert report.confidence == 0.0
    assert report.graph["edges"] == []
    assert report.provenance["measured"] is False
    assert report.provenance["publication_eligible"] is False
    assert report.provenance["reason"]


def test_path_patching_declares_the_real_procedure():
    from backend.interpretability.discovery.algorithms.path_patching import (
        PathPatchingAlgorithm,
    )

    # Both interventions are what make the measurement a path effect.
    assert PathPatchingAlgorithm.implements_published_method is True

    import inspect

    from backend.interpretability.discovery import live_measure
    source = inspect.getsource(live_measure.path_patch)
    assert "register_forward_hook" in source   # freezes the sender
    assert "register_forward_pre_hook" in source  # swaps the receiver input


def test_path_patching_is_live_measurement_code():
    """No fabricated `0.5 + delta` confidence on edges."""
    import inspect

    from backend.interpretability.discovery.algorithms import path_patching

    source = inspect.getsource(path_patching)
    assert "0.5 + delta" not in source
    assert "path_effect" in source
