"""Attestation guarantees for ModelFingerprintEngine.

These tests exist because the engine previously returned hardcoded placeholder
hashes (``"sha256:8f43c...model_weights_mock"``) and a hardcoded parameter
count, which made ``verify()`` report a match between any two captures and
turned drift detection into a no-op.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.science.publications.paper_validator import PaperValidator
from backend.science.reproducibility.model_fingerprint import (
    ModelFingerprint,
    ModelFingerprintEngine,
)
from backend.science.reproducibility.scientific_validator import ScientificValidator

import torch


def _adapter(mock_mode: bool = False, with_model: bool = True):
    """Build a minimal duck-typed ModelAdapter."""
    spec = SimpleNamespace(
        model_id="gpt2-small",
        hf_repo_id="gpt2",
        mock_mode=mock_mode,
    )
    model = SimpleNamespace(
        state_dict=lambda: {"w": torch.tensor([1.0, 2.0])},
        config=SimpleNamespace(to_dict=lambda: {"n_layer": 2}, architectures=["GPT2LMHeadModel"]),
        parameters=lambda: iter([torch.nn.Parameter(torch.tensor([1.0, 2.0]))]),
    )
    tokenizer = SimpleNamespace(get_vocab=lambda: {"a": 0, "b": 1})
    return SimpleNamespace(
        spec=spec,
        _model=model if with_model else None,
        _tokenizer=tokenizer,
    )


def test_capture_hashes_real_artifacts_not_placeholders():
    fp = ModelFingerprintEngine().capture(_adapter())

    assert fp.attested is True, fp.attestation_reason
    # A real 64-hex sha256 digest, not a truncated "...mock" literal.
    for value in (fp.weights_sha256, fp.config_sha256, fp.tokenizer_sha256):
        assert value.startswith("sha256:")
        digest = value.split(":", 1)[1]
        assert len(digest) == 64, value
        assert all(c in "0123456789abcdef" for c in digest), value
    assert "mock" not in fp.weights_sha256


def test_parameter_count_comes_from_the_model():
    fp = ModelFingerprintEngine().capture(_adapter())
    assert fp.parameter_count == 2  # the two real parameters, not a hardcoded 124M


def test_mock_mode_adapter_fails_closed():
    fp = ModelFingerprintEngine().capture(_adapter(mock_mode=True))

    assert fp.attested is False
    assert fp.weights_sha256 == "unattested"
    assert "mock_mode" in fp.attestation_reason


def test_missing_model_fails_closed():
    fp = ModelFingerprintEngine().capture(_adapter(with_model=False))

    assert fp.attested is False
    assert "no loaded model" in fp.attestation_reason


def test_unattested_pair_never_reports_a_match():
    """The old bug: two captures always compared equal, so drift never fired."""
    engine = ModelFingerprintEngine()
    a = engine.capture(_adapter(mock_mode=True))
    b = engine.capture(_adapter(mock_mode=True))

    verdict = engine.verify(a, b)

    assert verdict["is_match"] is False
    assert verdict["drift_severity"] == "UNVERIFIABLE"
    assert "attestation" in verdict["mismatches"]


def test_identical_real_fingerprints_match():
    engine = ModelFingerprintEngine()
    a = engine.capture(_adapter())
    b = engine.capture(_adapter())

    assert engine.verify(a, b)["is_match"] is True


def test_changed_weights_are_detected():
    """Drift detection must actually be able to fire now."""
    engine = ModelFingerprintEngine()
    a = engine.capture(_adapter())

    drifted = _adapter()
    drifted._model.state_dict = lambda: {"w": torch.tensor([1.0, 99.0])}
    b = engine.capture(drifted)

    verdict = engine.verify(a, b)
    assert verdict["is_match"] is False
    assert verdict["drift_severity"] == "CRITICAL"
    assert "weights_sha256" in verdict["mismatches"]


def test_scientific_validator_refuses_unattested_model():
    """A certificate must never be minted from mock weights."""
    validator = ScientificValidator()

    with pytest.raises(ValueError, match="not attested"):
        validator.generate_validation_artifacts(
            benchmark_results=[],
            adapter=_adapter(mock_mode=True),
            dataset_id="ioi",
            output_dir="pytest_attestation_artifacts",
        )


def test_paper_validator_rejects_unattested_fingerprint():
    base = {
        "power": {"observed_power": 0.95},
        "paper": {"doi": "10.1234/x"},
        "dataset_certificate": {"hash": "abc"},
        "statistical_results": {"n": 100},
    }

    good = ModelFingerprintEngine().capture(_adapter()).to_dict()
    assert PaperValidator.validate({**base, "model_fingerprint": good})["is_valid"] is True

    bad = ModelFingerprintEngine().capture(_adapter(mock_mode=True)).to_dict()
    result = PaperValidator.validate({**base, "model_fingerprint": bad})
    assert result["is_valid"] is False
    assert any("not attested" in e for e in result["errors"])

    missing = PaperValidator.validate(base)
    assert missing["is_valid"] is False


def test_no_placeholder_hashes_remain_in_source():
    """Guard against the mock literals being reintroduced."""
    source = (
        Path(__file__).resolve().parents[2]
        / "backend" / "science" / "reproducibility" / "model_fingerprint.py"
    ).read_text(encoding="utf-8")

    assert "_mock" not in source
    assert "124_000_000" not in source
