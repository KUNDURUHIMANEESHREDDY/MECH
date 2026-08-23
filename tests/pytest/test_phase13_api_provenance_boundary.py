"""Phase 13 — API Provenance Boundary Enforcement Tests.

Tests that verify the invariant:
  A scientific result CANNOT reach the frontend without a valid manifest_id
  at the API router boundary.

Four tests:
    13.1  _wrap() with a real manifest returns envelope with manifest_id + provenance_hash
    13.2  _wrap() with no manifest_id returns UNEXECUTED, result=None
    13.3  _wrap() with declared manifest_id missing from store returns UNEXECUTED, result=None
    13.4  manifest_id from _wrap is retrievable from manifest_store
"""

from __future__ import annotations

import hashlib
import json

import pytest

from backend.api.scientific_router import ScientificResponseEnvelope, _wrap
from backend.science.provenance.experiment_manifest import (
    ExperimentManifest,
    build_control_distribution,
    build_bootstrap_statistics,
    _library_versions,
    _hash_prompts,
)
from backend.science.provenance import manifest_store
from backend.science.scientific_data_model import EvidenceLevel


def _make_and_write_manifest(exp_id: str) -> ExperimentManifest:
    ctrl_dist = build_control_distribution([0.01, 0.02, 0.015])
    bs = build_bootstrap_statistics([0.25, 0.30, 0.28])
    manifest = ExperimentManifest(
        experiment_id=exp_id,
        experiment_type="induction_head",
        timestamp_utc="2026-08-17T00:00:00Z",
        mech_version="2.0.0",
        device="cpu",
        precision="float32",
        library_versions=_library_versions(),
        model_id="gpt2",
        model_weights_sha256="test_sha256",
        tokenizer_hash="test_tok_sha256",
        sae_id=None,
        sae_weights_sha256=None,
        prompt_dataset_hash=_hash_prompts(["test prompt"]),
        prompts_used=["test prompt"],
        random_seeds=[42],
        intervention_specification={"layer": 5, "head": 5},
        baseline_measurements={"prefix_attention_score": 0.42},
        raw_effect_measurements=[0.25, 0.30, 0.28],
        control_distribution=ctrl_dist,
        replication_measurements=[],
        bootstrap_statistics=bs,
        spearman_rho=None,
        spearman_p_value=None,
        pearson_r=None,
        dla_approximation_quality=None,
        evidence_level=EvidenceLevel.SUPPORTED.value,
        falsification_status=EvidenceLevel.SUPPORTED.value,
        is_reproducible=True,
        statistical_caveat="Test caveat for Phase 13 API boundary test.",
    )
    manifest_store.write(manifest)
    return manifest


class TestApiProvenanceBoundary:

    def test_13_1_wrap_with_real_manifest_returns_valid_envelope(self):
        """_wrap() with a real manifest_id produces a valid envelope with provenance_hash."""
        exp_id = hashlib.sha256(b"test_13_1_api_boundary").hexdigest()[:16]
        manifest = _make_and_write_manifest(exp_id)

        engine_result = {
            "manifest_id": exp_id,
            "evidence_level": EvidenceLevel.SUPPORTED.value,
            "statistical_caveat": "Test caveat for Phase 13 API boundary test.",
            "prefix_attention_score": 0.42,
        }
        envelope = _wrap(engine_result)

        # Must be a proper envelope
        assert isinstance(envelope, ScientificResponseEnvelope)
        assert envelope.manifest_id == exp_id
        assert envelope.provenance_hash is not None
        assert len(envelope.provenance_hash) == 64  # SHA-256 hex
        assert envelope.evidence_level == EvidenceLevel.SUPPORTED.value
        assert envelope.result is not None
        assert envelope.is_reproducible is True

    def test_13_2_wrap_without_manifest_id_returns_unexecuted_no_result(self):
        """_wrap() with no manifest_id in the engine result returns UNEXECUTED, result=None.

        This is the critical loophole: a legacy endpoint that never writes a manifest
        must be blocked from delivering a result payload to the frontend.
        """
        engine_result = {
            # No manifest_id field — simulates legacy/custom endpoint
            "evidence_level": EvidenceLevel.SUPPORTED.value,
            "prefix_attention_score": 0.42,
        }
        envelope = _wrap(engine_result)

        assert envelope.manifest_id is None
        assert envelope.provenance_hash is None
        assert envelope.evidence_level == EvidenceLevel.UNEXECUTED.value
        assert envelope.result is None
        assert envelope.is_reproducible is False
        assert "No manifest_id" in envelope.statistical_caveat or "UNEXECUTED" in envelope.statistical_caveat

    def test_13_3_wrap_with_missing_manifest_returns_unexecuted_no_result(self):
        """_wrap() with declared manifest_id that is not in the store returns UNEXECUTED, result=None.

        This covers the case where a manifest_id field is present but the manifest was
        never actually written to disk.
        """
        ghost_id = "ghost_experiment_id_does_not_exist"
        engine_result = {
            "manifest_id": ghost_id,
            "evidence_level": EvidenceLevel.SUPPORTED.value,
            "prefix_attention_score": 0.42,
        }
        envelope = _wrap(engine_result)

        assert envelope.manifest_id == ghost_id
        assert envelope.provenance_hash is None
        assert envelope.evidence_level == EvidenceLevel.UNEXECUTED.value
        assert envelope.result is None
        assert "not found on disk" in envelope.statistical_caveat or "UNEXECUTED" in envelope.statistical_caveat

    def test_13_4_manifest_id_from_wrap_is_retrievable_from_store(self):
        """After _wrap() succeeds, manifest_store.retrieve(manifest_id) returns the matching manifest."""
        exp_id = hashlib.sha256(b"test_13_4_retrievable").hexdigest()[:16]
        _make_and_write_manifest(exp_id)

        engine_result = {
            "manifest_id": exp_id,
            "evidence_level": EvidenceLevel.SUPPORTED.value,
            "statistical_caveat": "Test caveat for Phase 13 API boundary test.",
        }
        envelope = _wrap(engine_result)
        assert envelope.manifest_id is not None

        # Must be retrievable from disk
        retrieved = manifest_store.retrieve(envelope.manifest_id)
        assert retrieved["experiment_id"] == exp_id
        assert retrieved["evidence_level"] == EvidenceLevel.SUPPORTED.value
        assert retrieved["is_reproducible"] is True

        # Provenance hash must match
        recomputed_hash = hashlib.sha256(
            json.dumps(retrieved, sort_keys=True).encode()
        ).hexdigest()
        assert envelope.provenance_hash == recomputed_hash
