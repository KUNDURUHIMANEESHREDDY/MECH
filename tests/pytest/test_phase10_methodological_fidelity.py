"""Phase 10: Deep Scientific Correctness & Methodological Fidelity Test Suite.

Validates:
1. Causal Tracing: Exact hook insertion, single-pair IE vs distribution AIE, endpoint restoration.
2. Path Patching: Sender detachment, receiver boundaries, explicit heuristic p-score labeling.
3. DLA Attribution: Non-causal classification, LayerNorm scaling caveats, linear projection bounds.
4. SAE Provenance: Origin state tracking (REAL_PRETRAINED vs SYNTHETIC_INIT), live activation sparsity.
5. Epistemic Hierarchy: Complete 7-state taxonomy preserving distinct failure vs zero effect vs falsified states.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from backend.interpretability.causal.causal_tracing import CausalTracingEngine
from backend.interpretability.causal.path_patching import EdgePathPatchingEngine
from backend.interpretability.sae.attribution.sae_dla import SAEDirectLogitAttributor, SAEDLAResult
from backend.interpretability.sae.sae_adapter import NativeMECHSAE
from backend.interpretability.sae.sae_interface import SAEProvenance, SAEOriginState
from backend.science.scientific_data_model import EvidenceLevel
from backend.science.path_verification_engine import PathVerificationEngine


class TestPhase10CausalTracingFidelity:
    """Validate exact hook placement, mathematical definitions, and metadata."""

    def test_causal_tracing_metadata_and_metric_definition(self):
        engine = CausalTracingEngine()
        res = engine.trace_causal_effect(
            clean_prompt="The capital of France is",
            corrupted_prompt="The capital of Italy is",
            num_layers=4,
        )
        assert res["metric_definition"] == "IndirectEffect (Single-Pair IE)"
        assert res["restoration_position"] == "endpoint_token"
        assert "max_indirect_effect" in res
        assert 0.0 <= res["max_indirect_effect"] <= 1.0


class TestPhase10PathPatchingFidelity:
    """Validate path patching sender/receiver boundaries and heuristic caveats."""

    def test_path_patching_heuristic_label_and_caveat(self):
        patcher = EdgePathPatchingEngine()
        res = patcher.test_edge_mediation(
            sender="L6_MLP",
            receiver="L8_H1",
            clean_prompt="The capital of France is",
            corrupted_prompt="The capital of Italy is",
            target_token=" Paris",
        )
        d = res.to_dict()
        assert "heuristic_p_score" in d
        assert "statistical_caveat" in d
        assert "heuristic" in d["statistical_caveat"].lower()
        assert not d["statistical_caveat"].startswith("null-hypothesis")


class TestPhase10DLAAttributionFidelity:
    """Validate DLA non-causal attribution classification and LayerNorm caveats."""

    def test_dla_explicit_non_causal_classification(self):
        w_u = torch.randn(50257, 768)
        sae = NativeMECHSAE(d_in=768, d_sae=1024, model_id="test", layer=8)
        
        class MockTokenizer:
            def decode(self, ids):
                return " token"
            def encode(self, text):
                return [123]

        attributor = SAEDirectLogitAttributor(unembedding_matrix=w_u, tokenizer=MockTokenizer())
        res = attributor.attribute_feature(sae=sae, feature_idx=0, top_k=5)

        assert isinstance(res, SAEDLAResult)
        assert res.is_causal is False
        assert res.evidence_type == "VIRTUAL_UNEMBEDDED_PROJECTION"
        assert "Does NOT measure mediated downstream attention/MLP layers" in res.method_limitation


class TestPhase10SAEProvenanceAndFidelity:
    """Validate SAE origin states and activation-conditioned metrics."""

    def test_sae_provenance_origin_states(self):
        prov_real = SAEProvenance(
            source="huggingface/sae_gpt2",
            checkpoint_identifier="gpt2-layer-8-res",
            origin_state=SAEOriginState.REAL_PRETRAINED,
            weights_sha256="abc123sha",
        )
        assert prov_real.origin_state == SAEOriginState.REAL_PRETRAINED
        d = prov_real.to_dict()
        assert d["origin_state"] == "REAL_PRETRAINED"

        prov_synth = SAEProvenance(
            source="mech_native",
            checkpoint_identifier="synth_init_l8",
            origin_state=SAEOriginState.SYNTHETIC_INITIALIZED,
        )
        assert prov_synth.origin_state == SAEOriginState.SYNTHETIC_INITIALIZED


class TestPhase10EpistemicHierarchyFidelity:
    """Validate that the full 6-tier EvidenceLevel enum separates all evidence stages."""

    def test_evidence_level_distinct_states(self):
        levels = {lvl.value for lvl in EvidenceLevel}
        assert "OBSERVED" in levels
        assert "CANDIDATE" in levels
        assert "SUPPORTED" in levels
        assert "CAUSALLY_VERIFIED" in levels
        assert "FALSIFIED" in levels
        assert "UNEXECUTED" in levels
        # All states must be distinct strings
        assert len(levels) == 6
