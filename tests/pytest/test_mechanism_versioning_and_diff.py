import tempfile
from pathlib import Path
import pytest

from backend.storage.database import DesktopStorage
from backend.science.mechanism_versioning import (
    MechanismVersioningEngine,
    MechanismVersion,
    compute_mechanism_diff,
)


def test_mechanism_versioning_and_diff():
    """Validates mechanism snapshot lineage and differential computation."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        storage = DesktopStorage(Path(tmpdir) / "mech_ver.db")
        storage.initialize()
        engine = MechanismVersioningEngine(storage=storage)

        # Version 1
        v1 = MechanismVersion(
            investigation_id="inv_ioi_evo",
            version_number=1,
            title="Initial IOI Circuit Hypothesis",
            components=["L9H9"],
            edges=[{"source": "L9H9", "target": "residual", "relationship": "CAUSAL"}],
            evidence_ids=["evi_1"],
            scientific_rationale="Initial Name Mover identification via single head ablation.",
        )
        saved_v1 = engine.save_version(v1)

        # Version 2
        v2 = MechanismVersion(
            investigation_id="inv_ioi_evo",
            version_number=2,
            parent_version_id=saved_v1.id,
            title="Refined Dual Name Mover & Feedforward Circuit",
            components=["L9H9", "L10H0", "MLP_L8"],
            edges=[
                {"source": "L9H9", "target": "residual", "relationship": "CAUSAL"},
                {"source": "L10H0", "target": "residual", "relationship": "CAUSAL"},
                {"source": "MLP_L8", "target": "residual", "relationship": "CANDIDATE"},
            ],
            evidence_ids=["evi_1", "evi_2", "evi_3"],
            scientific_rationale="Added secondary Name Mover L10H0 and candidate associative feedforward layer MLP_L8.",
        )
        saved_v2 = engine.save_version(v2)

        # Compute Diff
        diff = compute_mechanism_diff(saved_v1, saved_v2)
        assert diff.v1_number == 1
        assert diff.v2_number == 2
        assert diff.added_components == ["L10H0", "MLP_L8"]
        assert len(diff.added_edges) == 2
        assert diff.new_evidence_citations == ["evi_2", "evi_3"]

        # List version history
        history = engine.list_versions("inv_ioi_evo")
        assert len(history) == 2
        assert history[0].version_number == 1
        assert history[1].version_number == 2
