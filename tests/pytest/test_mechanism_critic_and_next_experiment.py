import tempfile
from pathlib import Path
import pytest

from backend.storage.database import DesktopStorage
from backend.storage.scientific_entities import ExperimentRun
from backend.science.mechanism_critic import MechanismCriticEngine


def test_mechanism_critic_audit_and_gap_detection():
    """Validates mechanism critique and missing control identification."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        storage = DesktopStorage(Path(tmpdir) / "critic_test.db")
        storage.initialize()
        engine = MechanismCriticEngine(storage=storage)

        # 1. State with only 1 un-controlled run
        storage.save_run(
            ExperimentRun(
                id="run_1",
                experiment_id="exp_1",
                investigation_id="inv_critic_test",
                delta_logit=1.85,
            ).model_dump()
        )

        audit1 = engine.audit_claim("inv_critic_test", "L9H9")
        assert audit1.has_causal_intervention is True
        assert audit1.has_negative_control is False
        assert audit1.epistemic_grade == "OBSERVATIONAL_CORRELATION"

        # Recommends negative control as next best experiment
        rec1 = engine.recommend_next_experiment("inv_critic_test")
        assert rec1.target_component == "L0H0"
        assert rec1.expected_information_gain == "HIGH"

        # 2. Add negative control and second replication run
        storage.save_run(
            ExperimentRun(
                id="run_2",
                experiment_id="exp_2",
                investigation_id="inv_critic_test",
                delta_logit=1.84,
                control_delta_logit=0.03,
            ).model_dump()
        )

        audit2 = engine.audit_claim("inv_critic_test", "L9H9")
        assert audit2.has_negative_control is True
        assert audit2.has_deterministic_replication is True
        assert audit2.epistemic_grade == "ROBUST_CAUSAL"

        # Next best experiment pivots to resolving competing MLP hypothesis
        rec2 = engine.recommend_next_experiment("inv_critic_test")
        assert rec2.target_component == "MLP_L8"
