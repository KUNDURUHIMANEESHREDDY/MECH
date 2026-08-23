import tempfile
from pathlib import Path
import pytest

from backend.storage.database import DesktopStorage
from backend.science.preregistration_engine import (
    PreregistrationEngine,
    PreregistrationRecord,
    LedgerEntry,
)


def test_preregistration_lifecycle_and_locking():
    """Validates creation, protocol freezing, and lock verification."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        storage = DesktopStorage(Path(tmpdir) / "prereg_test.db")
        storage.initialize()
        engine = PreregistrationEngine(storage=storage)

        record = PreregistrationRecord(
            investigation_id="inv_ioi_rigor",
            hypothesis_id="hyp_l9h9",
            hypothesis_title="L9H9 Name Mover Mediation",
            prediction_statement="Ablating L9H9 will decrease target indirect-object logit by > 1.0",
            target_component="L9H9",
            negative_control_component="L0H0",
            primary_metric="delta_logit",
            falsification_condition="delta_logit < 0.2",
            falsification_threshold=0.2,
            sample_size=100,
        )

        saved = engine.create_preregistration(record)
        assert saved.is_locked is False

        # Lock the protocol permanently
        locked = engine.lock_preregistration(saved.id)
        assert locked.is_locked is True
        assert locked.locked_at is not None

        # Verify persistence
        fetched = engine.get_preregistration(saved.id)
        assert fetched is not None
        assert fetched.is_locked is True
        assert fetched.target_component == "L9H9"


def test_experiment_ledger_and_negative_result_dignity():
    """Validates immutable ledger logging and proper handling of falsified hypotheses."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        storage = DesktopStorage(Path(tmpdir) / "ledger_test.db")
        storage.initialize()
        engine = PreregistrationEngine(storage=storage)

        # 1. Exploratory run
        engine.record_ledger_entry(
            LedgerEntry(
                investigation_id="inv_test",
                run_id="run_exp_1",
                mode="EXPLORATORY",
                prediction_matched=True,
                empirical_result_summary="Target logit shift observed: +1.85",
                falsification_evaluated=False,
                is_falsified=False,
            )
        )

        # 2. Confirmatory run with falsification outcome (Negative result)
        engine.record_ledger_entry(
            LedgerEntry(
                investigation_id="inv_test",
                preregistration_id="prereg_early_heads",
                run_id="run_conf_2",
                mode="CONFIRMATORY_LOCKED",
                prediction_matched=False,
                empirical_result_summary="Early heads ablation yielded delta_logit = 0.04 (FALSIFIED)",
                falsification_evaluated=True,
                is_falsified=True,
            )
        )

        entries = engine.list_ledger_entries("inv_test")
        assert len(entries) == 2
        assert entries[0].mode == "EXPLORATORY"
        assert entries[1].mode == "CONFIRMATORY_LOCKED"
        assert entries[1].is_falsified is True


def test_harking_protection_prevents_modifying_locked_preregistration():
    """Validates that any attempt to post-hoc mutate a locked protocol raises PreregistrationLockedError."""
    from backend.science.preregistration_engine import PreregistrationLockedError

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        storage = DesktopStorage(Path(tmpdir) / "hark_test.db")
        storage.initialize()
        engine = PreregistrationEngine(storage=storage)

        record = PreregistrationRecord(
            id="prereg_frozen_1",
            investigation_id="inv_test",
            hypothesis_id="hyp_1",
            hypothesis_title="Initial Hypothesis",
            prediction_statement="Predicts delta_logit > 1.0",
            target_component="L9H9",
            negative_control_component="L0H0",
            primary_metric="delta_logit",
            falsification_condition="delta_logit < 0.2",
            falsification_threshold=0.2,
        )
        engine.create_preregistration(record)
        engine.lock_preregistration(record.id)

        # Attempting post-hoc mutation (HARKing) MUST fail
        mutated_record = PreregistrationRecord(
            id="prereg_frozen_1",
            investigation_id="inv_test",
            hypothesis_id="hyp_1",
            hypothesis_title="Mutated Post-Hoc Hypothesis",
            prediction_statement="Lowered expectation post-hoc: delta_logit > 0.05",
            target_component="L0H0",  # Changed target component
            negative_control_component="L1H1",
            primary_metric="delta_logit",
            falsification_condition="delta_logit < 0.01",
            falsification_threshold=0.01,
        )

        with pytest.raises(PreregistrationLockedError):
            engine.create_preregistration(mutated_record)

