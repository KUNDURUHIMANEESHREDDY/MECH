"""Prediction Calibration Engine with Versioned Calibration Models.

Measures error diffs between predicted simulator metrics (runtime, VRAM) and actual observed telemetry,
maintaining immutable calibration model revisions (cal_v1, cal_v2, cal_v3) with complete update rationale.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class CalibrationRecord:
    record_id: str
    model_name: str
    target_backend: str
    predicted_runtime_sec: float
    observed_runtime_sec: float
    predicted_vram_gb: float
    observed_vram_gb: float
    runtime_calibration_error_pct: float
    vram_calibration_error_pct: float
    calibration_factor: float
    recorded_at: str


@dataclass
class CalibrationModelVersion:
    version_id: str
    version_number: str
    model_name: str
    target_backend: str
    training_episodes_count: int
    average_error_pct: float
    calibration_factor: float
    confidence_score: float
    update_rationale: str
    created_at: str


class PredictionCalibrationEngine:
    """Tracks calibration drift and maintains immutable versioned calibration models."""

    def __init__(self) -> None:
        self.calibration_history: List[CalibrationRecord] = [
            CalibrationRecord(
                record_id="cal_init_1",
                model_name="GPT-2 Small",
                target_backend="Ray",
                predicted_runtime_sec=48.0,
                observed_runtime_sec=51.2,
                predicted_vram_gb=8.4,
                observed_vram_gb=8.9,
                runtime_calibration_error_pct=6.67,
                vram_calibration_error_pct=5.95,
                calibration_factor=1.0667,
                recorded_at=_dt.datetime.utcnow().isoformat() + "Z",
            )
        ]
        init_version = CalibrationModelVersion(
            version_id="cal_mod_v1",
            version_number="1.0.0",
            model_name="GPT-2 Small",
            target_backend="Ray",
            training_episodes_count=1,
            average_error_pct=6.67,
            calibration_factor=1.0667,
            confidence_score=0.94,
            update_rationale="Initial baseline calibration model trained on 1 episode.",
            created_at=_dt.datetime.utcnow().isoformat() + "Z",
        )
        self.model_versions: Dict[str, List[CalibrationModelVersion]] = {"GPT-2 Small_Ray": [init_version]}

    def record_execution_outcome(
        self,
        model_name: str,
        target_backend: str,
        predicted_runtime_sec: float,
        observed_runtime_sec: float,
        predicted_vram_gb: float,
        observed_vram_gb: float,
    ) -> Dict[str, Any]:
        runtime_err = round(abs(observed_runtime_sec - predicted_runtime_sec) / (predicted_runtime_sec or 1.0) * 100.0, 2)
        vram_err = round(abs(observed_vram_gb - predicted_vram_gb) / (predicted_vram_gb or 1.0) * 100.0, 2)
        cal_factor = round(observed_runtime_sec / (predicted_runtime_sec or 1.0), 4)

        record_id = f"cal_{model_name.lower().replace(' ', '_')}_{len(self.calibration_history) + 1}"
        record = CalibrationRecord(
            record_id=record_id,
            model_name=model_name,
            target_backend=target_backend,
            predicted_runtime_sec=predicted_runtime_sec,
            observed_runtime_sec=observed_runtime_sec,
            predicted_vram_gb=predicted_vram_gb,
            observed_vram_gb=observed_vram_gb,
            runtime_calibration_error_pct=runtime_err,
            vram_calibration_error_pct=vram_err,
            calibration_factor=cal_factor,
            recorded_at=_dt.datetime.utcnow().isoformat() + "Z",
        )
        self.calibration_history.append(record)

        # Update versioned model
        key = f"{model_name}_{target_backend}"
        versions = self.model_versions.setdefault(key, [])
        v_num = len(versions) + 1
        new_version = CalibrationModelVersion(
            version_id=f"cal_mod_{key.lower().replace(' ', '_')}_v{v_num}",
            version_number=f"{v_num}.0.0",
            model_name=model_name,
            target_backend=target_backend,
            training_episodes_count=len([r for r in self.calibration_history if r.model_name == model_name]),
            average_error_pct=runtime_err,
            calibration_factor=cal_factor,
            confidence_score=round(max(0.70, 0.98 - runtime_err * 0.01), 2),
            update_rationale=f"Updated calibration model version {v_num}.0.0 with new telemetry diff ({runtime_err}% error).",
            created_at=_dt.datetime.utcnow().isoformat() + "Z",
        )
        versions.append(new_version)

        return {"record": asdict(record), "latest_model_version": asdict(new_version)}

    def get_calibration_factor(self, model_name: str, target_backend: str) -> float:
        key = f"{model_name}_{target_backend}"
        versions = self.model_versions.get(key)
        if versions:
            return versions[-1].calibration_factor
        return 1.0

    def list_model_versions(self, model_name: str = "GPT-2 Small", target_backend: str = "Ray") -> List[Dict[str, Any]]:
        key = f"{model_name}_{target_backend}"
        versions = self.model_versions.get(key, [])
        return [asdict(v) for v in versions]

    def get_calibration_summary(self) -> Dict[str, Any]:
        if not self.calibration_history:
            return {"status": "NoData", "average_calibration_error_pct": 0.0}

        avg_err = sum(r.runtime_calibration_error_pct for r in self.calibration_history) / len(self.calibration_history)
        return {
            "total_records": len(self.calibration_history),
            "average_runtime_calibration_error_pct": round(avg_err, 2),
            "latest_record": asdict(self.calibration_history[-1]),
            "model_is_well_calibrated": avg_err < 10.0,
        }
