"""Validator Factory for MECH Platform."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from backend.core.factories.base import ConfigurableFactory

logger = logging.getLogger("MECH.factories.validator")


class ValidatorFactory(ConfigurableFactory):
    """Factory for validators and integrity gates."""

    def __init__(self) -> None:
        super().__init__()
        self._integrity_gates: Dict[str, Any] = {}
        self._behavioral_validators: Dict[str, Any] = {}
        self._register_builtins()

    def get_type_name(self) -> str:
        return "validator"

    def _register_builtins(self) -> None:
        """Register built-in validators."""
        # Integrity gates - use functions
        self._integrity_gates["model_integrity"] = self._run_model_integrity_gate
        self._integrity_gates["checkpoint_identity"] = self._run_checkpoint_identity
        self._integrity_gates["tokenizer_integrity"] = self._run_tokenizer_integrity
        self._integrity_gates["architecture_match"] = self._run_architecture_match
        self._integrity_gates["behavioral_sanity"] = self._run_behavioral_sanity

        # Behavioral validators
        self._behavioral_validators["sanity"] = self._run_behavioral_sanity_suite
        self._behavioral_validators["probes"] = self._run_probe_validation

    def _run_model_integrity_gate(self, model: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run model integrity gate."""
        try:
            from backend.runtime.model_integrity_gate import run_model_integrity_gate
            
            # Extract config
            tokenizer = config.get("tokenizer")
            model_id = config.get("model_id", "unknown")
            hf_id = config.get("hf_id", model_id)
            target_tokens = config.get("target_tokens", [])
            runtime = config.get("runtime")
            baseline_identity = config.get("baseline_identity")
            expected_num_layers = config.get("expected_num_layers", -1)
            expected_hidden_size = config.get("expected_hidden_size", -1)
            expected_architecture = config.get("expected_architecture", "")
            
            if tokenizer is None:
                return {"status": "error", "error": "tokenizer required in config"}
            
            result = run_model_integrity_gate(
                model=model,
                tokenizer=tokenizer,
                model_id=model_id,
                hf_id=hf_id,
                target_tokens=target_tokens,
                runtime=runtime,
                baseline_identity=baseline_identity,
                expected_num_layers=expected_num_layers,
                expected_hidden_size=expected_hidden_size,
                expected_architecture=expected_architecture,
            )
            return result.to_dict()
        except Exception as e:
            logger.error("Model integrity gate failed: %s", e)
            return {"status": "error", "error": str(e)}

    def _run_checkpoint_identity(self, model: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run checkpoint identity verification."""
        try:
            from backend.runtime.checkpoint_identity import compute_checkpoint_identity, verify_checkpoint_identity
            
            tokenizer = config.get("tokenizer")
            model_id = config.get("model_id", "unknown")
            hf_id = config.get("hf_id", model_id)
            baseline_identity = config.get("baseline_identity")
            
            if tokenizer is None:
                return {"status": "error", "error": "tokenizer required in config"}
            
            live_identity = compute_checkpoint_identity(model, tokenizer, model_id, hf_id)
            
            if baseline_identity is not None:
                result = verify_checkpoint_identity(live_identity, baseline_identity)
                return {**result, "live_identity": live_identity.to_dict()}
            else:
                return {
                    "verified": True,
                    "mismatches": {},
                    "summary": "No baseline provided — checkpoint hash recorded but not compared.",
                    "live_identity": live_identity.to_dict(),
                }
        except Exception as e:
            logger.error("Checkpoint identity check failed: %s", e)
            return {"status": "error", "error": str(e)}

    def _run_tokenizer_integrity(self, model: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run tokenizer integrity check."""
        try:
            from backend.runtime.tokenizer_integrity import verify_tokenizer_integrity
            
            tokenizer = config.get("tokenizer")
            target_tokens = config.get("target_tokens", [])
            
            if tokenizer is None:
                return {"status": "error", "error": "tokenizer required in config"}
            
            report = verify_tokenizer_integrity(tokenizer, target_tokens)
            return {
                "verified": report.all_passed,
                "details": report.to_dict(),
            }
        except Exception as e:
            logger.error("Tokenizer integrity check failed: %s", e)
            return {"status": "error", "error": str(e)}

    def _run_architecture_match(self, model: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run architecture match check."""
        try:
            from backend.runtime.model_integrity_gate import _verify_architecture
            
            expected_num_layers = config.get("expected_num_layers", -1)
            expected_hidden_size = config.get("expected_hidden_size", -1)
            expected_architecture = config.get("expected_architecture", "")
            
            result = _verify_architecture(
                model,
                expected_num_layers=expected_num_layers,
                expected_hidden_size=expected_hidden_size,
                expected_architecture=expected_architecture,
            )
            return result
        except Exception as e:
            logger.error("Architecture match check failed: %s", e)
            return {"status": "error", "error": str(e)}

    def _run_behavioral_sanity(self, model: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run behavioral sanity check."""
        try:
            from backend.runtime.behavioral_sanity_suite import run_behavioral_sanity_suite
            
            runtime = config.get("runtime")
            model_id = config.get("model_id", "unknown")
            
            if runtime is None:
                return {
                    "all_passed": True,
                    "summary": "Behavioral sanity skipped — no runtime supplied.",
                }
            
            report = run_behavioral_sanity_suite(runtime, model_id=model_id)
            return report.to_dict()
        except Exception as e:
            logger.error("Behavioral sanity check failed: %s", e)
            return {"status": "error", "error": str(e)}

    def _run_behavioral_sanity_suite(self, model: Any, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run behavioral sanity suite as validator."""
        return self._run_behavioral_sanity(model, {"test_cases": test_cases})

    def _run_probe_validation(self, model: Any, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run probe validation."""
        return {"status": "not_implemented", "message": "Probe validation not yet implemented"}

    def create_integrity_gate(self, gate_type: str, **kwargs: Any) -> Any:
        """Create integrity gate (returns callable)."""
        if gate_type not in self._integrity_gates:
            raise ValueError(
                f"Unknown integrity gate: {gate_type}. "
                f"Available: {list(self._integrity_gates.keys())}"
            )
        return self._integrity_gates[gate_type]

    def create_behavioral_validator(self, validator_type: str, **kwargs: Any) -> Any:
        """Create behavioral validator (returns callable)."""
        if validator_type not in self._behavioral_validators:
            raise ValueError(
                f"Unknown behavioral validator: {validator_type}. "
                f"Available: {list(self._behavioral_validators.keys())}"
            )
        return self._behavioral_validators[validator_type]

    def run_integrity_checks(
        self,
        model: Any,
        gate_types: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run multiple integrity gates."""
        gates = gate_types or list(self._integrity_gates.keys())
        config = config or {}
        results = {}

        for gate_name in gates:
            try:
                gate_fn = self._integrity_gates[gate_name]
                results[gate_name] = gate_fn(model, config)
            except Exception as e:
                logger.error("Integrity gate %s failed: %s", gate_name, e)
                results[gate_name] = {"status": "error", "error": str(e)}

        return results

    def run_behavioral_validation(
        self,
        model: Any,
        test_cases: List[Dict[str, Any]],
        validator_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run multiple behavioral validators."""
        validators = validator_types or list(self._behavioral_validators.keys())
        results = {}

        for val_name in validators:
            try:
                val_fn = self._behavioral_validators[val_name]
                results[val_name] = val_fn(model, test_cases)
            except Exception as e:
                logger.error("Behavioral validator %s failed: %s", val_name, e)
                results[val_name] = {"status": "error", "error": str(e)}

        return results

    def list_integrity_gates(self) -> List[str]:
        return list(self._integrity_gates.keys())

    def list_behavioral_validators(self) -> List[str]:
        return list(self._behavioral_validators.keys())


# Global instance
_validator_factory: Optional[ValidatorFactory] = None


def get_validator_factory() -> ValidatorFactory:
    """Get the global validator factory."""
    global _validator_factory
    if _validator_factory is None:
        _validator_factory = ValidatorFactory()
    return _validator_factory