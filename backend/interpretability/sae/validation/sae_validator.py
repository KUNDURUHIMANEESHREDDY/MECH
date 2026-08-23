r"""SAE Validator for MECH.

Performs structural compatibility, numerical integrity, reconstruction fidelity,
and sparsity boundary audits on any SAE implementing SAEInterface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch

from ..sae_interface import SAEInterface
from .errors import SAEIncompatibleError, SAECorruptedError


@dataclass
class SAEValidationAudit:
    """Complete diagnostic audit of an SAE instance."""
    is_valid: bool
    sae_id: str
    model_id: str
    layer: int
    d_in: int
    d_sae: int
    origin_state: str
    reconstruction_mse: float
    explained_variance: float
    mean_l0: float
    active_ratio: float
    dead_features_count: int
    checks_passed: List[str]
    failures: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "sae_id": self.sae_id,
            "model_id": self.model_id,
            "layer": self.layer,
            "d_in": self.d_in,
            "d_sae": self.d_sae,
            "origin_state": self.origin_state,
            "reconstruction_mse": round(self.reconstruction_mse, 6),
            "explained_variance": round(self.explained_variance, 4),
            "mean_l0": round(self.mean_l0, 2),
            "active_ratio": round(self.active_ratio, 6),
            "dead_features_count": self.dead_features_count,
            "checks_passed": self.checks_passed,
            "failures": self.failures,
        }


class SAEValidator:
    """Validates SAE instances against empirical interpretability quality standards."""

    @classmethod
    def validate_compatibility(
        cls,
        sae: SAEInterface,
        expected_d_in: int = 768,
        expected_layer: Optional[int] = None,
        expected_model_id: Optional[str] = None,
    ) -> None:
        """Validates that the SAE configuration matches model parameters."""
        meta = sae.metadata
        if meta.d_in != expected_d_in:
            raise SAEIncompatibleError(
                f"SAE d_in ({meta.d_in}) does not match expected model dimension ({expected_d_in})."
            )
        if expected_layer is not None and meta.layer != expected_layer:
            raise SAEIncompatibleError(
                f"SAE layer ({meta.layer}) does not match target layer ({expected_layer})."
            )
        if expected_model_id is not None and meta.model_id != expected_model_id:
            raise SAEIncompatibleError(
                f"SAE model_id ('{meta.model_id}') does not match expected model ('{expected_model_id}')."
            )

    @classmethod
    def validate_numerical_integrity(cls, sae: SAEInterface) -> None:
        """Checks for NaNs, Infs, or degenerated zero vectors in SAE feature weights."""
        meta = sae.metadata
        sample_feat = sae.get_feature_direction(0)
        if torch.isnan(sample_feat).any() or torch.isinf(sample_feat).any():
            raise SAECorruptedError(f"SAE '{meta.sae_id}' contains NaN or Inf feature directions.")
        norm = torch.norm(sample_feat).item()
        if norm < 1e-6:
            raise SAECorruptedError(f"SAE '{meta.sae_id}' has degenerated zero feature direction (norm={norm}).")

    @classmethod
    def audit_full_sae(
        cls,
        sae: SAEInterface,
        sample_activations: torch.Tensor,
        max_mse_threshold: float = 2.0,
        min_explained_variance: float = 0.40,
    ) -> SAEValidationAudit:
        """Runs an end-to-end diagnostic audit on an SAE with real or sample activations."""
        checks_passed = []
        failures = []

        meta = sae.metadata
        origin = meta.provenance.origin_state.value if meta.provenance else "UNKNOWN"

        # Check 1: Dimensions
        if meta.d_in > 0 and meta.d_sae > 0:
            checks_passed.append("Dimension non-zero check passed")
        else:
            failures.append("Invalid non-positive dimensions")

        # Check 2: Numerical Integrity
        try:
            cls.validate_numerical_integrity(sae)
            checks_passed.append("Numerical integrity (no NaN/Inf) check passed")
        except Exception as exc:
            failures.append(f"Numerical integrity failure: {exc}")

        # Check 3: Reconstruction & Sparsity
        z, x_hat = sae.reconstruct(sample_activations)
        rec_metrics = sae.get_reconstruction_error(sample_activations, x_hat)
        sparsity_metrics = sae.get_sparsity(z)

        mse = rec_metrics["mse"]
        exp_var = rec_metrics["explained_variance"]
        l0 = sparsity_metrics["l0"]
        active_ratio = sparsity_metrics["active_ratio"]

        if mse <= max_mse_threshold:
            checks_passed.append(f"Reconstruction MSE ({mse:.4f}) <= threshold ({max_mse_threshold})")
        else:
            failures.append(f"Reconstruction MSE ({mse:.4f}) exceeds threshold ({max_mse_threshold})")

        if exp_var >= min_explained_variance:
            checks_passed.append(f"Explained variance ({exp_var:.2%}) >= threshold ({min_explained_variance:.2%})")
        else:
            failures.append(f"Explained variance ({exp_var:.2%}) below threshold ({min_explained_variance:.2%})")

        # Check 4: Dead features on sample
        z_flat = z.view(-1, meta.d_sae)
        active_features_mask = (z_flat > 1e-4).any(dim=0)
        dead_count = int((~active_features_mask).sum().item())

        is_valid = len(failures) == 0

        return SAEValidationAudit(
            is_valid=is_valid,
            sae_id=meta.sae_id,
            model_id=meta.model_id,
            layer=meta.layer,
            d_in=meta.d_in,
            d_sae=meta.d_sae,
            origin_state=origin,
            reconstruction_mse=mse,
            explained_variance=exp_var,
            mean_l0=l0,
            active_ratio=active_ratio,
            dead_features_count=dead_count,
            checks_passed=checks_passed,
            failures=failures,
        )
