r"""Typed Exceptions for SAE Subsystem Validation and Execution."""

from __future__ import annotations


class SAEBaseError(Exception):
    """Base exception for all SAE errors in MECH."""
    pass


class SAELoadError(SAEBaseError):
    """Raised when an SAE checkpoint fails to load from disk, Hugging Face, or SAELens."""
    pass


class SAEIncompatibleError(SAEBaseError):
    """Raised when an SAE configuration does not match the target model architecture or layer."""
    pass


class SAECorruptedError(SAEBaseError):
    """Raised when SAE weight tensors contain NaNs, Infs, or mismatched dimensions."""
    pass


class SAEExecutionError(SAEBaseError):
    """Raised when an SAE forward pass, intervention, or reconstruction fails."""
    pass
