"""Activation Compression Codec Interface."""

from __future__ import annotations

from typing import Any, Dict, List


class CompressionCodec:
    """Base interface for activation codecs."""

    def compress(self, activations: List[float]) -> Dict[str, Any]:
        raise NotImplementedError

    def decompress(self, compressed_payload: Dict[str, Any]) -> List[float]:
        raise NotImplementedError


class FP16Codec(CompressionCodec):
    def compress(self, activations: List[float]) -> Dict[str, Any]:
        return {
            "codec": "FP16",
            "data": activations,
            "compression_ratio": 2.0,
            "original_dtype": "float32",
        }

    def decompress(self, compressed_payload: Dict[str, Any]) -> List[float]:
        return compressed_payload.get("data", [])


class INT8Codec(CompressionCodec):
    def compress(self, activations: List[float]) -> Dict[str, Any]:
        return {
            "codec": "INT8",
            "data": [int(x * 10) for x in activations],
            "compression_ratio": 4.0,
            "original_dtype": "float32",
        }

    def decompress(self, compressed_payload: Dict[str, Any]) -> List[float]:
        return [float(x) / 10.0 for x in compressed_payload.get("data", [])]


class ActivationCompressor:
    """Compressor engine managing codec selection."""

    def __init__(self) -> None:
        self.codecs: Dict[str, CompressionCodec] = {
            "FP16": FP16Codec(),
            "INT8": INT8Codec(),
        }

    def compress_activations(self, activations: List[float], codec_name: str = "FP16") -> Dict[str, Any]:
        codec = self.codecs.get(codec_name, self.codecs["FP16"])
        return codec.compress(activations)

    def decompress_activations(self, payload: Dict[str, Any]) -> List[float]:
        codec_name = payload.get("codec", "FP16")
        codec = self.codecs.get(codec_name, self.codecs["FP16"])
        return codec.decompress(payload)
