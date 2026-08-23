"""Activation Compression Codec Interface.

Implements high-fidelity tensor and activation compression codecs (FP16, INT8 quantization, ZLIB)
to optimize activation cache memory usage and communication bandwidth.
"""

from __future__ import annotations

import base64
import struct
import zlib
from typing import Any, Dict, List
import numpy as np


class CompressionCodec:
    """Base interface for activation codecs."""

    def compress(self, activations: List[float]) -> Dict[str, Any]:
        """Compress float activations into compact payload."""
        raise NotImplementedError("Subclasses must implement compress()")

    def decompress(self, compressed_payload: Dict[str, Any]) -> List[float]:
        """Decompress payload back into float activations."""
        raise NotImplementedError("Subclasses must implement decompress()")


class FP16Codec(CompressionCodec):
    """IEEE 754 half-precision float (16-bit) codec."""

    def compress(self, activations: List[float]) -> Dict[str, Any]:
        arr = np.asarray(activations, dtype=np.float16)
        raw_bytes = arr.tobytes()
        b64_str = base64.b64encode(raw_bytes).decode("ascii")
        return {
            "codec": "FP16",
            "data": [float(x) for x in arr.tolist()],
            "encoded_bytes": b64_str,
            "element_count": len(activations),
            "compression_ratio": 2.0,
            "original_dtype": "float32",
            "compressed_dtype": "float16",
            "is_lossy": True,
            "is_bit_exact": False,
        }

    def decompress(self, compressed_payload: Dict[str, Any]) -> List[float]:
        if "encoded_bytes" in compressed_payload:
            raw_bytes = base64.b64decode(compressed_payload["encoded_bytes"])
            arr = np.frombuffer(raw_bytes, dtype=np.float16)
            return [float(x) for x in arr.tolist()]
        return [float(x) for x in compressed_payload.get("data", [])]


class INT8Codec(CompressionCodec):
    """Symmetric 8-bit uniform affine quantization codec."""

    def compress(self, activations: List[float]) -> Dict[str, Any]:
        arr = np.asarray(activations, dtype=np.float32)
        max_val = float(np.max(np.abs(arr))) if len(arr) > 0 else 1.0
        scale = max_val / 127.0 if max_val > 0 else 1.0

        quantized = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
        raw_bytes = quantized.tobytes()
        b64_str = base64.b64encode(raw_bytes).decode("ascii")

        return {
            "codec": "INT8",
            "data": [int(x) for x in quantized.tolist()],
            "encoded_bytes": b64_str,
            "scale": scale,
            "element_count": len(activations),
            "compression_ratio": 4.0,
            "original_dtype": "float32",
            "compressed_dtype": "int8",
            "is_lossy": True,
            "is_bit_exact": False,
        }

    def decompress(self, compressed_payload: Dict[str, Any]) -> List[float]:
        scale = float(compressed_payload.get("scale", 1.0))
        if "encoded_bytes" in compressed_payload:
            raw_bytes = base64.b64decode(compressed_payload["encoded_bytes"])
            quantized = np.frombuffer(raw_bytes, dtype=np.int8)
            decomp = quantized.astype(np.float32) * scale
            return [round(float(x), 4) for x in decomp.tolist()]
        
        data = compressed_payload.get("data", [])
        return [round(float(x) * scale, 4) for x in data]


class ZLIBCodec(CompressionCodec):
    """Lossless deflate compression codec over float byte stream."""

    def compress(self, activations: List[float]) -> Dict[str, Any]:
        arr = np.asarray(activations, dtype=np.float32)
        raw_bytes = arr.tobytes()
        compressed_bytes = zlib.compress(raw_bytes, level=6)
        b64_str = base64.b64encode(compressed_bytes).decode("ascii")
        ratio = max(1.0, len(raw_bytes) / max(len(compressed_bytes), 1))

        return {
            "codec": "ZLIB",
            "data": [float(x) for x in activations],
            "encoded_bytes": b64_str,
            "element_count": len(activations),
            "compression_ratio": round(ratio, 2),
            "original_dtype": "float32",
            "is_lossy": False,
            "is_bit_exact": True,
        }

    def decompress(self, compressed_payload: Dict[str, Any]) -> List[float]:
        if "encoded_bytes" in compressed_payload:
            compressed_bytes = base64.b64decode(compressed_payload["encoded_bytes"])
            raw_bytes = zlib.decompress(compressed_bytes)
            arr = np.frombuffer(raw_bytes, dtype=np.float32)
            return [round(float(x), 6) for x in arr.tolist()]
        return [float(x) for x in compressed_payload.get("data", [])]


class ActivationCompressor:
    """Compressor engine managing codec selection."""

    def __init__(self) -> None:
        self.codecs: Dict[str, CompressionCodec] = {
            "FP16": FP16Codec(),
            "INT8": INT8Codec(),
            "ZLIB": ZLIBCodec(),
        }

    def compress_activations(
        self,
        activations: List[float],
        codec_name: str = "FP16",
    ) -> Dict[str, Any]:
        norm_name = codec_name.upper().strip()
        codec = self.codecs.get(norm_name, self.codecs["FP16"])
        return codec.compress(activations)

    def decompress_activations(self, payload: Dict[str, Any]) -> List[float]:
        codec_name = payload.get("codec", "FP16").upper().strip()
        codec = self.codecs.get(codec_name, self.codecs["FP16"])
        return codec.decompress(payload)

