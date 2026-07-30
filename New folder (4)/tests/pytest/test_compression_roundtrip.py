"""Integration Test: Activation Compression Round-Trip.

Validates: Activations -> Compress -> Decompress -> Verify Numerical Integrity.
"""
from api.dispatcher import build_dispatcher


def test_compression_roundtrip():
    dispatcher = build_dispatcher()
    original_activations = [1.25, 4.5, 8.75, 12.0]

    # 1. FP16 Round-Trip
    comp_fp16 = dispatcher["runtime/memory/compress"]({"activations": original_activations, "codec": "FP16"})
    assert comp_fp16["codec"] == "FP16"
    decomp_fp16 = dispatcher["runtime/memory/decompress"](comp_fp16)
    assert decomp_fp16 == original_activations

    # 2. INT8 Round-Trip
    comp_int8 = dispatcher["runtime/memory/compress"]({"activations": original_activations, "codec": "INT8"})
    assert comp_int8["codec"] == "INT8"
    decomp_int8 = dispatcher["runtime/memory/decompress"](comp_int8)
    assert len(decomp_int8) == len(original_activations)
