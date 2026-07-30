"""Workload Fingerprints & Similarity Retrieval Engine.

Encodes hardware workload specifications (model, sequence_length, batch_size, gpu, precision)
into vector embeddings to find similar historical executions.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class WorkloadFingerprint:
    """Dataclass storing workload fingerprint metadata and vector embedding."""

    fingerprint_id: str
    model_name: str
    sequence_length: int
    batch_size: int
    gpu_type: str
    precision: str
    vector: List[float]
    indexed_at: str


class WorkloadFingerprintsEngine:
    """Vector database index for workload fingerprints allowing historical execution retrieval."""

    def __init__(self) -> None:
        init_fp = WorkloadFingerprint(
            fingerprint_id="fp_gpt2_h100",
            model_name="GPT-2 Small",
            sequence_length=2048,
            batch_size=16,
            gpu_type="H100",
            precision="FP16",
            vector=[0.24, 0.76, 0.51, 0.88, 0.12, 0.65, 0.43, 0.90],
            indexed_at=_dt.datetime.utcnow().isoformat() + "Z",
        )
        self.index: Dict[str, WorkloadFingerprint] = {init_fp.fingerprint_id: init_fp}

    def _compute_vector(self, model_name: str, seq_len: int, batch_size: int, gpu: str, precision: str) -> List[float]:
        text = f"{model_name}_{seq_len}_{batch_size}_{gpu}_{precision}"
        seed = sum(ord(c) for c in text)
        return [round(math.sin(seed * (i + 1)) * 0.5 + 0.5, 4) for i in range(8)]

    def create_fingerprint(
        self,
        model_name: str,
        sequence_length: int,
        batch_size: int,
        gpu_type: str = "H100",
        precision: str = "FP16",
    ) -> Dict[str, Any]:
        fp_id = f"fp_{model_name.lower().replace(' ', '_')}_{sequence_length}_{batch_size}"
        vector = self._compute_vector(model_name, sequence_length, batch_size, gpu_type, precision)

        fp = WorkloadFingerprint(
            fingerprint_id=fp_id,
            model_name=model_name,
            sequence_length=sequence_length,
            batch_size=batch_size,
            gpu_type=gpu_type,
            precision=precision,
            vector=vector,
            indexed_at=_dt.datetime.utcnow().isoformat() + "Z",
        )
        self.index[fp_id] = fp
        return asdict(fp)

    def find_similar_workloads(
        self,
        model_name: str,
        sequence_length: int,
        batch_size: int,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        query_vec = self._compute_vector(model_name, sequence_length, batch_size, "H100", "FP16")
        results = []

        for fp in self.index.values():
            dot = sum(a * b for a, b in zip(query_vec, fp.vector))
            norm_q = math.sqrt(sum(a * a for a in query_vec)) or 1.0
            norm_f = math.sqrt(sum(b * b for b in fp.vector)) or 1.0
            similarity = round(dot / (norm_q * norm_f), 4)

            results.append({
                "fingerprint_id": fp.fingerprint_id,
                "model_name": fp.model_name,
                "sequence_length": fp.sequence_length,
                "batch_size": fp.batch_size,
                "gpu_type": fp.gpu_type,
                "precision": fp.precision,
                "similarity_score": similarity,
            })

        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]
