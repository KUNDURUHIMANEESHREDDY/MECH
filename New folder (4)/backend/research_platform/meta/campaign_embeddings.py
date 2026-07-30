"""Campaign Vector Embeddings & Similarity Search.

Generates dense vector embeddings for research campaign summaries and enables similarity retrieval.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class CampaignVectorEntry:
    """Dataclass storing campaign vector embeddings and metadata."""

    campaign_id: str
    topic: str
    summary_text: str
    vector: List[float]
    indexed_at: str


class CampaignEmbeddingsEngine:
    """Vector database index for campaign summaries allowing fast semantic retrieval."""

    def __init__(self) -> None:
        init_entry = CampaignVectorEntry(
            campaign_id="camp_s6_ioi",
            topic="Indirect Object Identification Circuit Discovery",
            summary_text="Investigated IOI circuit in GPT-2 Small using SAE L8 and causal path patching.",
            vector=[0.12, 0.85, 0.44, 0.92, 0.31, 0.78, 0.15, 0.64],
            indexed_at=_dt.datetime.utcnow().isoformat() + "Z",
        )
        self.index: Dict[str, CampaignVectorEntry] = {init_entry.campaign_id: init_entry}

    def _compute_mock_vector(self, text: str) -> List[float]:
        """Generates a deterministic 8-dimensional embedding from text."""
        seed = sum(ord(c) for c in text)
        return [round(math.sin(seed * (i + 1)) * 0.5 + 0.5, 4) for i in range(8)]

    def index_campaign(self, campaign_id: str, topic: str, summary_text: str) -> Dict[str, Any]:
        vector = self._compute_mock_vector(topic + " " + summary_text)
        entry = CampaignVectorEntry(
            campaign_id=campaign_id,
            topic=topic,
            summary_text=summary_text,
            vector=vector,
            indexed_at=_dt.datetime.utcnow().isoformat() + "Z",
        )
        self.index[campaign_id] = entry
        return asdict(entry)

    def find_similar_campaigns(self, query_topic: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_vec = self._compute_mock_vector(query_topic)
        results = []

        for entry in self.index.values():
            dot = sum(a * b for a, b in zip(query_vec, entry.vector))
            norm_q = math.sqrt(sum(a * a for a in query_vec)) or 1.0
            norm_e = math.sqrt(sum(b * b for b in entry.vector)) or 1.0
            similarity = round(dot / (norm_q * norm_e), 4)

            results.append({
                "campaign_id": entry.campaign_id,
                "topic": entry.topic,
                "summary_text": entry.summary_text,
                "similarity_score": similarity,
            })

        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]
