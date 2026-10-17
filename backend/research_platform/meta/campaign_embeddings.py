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
    """Vector index over campaign summaries.

    It does not perform semantic retrieval: the vectors are hashes (see
    `_compute_mock_vector`), so `find_similar_campaigns` returns a stable
    ordering, not a relevance measure.
    """

    def __init__(self) -> None:
        # Empty, not seeded. A `camp_s6_ioi` entry with a fixed vector and an
        # invented IOI summary used to be installed here, and
        # `MetaResearchEngine.run_closed_feedback_loop` defaults its
        # `campaign_id` to that same id -- so a replay of a campaign that never
        # ran found a record waiting for it.
        self.index: Dict[str, CampaignVectorEntry] = {}

    def _compute_mock_vector(self, text: str) -> List[float]:
        """A deterministic 8-dimensional vector derived from a character sum.

        This is a hash, not an embedding. Nothing was trained and no text
        encoder was run: two texts that mean the same thing produce unrelated
        vectors, and two unrelated texts that share a character sum produce
        near-identical ones. Cosine similarity between these vectors is
        therefore not semantic similarity, and `find_similar_campaigns` marks
        its scores accordingly rather than presenting them as retrieval
        quality.
        """
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
                # Kept so the ordering stays reproducible, but it is a
                # similarity between hash vectors, not between meanings.
                "similarity_score": similarity,
                "similarity_is_semantic": False,
                "source": "mock_hash_vector",
                "measured": False,
                "validation_eligible": False,
                "publication_eligible": False,
                "reason": (
                    "Cosine similarity over deterministic hash vectors "
                    "(_compute_mock_vector). No embedding model was run, so "
                    "this is not semantic similarity and not evidence."
                ),
            })

        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]
