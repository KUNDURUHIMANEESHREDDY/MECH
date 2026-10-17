"""Immutable Execution Episodes Engine with Vector Search.

Encapsulates complete end-to-end execution trajectories into immutable episode objects
and generates dense vector embeddings for similarity search across past runs.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class ExecutionEpisode:
    episode_id: str
    campaign_id: str
    model_name: str
    plan_summary: Dict[str, Any]
    simulation_output: Dict[str, Any]
    decision_record: Dict[str, Any]
    execution_telemetry: Dict[str, Any]
    diagnostics_record: Dict[str, Any] | None
    outcome_status: str  # "Success", "Recovered", "Failed"
    total_duration_sec: float
    total_cost_usd: float
    vector: List[float]
    recorded_at: str


class ExecutionEpisodesEngine:
    """Store for immutable execution episode records with vector similarity search."""

    def __init__(self) -> None:
        # Empty by construction.
        #
        # This used to seed one complete episode for every engine, before any
        # job had been recorded: "ep_ioi_001", outcome "Success", throughput
        # 1420.5 tok/s, GPU utilisation 94.2%, duration 51.2 s, cost $0.12, and
        # a simulation confidence of 94.0%. No job ran, so every one of those
        # values was a literal. An episode store that already contains a
        # successful run is the worst default available here, because a reader
        # cannot tell it apart from a run that was actually recorded.
        self.episodes: Dict[str, ExecutionEpisode] = {}

    def _compute_vector(self, model_name: str, outcome_status: str, duration: float) -> List[float]:
        text = f"{model_name}_{outcome_status}_{duration}"
        seed = sum(ord(c) for c in text)
        return [round(math.sin(seed * (i + 1)) * 0.5 + 0.5, 4) for i in range(8)]

    def record_episode(
        self,
        campaign_id: str,
        model_name: str,
        plan_summary: Dict[str, Any],
        simulation_output: Dict[str, Any],
        decision_record: Dict[str, Any],
        execution_telemetry: Dict[str, Any],
        diagnostics_record: Dict[str, Any] | None = None,
        outcome_status: str = "Success",
        total_duration_sec: float = 45.0,
        total_cost_usd: float = 0.10,
    ) -> Dict[str, Any]:
        ep_id = f"ep_{model_name.lower().replace(' ', '_')}_{len(self.episodes) + 1}"
        vector = self._compute_vector(model_name, outcome_status, total_duration_sec)

        ep = ExecutionEpisode(
            episode_id=ep_id,
            campaign_id=campaign_id,
            model_name=model_name,
            plan_summary=plan_summary,
            simulation_output=simulation_output,
            decision_record=decision_record,
            execution_telemetry=execution_telemetry,
            diagnostics_record=diagnostics_record,
            outcome_status=outcome_status,
            total_duration_sec=total_duration_sec,
            total_cost_usd=total_cost_usd,
            vector=vector,
            recorded_at=_dt.datetime.utcnow().isoformat() + "Z",
        )
        self.episodes[ep_id] = ep
        return asdict(ep)

    def find_similar_episodes(self, query_model: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_vec = self._compute_vector(query_model, "Success", 50.0)
        results = []

        for ep in self.episodes.values():
            dot = sum(a * b for a, b in zip(query_vec, ep.vector))
            norm_q = math.sqrt(sum(a * a for a in query_vec)) or 1.0
            norm_e = math.sqrt(sum(b * b for b in ep.vector)) or 1.0
            similarity = round(dot / (norm_q * norm_e), 4)

            results.append({
                "episode_id": ep.episode_id,
                "campaign_id": ep.campaign_id,
                "model_name": ep.model_name,
                "outcome_status": ep.outcome_status,
                "total_duration_sec": ep.total_duration_sec,
                "similarity_score": similarity,
            })

        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]

    def get_episode(self, episode_id: str) -> Dict[str, Any] | None:
        ep = self.episodes.get(episode_id)
        return asdict(ep) if ep else None

    def list_episodes(self) -> List[Dict[str, Any]]:
        return [asdict(e) for e in self.episodes.values()]
