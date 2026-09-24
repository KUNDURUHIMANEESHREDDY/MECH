"""Society Scribe agent (capability: publish).

Turns a run trace into durable artifacts. Wraps:
- backend.services.report_service.ReportService
  .generate_report(experiment_id, title) -> {markdown, html, ...}
- backend.core.evidence_graph.TraceableEvidenceGraph
  .to_dict() / .get_provenance_trace(target_id)
- backend.core.unified_registry.UnifiedRegistry (report/paper catalog reads)

Legacy stubs replaced: _mechanistic_reports ("IOI Circuit Report..."),
_dashboard_summary ({goals:3, discoveries:14}), _evidence_rank,
_confidence_score — all hardcoded literals, now built from real outputs.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List


class Scribe:
    """Publishes mechanistic reports from run traces."""

    capability = "publish"

    def evidence(self) -> Dict[str, Any]:
        try:
            from backend.core.evidence_graph import TraceableEvidenceGraph
            return {"status": "completed",
                    "result": TraceableEvidenceGraph().to_dict()}
        except Exception as exc:
            return {"status": "error", "error": str(exc)[:500]}

    def report(self, experiment_id: str, title: str) -> Dict[str, Any]:
        try:
            from backend.services.report_service import ReportService
            res = ReportService().generate_report(
                experiment_id=experiment_id, title=title)
            return {"status": "completed", "result": res}
        except Exception as exc:
            return {"status": "error", "error": str(exc)[:500]}

    def publish(self, goal: str, trace: List[Dict[str, Any]],
                reflection: Dict[str, Any]) -> Dict[str, Any]:
        """Assemble the final deliverable: report + evidence + reflection."""
        exp_id = f"exp_{abs(hash(goal)) % 10000:04d}"
        rep = self.report(experiment_id=exp_id,
                          title=f"Mechanistic Report: {goal[:60]}")
        ev = self.evidence()
        ok_steps = sum(1 for t in trace
                       if t.get("status") in ("completed", "ok", "loaded"))
        return {
            "status": "completed",
            "goal": goal,
            "experiment_id": exp_id,
            "steps_completed": f"{ok_steps}/{len(trace)}",
            "report": rep.get("result", rep),
            "evidence_graph": ev.get("result", ev),
            "reflection": reflection,
            "published_at": _dt.datetime.utcnow().isoformat() + "Z",
        }
