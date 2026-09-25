"""Report Generation Service.

Generates structured experiment summaries in Markdown and HTML formats.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict


class ReportService:
    """Create a transport report without inventing scientific findings."""

    def generate_report(
        self,
        experiment_id: str,
        title: str = "Experiment Report",
        provenance: str = "unavailable",
    ) -> Dict[str, Any]:
        now = _dt.datetime.utcnow().isoformat() + "Z"
        markdown = f"""# {title}

**Experiment ID:** {experiment_id}
**Generated At:** {now}
**Evidence provenance:** {provenance}

## Evidence boundary
- This document is a transport summary for the supplied run record.
- No layer, intervention, probability, or mechanistic claim is synthesized here.
- Consult the run's explicitly live evidence payload for measurements.
"""
        html = f"<html><body><pre>{markdown}</pre></body></html>"
        return {
            "experiment_id": experiment_id,
            "title": title,
            "provenance": provenance,
            "field_provenance": {
                "experiment_id": provenance,
                "title": provenance,
                "markdown": provenance,
                "html": provenance,
                "generated_at": provenance,
            },
            "markdown": markdown,
            "html": html,
            "generated_at": now,
        }
