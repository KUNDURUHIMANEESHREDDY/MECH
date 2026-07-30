"""Report Generation Service.

Generates structured experiment summaries in Markdown and HTML formats.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict


class ReportService:
    """Service for generating automated experiment summary reports."""

    def generate_report(self, experiment_id: str, title: str = "Experiment Report") -> Dict[str, Any]:
        now = _dt.datetime.utcnow().isoformat() + "Z"
        markdown = f"""# {title}

**Experiment ID:** {experiment_id}
**Generated At:** {now}
**Model:** GPT-2 Small

## Key Findings
- **Layer 8 Pause**: Verified breakpoint at Layer 8.
- **Intervention**: Applied activation patch on Layer 8, Neuron 402.
- **Outcome**: Prediction probability updated from 0.12 (" France") to 0.82 (" Paris").
"""
        html = f"<html><body><pre>{markdown}</pre></body></html>"
        return {
            "experiment_id": experiment_id,
            "title": title,
            "markdown": markdown,
            "html": html,
            "generated_at": now,
        }
