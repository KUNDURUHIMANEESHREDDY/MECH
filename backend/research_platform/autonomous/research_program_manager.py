"""Long-Term Research Program Manager."""

from __future__ import annotations

from backend.core.identifiers import content_id

import datetime as _dt
from typing import Any, Dict, List


class LongTermResearchProgramManager:
    """Manages multi-week research programs (Goal ➔ Projects ➔ Experiments ➔ Publications)."""

    def create_program(self, program_title: str) -> Dict[str, Any]:
        return {
            "program_id": content_id(program_title, prefix="prog_"),
            "title": program_title,
            "projects": ["IOI Circuit Discovery", "Polysemantic SAE Analysis", "Cross-Family Gemma Alignment"],
            "created_at": _dt.datetime.utcnow().isoformat() + "Z",
            "status": "Active",
        }
