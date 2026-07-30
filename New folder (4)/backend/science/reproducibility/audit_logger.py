"""Scientific Audit Logger — Step-by-Step Validation Traceability.

Records every check, warning, and decision made during the validation process
to ensure complete transparency.
"""

from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AuditEntry:
    timestamp: str
    component: str
    action: str
    status: str                # INFO, PASS, WARN, FAIL
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ScientificAuditLogger:
    """Session-based logger for validation audit trails."""

    def __init__(self) -> None:
        self.entries: List[AuditEntry] = []

    def log(self, component: str, action: str, status: str, message: str, **kwargs) -> None:
        """Records a validation event."""
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        entry = AuditEntry(
            timestamp=timestamp,
            component=component,
            action=action,
            status=status,
            message=message,
            metadata=kwargs
        )
        self.entries.append(entry)

    def get_entries(self) -> List[Dict[str, Any]]:
        return [asdict(e) for e in self.entries]

    def reset(self) -> None:
        self.entries = []
