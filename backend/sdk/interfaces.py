"""Public Plugin SDK Interfaces."""

from __future__ import annotations

from typing import Any, Dict, List


class BasePlugin:
    """Base interface for all platform plugins."""

    def __init__(self, plugin_id: str, name: str, version: str) -> None:
        self.plugin_id = plugin_id
        self.name = name
        self.version = version


class AlgorithmPlugin(BasePlugin):
    """Interface for custom interpretability algorithm plugins."""

    def execute_algorithm(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class PanelPlugin(BasePlugin):
    """Interface for custom visualization panel plugins."""

    def render_spec(self) -> Dict[str, Any]:
        return {"component": self.name, "layout": "flex"}


class ReportPlugin(BasePlugin):
    """Interface for custom report template plugins."""

    def generate_report(self, data: Dict[str, Any]) -> str:
        raise NotImplementedError


class ExporterPlugin(BasePlugin):
    """Interface for custom exporter plugins."""

    def export(self, payload: Dict[str, Any]) -> bytes:
        raise NotImplementedError


class WorkflowPlugin(BasePlugin):
    """Interface for custom workflow stage plugins."""

    def execute_stage(self, stage_id: str) -> Dict[str, Any]:
        return {"status": "executed"}


class PipelineNodePlugin(BasePlugin):
    """Interface for custom pipeline node plugins."""

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return data


class DatasetPlugin(BasePlugin):
    """Interface for custom dataset provider plugins."""

    def fetch_samples(self, limit: int = 10) -> List[Dict[str, Any]]:
        return []


class CommandPlugin(BasePlugin):
    """Interface for custom command palette action plugins."""

    def trigger(self) -> Dict[str, Any]:
        return {"status": "command_triggered"}
