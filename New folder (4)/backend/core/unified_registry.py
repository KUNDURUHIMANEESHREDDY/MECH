"""Unified Registry Orchestrator.

Manages individual entity registries (Papers, Mechanisms, Circuits, etc.) 
under one unified interface to prevent registry bloat and centralize data fetching.
"""

from typing import Any, Dict, List, Optional

class PaperRegistry:
    def list_items(self) -> List[Dict[str, Any]]:
        return [{"id": "ioi", "title": "Indirect Object Identification"}]

class MechanismRegistry:
    def list_items(self) -> List[Dict[str, Any]]:
        return [{"id": "name_mover", "title": "Name Mover Head"}]

class CircuitRegistry:
    def list_items(self) -> List[Dict[str, Any]]:
        return [{"id": "ioi_circuit", "title": "IOI Circuit"}]

class ExperimentRegistry:
    def list_items(self) -> List[Dict[str, Any]]:
        return [{"id": "exp_1", "title": "Activation Patching L9H9"}]

class DiscoveryRegistry:
    def list_items(self) -> List[Dict[str, Any]]:
        return [{"id": "disc_1", "title": "S2 Inhibition"}]

class DatasetRegistry:
    def list_items(self) -> List[Dict[str, Any]]:
        return [{"id": "openwebtext", "title": "OpenWebText Chunk"}]

class ReportRegistry:
    def list_items(self) -> List[Dict[str, Any]]:
        return [{"id": "report_1", "title": "IOI Reproduction Report"}]

class SkillRegistry:
    def list_items(self) -> List[Dict[str, Any]]:
        return [{"id": "path_patching", "title": "Path Patching"}]

class UnifiedRegistry:
    """Orchestrator for all specialized entity registries."""
    
    def __init__(self) -> None:
        self.papers = PaperRegistry()
        self.mechanisms = MechanismRegistry()
        self.circuits = CircuitRegistry()
        self.experiments = ExperimentRegistry()
        self.discoveries = DiscoveryRegistry()
        self.datasets = DatasetRegistry()
        self.reports = ReportRegistry()
        self.skills = SkillRegistry()

    def list_catalog(self, item_type: str = "all") -> List[Dict[str, Any]]:
        """Return items from the corresponding sub-registry."""
        if item_type == "papers":
            return self.papers.list_items()
        elif item_type == "mechanisms":
            return self.mechanisms.list_items()
        elif item_type == "circuits":
            return self.circuits.list_items()
        elif item_type == "experiments":
            return self.experiments.list_items()
        elif item_type == "discoveries":
            return self.discoveries.list_items()
        elif item_type == "datasets":
            return self.datasets.list_items()
        elif item_type == "reports":
            return self.reports.list_items()
        elif item_type == "skills":
            return self.skills.list_items()
        elif item_type == "all":
            return [
                *self.papers.list_items(),
                *self.mechanisms.list_items(),
                *self.circuits.list_items(),
            ]
        return []
