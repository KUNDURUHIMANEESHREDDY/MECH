import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable

@dataclass
class Requirement:
    name: str
    weight: float
    is_hard: bool = True
    dependencies: List[str] = field(default_factory=list)
    description: str = ""
    evidence_file: Optional[str] = None
    evidence_check: Optional[Callable[[Any], bool]] = None

@dataclass
class CompletionCriteria:
    """
    Advanced Completion Criteria with weights, hard/soft requirements, dependencies,
    and automatic evidence collection.
    """
    requirements: List[Requirement]
    satisfied: List[str] = field(default_factory=list)

    def collect_evidence(self, project_dir: str):
        """
        Automatically scans the project directory for evidence files and validates them.
        """
        for req in self.requirements:
            if req.evidence_file:
                file_path = os.path.join(project_dir, req.evidence_file)
                if os.path.exists(file_path):
                    try:
                        if file_path.endswith('.json'):
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                        else:
                            with open(file_path, 'r') as f:
                                data = f.read()
                        
                        # Apply custom check if provided, else existence is enough
                        if req.evidence_check:
                            if req.evidence_check(data):
                                self.mark_satisfied(req.name)
                        else:
                            self.mark_satisfied(req.name)
                    except Exception as e:
                        print(f"Error validating evidence for {req.name}: {e}")

    def mark_satisfied(self, name: str):
        req_names = [r.name for r in self.requirements]
        if name in req_names and name not in self.satisfied:
            req = next(r for r in self.requirements if r.name == name)
            for dep in req.dependencies:
                if dep not in self.satisfied:
                    return
            self.satisfied.append(name)

    def calculate_score(self) -> float:
        total_weight = sum(r.weight for r in self.requirements)
        satisfied_weight = sum(r.weight for r in self.requirements if r.name in self.satisfied)
        if total_weight == 0: return 0.0
        return (satisfied_weight / total_weight) * 100.0

    def is_complete(self) -> bool:
        hard_reqs = [r.name for r in self.requirements if r.is_hard]
        return all(req in self.satisfied for req in hard_reqs)

    def status_report(self) -> Dict[str, Any]:
        return {
            "completion_score": round(self.calculate_score(), 1),
            "is_complete": self.is_complete(),
            "satisfied": self.satisfied,
            "missing": [r.name for r in self.requirements if r.name not in self.satisfied],
            "dependency_graph": {r.name: r.dependencies for r in self.requirements}
        }
