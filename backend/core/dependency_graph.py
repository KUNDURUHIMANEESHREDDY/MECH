"""Research Dependency Graph Engine."""

from __future__ import annotations

from typing import Any, Dict, List


class ResearchDependencyGraph:
    """Tracks dependency chains across research packages, datasets, and workflows."""

    def __init__(self) -> None:
        self.dependencies: Dict[str, List[str]] = {
            "pkg_ioi_circuit": ["ds_ioi_prompts", "pkg_sae_features"],
            "pkg_sae_features": ["mdl_gpt2_weights"],
            "pkg_publication_paper": ["pkg_ioi_circuit", "bench_ioi_suite"],
        }

    def get_dependencies(self, package_id: str) -> Dict[str, Any]:
        deps = self.dependencies.get(package_id, [])
        return {"package_id": package_id, "direct_dependencies": deps, "total_dependencies_count": len(deps)}
