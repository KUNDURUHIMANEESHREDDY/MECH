"""Representation Atlas Engine — Navigable Knowledge Hierarchies.

Provides a structured, hierarchical data view of discovered semantic concepts
to power the Representation Atlas UI.

Structure:
Knowledge Domain (e.g., Geography)
└── Concept Family (e.g., Capitals)
    └── Concept (e.g., Paris)
        └── [Constituent Features, Circuits, Activation Examples]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .representation_engine import RepresentationEngine, Concept


@dataclass
class AtlasDomain:
    name: str
    families: List[AtlasFamily] = field(default_factory=list)


@dataclass
class AtlasFamily:
    name: str
    concepts: List[Concept] = field(default_factory=list)


class RepresentationAtlasEngine:
    """Builds the hierarchical Representation Atlas data structure."""

    def __init__(self, repr_engine: Optional[RepresentationEngine] = None) -> None:
        self.repr_engine = repr_engine or RepresentationEngine()

    def get_atlas(self) -> List[AtlasDomain]:
        """Assembles the full Knowledge Atlas from discovered concepts."""
        concepts = self.repr_engine.list_concepts()

        # In a real system, we'd use the Knowledge Graph or metadata to group these
        domains: Dict[str, AtlasDomain] = {}

        # Simple heuristic grouping for MVP
        for c in concepts:
            # Domain mapping
            domain_name = c.metadata.get("domain", "General")
            family_name = c.metadata.get("family", "Uncategorized")

            if domain_name not in domains:
                domains[domain_name] = AtlasDomain(name=domain_name)

            domain = domains[domain_name]

            # Family mapping
            family = next((f for f in domain.families if f.name == family_name), None)
            if not family:
                family = AtlasFamily(name=family_name)
                domain.families.append(family)

            family.concepts.append(c)

        return list(domains.values())

    def get_domain_view(self, domain_name: str) -> Dict[str, Any]:
        """Returns a deep view of a specific knowledge domain."""
        atlas = self.get_atlas()
        domain = next((d for d in atlas if d.name.lower() == domain_name.lower()), None)

        if not domain:
            return {"error": f"Domain '{domain_name}' not found."}

        from dataclasses import asdict
        return asdict(domain)
