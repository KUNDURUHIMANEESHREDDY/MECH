"""Concept Evolution Engine — Lineage across Layers & Models.

Analyzes how semantic representations relate through the network (layer-wise)
and across model families.

Lineage categories, decided by comparing the two concept sets:
- ``shared``    the concepts appear in both
- ``novel``     the concept appears only in the target
- ``extinct``   the concept appears only in the source
- ``ancestor``  a source concept maps onto a target concept with the same name
- ``aligned``   a source concept maps onto a differently-named target concept
- ``split``     one source concept maps onto several target concepts
- ``merged``    several source concepts map onto one target concept

What this module used to do
---------------------------
`analyze_layer_progression` looped over adjacent layers and emitted the same
edge regardless of input::

    LineageEdge(f"Layer {l1}", f"Layer {l2}", "Names", "Cities", "ancestor", 0.72)

The concept names were literals. Passing `{"0": ["Python"], "1": ["Basketball"]}`
produced an edge asserting that "Names" is an ancestor of "Cities" at alignment
0.72 -- a specific claim about specific concepts, derived from nothing.

`analyze_cross_model_drift` returned constants (``drift_score: 0.15``,
``persistence: 0.85``, ``novel_concepts: ["PyTorch Syntax"]``) and
`compute_concept_persistence` returned ``0.94`` for any input.

What it does now
----------------
Concepts are compared as sets and edges are derived from the actual overlap. An
alignment score is a measured quantity -- currently Jaccard overlap of the token
sets, the "high overlap in activating tokens = lineage" heuristic the original
comment described but never implemented.

Where an alignment cannot be computed, the score is None with the reason, rather
than the constant 0.72 that made every edge look equally well-supported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set


@dataclass
class LineageEdge:
    source_model: str
    target_model: str
    source_concept: str
    target_concept: str
    type: str                  # shared, split, merged, novel, ancestor, aligned
    #: Optional. An edge asserting a relationship without being able to score
    #: the overlap carries None, not a placeholder that reads as a measurement.
    alignment_score: Optional[float] = None
    #: How many source concepts this edge stands for. >1 marks a merge.
    source_count: int = 1
    #: How many target concepts. >1 marks a split.
    target_count: int = 1
    #: Why the score is None, when it is.
    score_reason: str = ""
    #: Whether the score is a measured overlap.
    measured: bool = False


def _as_concepts(value: Any) -> Optional[Set[str]]:
    """Coerce a caller-supplied concept collection into a set of names.

    Accepts a list of names, a list of dicts with a `name`/`concept`/`semantic_name`
    key, or a space/comma-separated string. Returns None when nothing usable can
    be read, which is different from returning an empty set: an empty set says
    "this layer has no concepts", None says "I could not tell".
    """
    if value is None:
        return None
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace(",", " ").split() if p.strip()]
        return set(parts) if parts else None
    if isinstance(value, dict):
        # Descend into a recognised container key first. Treating the dict's own
        # keys as concept names meant `{"concepts": ["Python", "Jazz"]}` reported
        # a concept literally called "concepts" -- a plausible-looking answer
        # derived from the shape of the input rather than its content.
        for key in ("concepts", "concept_names", "semantic_concepts",
                    "detected_concepts", "names", "items"):
            if key in value:
                return _as_concepts(value[key])
        return _as_concepts(list(value.keys()))
    if isinstance(value, (list, tuple, set)):
        names: Set[str] = set()
        for item in value:
            if isinstance(item, str):
                if item.strip():
                    names.add(item.strip())
            elif isinstance(item, dict):
                for key in ("semantic_name", "name", "concept", "label"):
                    candidate = item.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        names.add(candidate.strip())
                        break
        return names or None
    return None


def _jaccard(a: Set[str], b: Set[str]) -> Optional[float]:
    """Overlap of two concept sets, or None when it cannot be computed.

    None when either set is empty: Jaccard of two empty sets is 0/0, and
    reporting that as 0.0 (perfectly dissimilar) or 1.0 (identical) would both
    be inventions.
    """
    if not a or not b:
        return None
    union = a | b
    return len(a & b) / len(union) if union else None


class ConceptEvolutionEngine:
    """Tracks relationships between concept sets across layers and models."""

    def analyze_layer_progression(
        self,
        layer_concepts: Dict[int, List[str]],
    ) -> List[LineageEdge]:
        """Relate the concepts of each layer to the next.

        One edge per concept relationship between adjacent layers. Nothing is
        emitted for a pair where neither concept set could be read, because that
        pair yields no relationship to assert.
        """
        edges: List[LineageEdge] = []
        layers = sorted(k for k in layer_concepts.keys())
        if len(layers) < 2:
            return edges

        for left, right in zip(layers, layers[1:]):
            source = _as_concepts(layer_concepts[left])
            target = _as_concepts(layer_concepts[right])
            if source is None or target is None:
                edges.append(LineageEdge(
                    source_model=f"Layer {left}",
                    target_model=f"Layer {right}",
                    source_concept="<unreadable>",
                    target_concept="<unreadable>",
                    type="unknown",
                    alignment_score=None,
                    score_reason=(
                        "Could not read a concept set for one of these layers, "
                        "so no relationship can be asserted."
                    ),
                ))
                continue

            shared = sorted(source & target)
            for name in shared:
                score = _jaccard(source, target)
                edges.append(LineageEdge(
                    source_model=f"Layer {left}",
                    target_model=f"Layer {right}",
                    source_concept=name,
                    target_concept=name,
                    type="shared",
                    alignment_score=(None if score is None else round(score, 4)),
                    score_reason="" if score is not None
                    else "Jaccard is undefined for an empty concept set.",
                    measured=score is not None,
                ))

            for name in sorted(source - target):
                edges.append(LineageEdge(
                    source_model=f"Layer {left}",
                    target_model=f"Layer {right}",
                    source_concept=name,
                    target_concept="<absent>",
                    type="extinct",
                    alignment_score=0.0,
                    measured=True,
                ))

            for name in sorted(target - source):
                edges.append(LineageEdge(
                    source_model=f"Layer {left}",
                    target_model=f"Layer {right}",
                    source_concept="<absent>",
                    target_concept=name,
                    type="novel",
                    alignment_score=0.0,
                    measured=True,
                ))

        return edges

    def analyze_cross_model_drift(
        self,
        model_a_results: Any,
        model_b_results: Any,
    ) -> Dict[str, Any]:
        """Compare concept sets between two models.

        Everything reported is derived from the two inputs. Previously this
        returned `drift_score: 0.15`, `persistence: 0.85`,
        `novel_concepts: ["PyTorch Syntax"]`, `shared_concepts: ["Geography",
        "Logic"]` and a hardcoded split of Programming into Python/JavaScript --
        for any arguments whatsoever.
        """
        a = _as_concepts(model_a_results)
        b = _as_concepts(model_b_results)

        if a is None or b is None:
            reason = (
                "Could not read a concept set from one of the two models, so no "
                "drift figure can be computed."
            )
            return {
                "measured": False,
                "drift_score": None,
                "persistence": None,
                "shared_concepts": None,
                "novel_concepts": None,
                "extinct_concepts": None,
                "n_a": None,
                "n_b": None,
                "reason": reason,
            }

        shared = sorted(a & b)
        novel = sorted(b - a)
        extinct = sorted(a - b)
        union = a | b

        return {
            "measured": True,
            # Share of the union that is not shared. 1.0 means nothing in common.
            "drift_score": round(len(a ^ b) / len(union), 4) if union else None,
            # Share of model A's concepts that also appear in model B.
            "persistence": round(len(a & b) / len(a), 4) if a else None,
            "shared_concepts": shared,
            "novel_concepts": novel,
            "extinct_concepts": extinct,
            "n_a": len(a),
            "n_b": len(b),
            "reason": None,
        }

    def compute_concept_persistence(
        self,
        concept_id: str,
        model_timeline: Sequence[Any],
    ) -> Dict[str, Any]:
        """How long a concept survives across a sequence of models.

        `model_timeline` is an ordered sequence whose entries indicate whether
        the concept is present: a model name, a bool, or a dict with a `present`
        / `concepts` field. The score is the fraction of models in which it
        appears.

        Returned as a dict, not a float: the previous signature returned `0.94`
        for any input, and a bare float has nowhere to say so.
        """
        if not model_timeline:
            return {
                "concept_id": concept_id,
                "measured": False,
                "persistence": None,
                "n_models": 0,
                "n_present": 0,
                "reason": (
                    "No model timeline supplied, so persistence cannot be "
                    "computed. Previously this returned 0.94 for every input."
                ),
            }

        present_flags: List[bool] = []
        unreadable = 0
        for entry in model_timeline:
            if isinstance(entry, bool):
                present_flags.append(entry)
            elif isinstance(entry, dict):
                if "present" in entry:
                    present_flags.append(bool(entry["present"]))
                else:
                    concepts = _as_concepts(entry.get("concepts"))
                    if concepts is None:
                        unreadable += 1
                    else:
                        present_flags.append(concept_id in concepts)
            elif isinstance(entry, str):
                concepts = _as_concepts(entry)
                if concepts is None:
                    unreadable += 1
                else:
                    present_flags.append(concept_id in concepts)
            else:
                unreadable += 1

        if not present_flags:
            return {
                "concept_id": concept_id,
                "measured": False,
                "persistence": None,
                "n_models": len(model_timeline),
                "n_present": None,
                "reason": (
                    f"None of the {len(model_timeline)} timeline entries could be "
                    f"interpreted as presence or absence of {concept_id!r}."
                ),
            }

        n_present = sum(present_flags)
        return {
            "concept_id": concept_id,
            "measured": True,
            "persistence": round(n_present / len(present_flags), 4),
            "n_models": len(present_flags),
            "n_present": n_present,
            "n_unreadable": unreadable,
            "reason": None,
        }