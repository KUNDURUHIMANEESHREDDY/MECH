"""RAG Integration Layer for MECH.

Bridges external literature retrieval (Context7, GitHub, arXiv) with MECH's
internal research graph, knowledge base, and experiment infrastructure.

Provides:
- Hybrid search (semantic + keyword) across external papers + internal results
- Paper metadata model with claims, methods, discoveries
- Hypothesis enrichment with citation augmentation
- Indexing pipeline from external sources
- Integration with KnowledgeBaseEngine and TypedResearchGraph
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set

from backend.research_platform.autonomous.knowledge_base import KnowledgeBaseEngine
from backend.research_platform.autonomous.research_graph import TypedResearchGraph


@dataclass
class PaperMetadata:
    """Structured metadata for a research paper in the MECH RAG index."""

    paper_id: str
    title: str
    authors: List[str]
    venue: str  # conference, journal, or arXiv
    year: int
    doi: str
    claims: List[str]  # mechanistic claims made in the paper
    methods: List[str]  # methodology/techniques used
    discoveries: List[str]  # key discoveries reported
    results_summary: str  # brief summary of results
    confidence_score: float  # platform-wide confidence in these assertions
    citation_count: int  # known citation count
    source: str  # "context7", "github", "arxiv", "manual"
    indexed_at: str  # ISO timestamp
    abstract: str = ""  # full abstract if available


@dataclass
class SearchResult:
    """A search result combining internal and external sources."""

    result_id: str
    source: str  # "external_paper", "internal_experiment", "internal_circuit", etc.
    title: str
    snippet: str  # search snippet/highlight
    relevance_score: float
    metadata: Dict[str, Any]  # PaperMetadata or internal entity data
    link: str  # URL or identifier


@dataclass
class SearchQuery:
    """Natural language search query for the RAG layer."""

    query: str
    # Search mode preferences
    semantic_weight: float = 0.6  # 0.0 = keyword-only, 1.0 = semantic-only
    domain_filter: Optional[Set[str]] = None  # e.g., {"mechanistic", "transformers", "IOI"}
    recency_filter: Optional[tuple] = None  # (min_year, max_year)


class RagIntegration:
    """Central RAG integration layer for MECH."""

    def __init__(
        self,
        knowledge_base: Optional[KnowledgeBaseEngine] = None,
        research_graph: Optional[TypedResearchGraph] = None,
    ) -> None:
        self.knowledge_base = knowledge_base or KnowledgeBaseEngine()
        self.research_graph = research_graph or TypedResearchGraph()

        # In-memory index of papers (in production, this would be a vector DB)
        self.paper_index: Dict[str, PaperMetadata] = {}
        self.keyword_index: Dict[str, Set[str]] = {}  # term -> paper_ids

        # Experiment result index for hybrid search
        self.experiment_index: Dict[str, Dict[str, Any]] = {}

    # ---- Paper Indexing ----

    def index_paper(self, paper: PaperMetadata) -> None:
        """Index a paper's metadata for search.

        Updates both the paper index and the keyword inverted index.
        Also extracts knowledge graph nodes via the literature learning pipeline.
        """
        self.paper_index[paper.paper_id] = paper

        # Build keyword index
        terms = self._extract_terms(paper.title, paper.abstract, paper.claims, paper.methods)
        for term in terms:
            if term.lower() not in self.keyword_index:
                self.keyword_index[term.lower()] = set()
            self.keyword_index[term.lower()].add(paper.paper_id)

        # Optionally add to research graph as a Publication node
        self.research_graph.add_node(
            node_id=paper.paper_id,
            node_type="Publication",
            label=paper.title,
        )

        # Store key claims/facts in the knowledge base for later retrieval
        for claim in paper.claims:
            self.knowledge_base.store_fact(
                entity=paper.title,
                prop="claim",
                value=claim,
                confidence=paper.confidence_score,
            )

        for method in paper.methods:
            self.knowledge_base.store_fact(
                entity=paper.title,
                prop="method",
                value=method,
                confidence=paper.confidence_score,
            )

    def index_experiment_result(
        self,
        experiment_id: str,
        title: str,
        description: str,
        key_findings: List[str],
        related_papers: Optional[List[str]] = None,
    ) -> None:
        """Index an experiment result for hybrid search with external literature.

        Links experiment findings to relevant papers, enabling cross-reference search.
        """
        entry = {
            "result_id": experiment_id,
            "title": title,
            "snippet": description,
            "source": "internal_experiment",
            "key_findings": key_findings,
            "related_papers": related_papers or [],
        }
        self.experiment_index[experiment_id] = entry

        # Also add key findings to knowledge base
        for finding in key_findings:
            self.knowledge_base.store_fact(
                entity=experiment_id,
                prop="finding",
                value=finding,
                confidence=0.8,
            )

    # ---- Hybrid Search ----

    def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform hybrid search across external papers + internal MECH results.

        Combines:
        1. Semantic search against paper index (title, claims, methods, abstract)
        2. Keyword search against inverted index
        3. Internal experiment/circuit graph search
        4. Fuses and ranks results
        """
        query_lower = query.query.lower().strip()
        if not query_lower:
            return []

        results: List[SearchResult] = []

        # 1. Search external paper index
        paper_results = self._search_papers_semantic(query_lower, query.semantic_weight)
        results.extend(paper_results)

        # 2. Search keyword index
        keyword_results = self._search_papers_keyword(query_lower)
        results.extend(keyword_results)

        # 3. Search internal experiment index
        experiment_results = self._search_experiments(query_lower)
        results.extend(experiment_results)

        # 4. Search research graph (related entities)
        graph_results = self._search_research_graph(query_lower)
        results.extend(graph_results)

        # Deduplicate and fuse results
        fused = self._fuse_results(results, query)

        # Sort by relevance score (descending)
        fused.sort(key=lambda r: r.relevance_score, reverse=True)

        return fused[:20]  # Return top 20

    def _search_papers_semantic(
        self, query: str, semantic_weight: float
    ) -> List[SearchResult]:
        """Semantic search across the paper index.

        Uses simple TF-IDF-inspired scoring based on term presence in
        claims, methods, title, and abstract.
        """
        if not self.paper_index:
            return []

        scored: List[tuple[float, SearchResult]] = []

        for paper_id, paper in self.paper_index.items():
            score = 0.0

            # Title match (highest weight)
            title_terms = re.findall(r"\w+", paper.title.lower())
            query_terms = set(re.findall(r"\w+", query))
            title_overlap = len(set(title_terms) & query_terms)
            if title_overlap > 0:
                score += 0.4 * (title_overlap / max(len(query_terms), 1))

            # Claims match
            claims_terms = set()
            # Safely extract claims - may be list or dict from metadata
            claims_data = paper.claims if hasattr(paper, 'claims') else paper.get("claims", [])
            if isinstance(claims_data, list):
                for claim in claims_data:
                    claims_terms.update(re.findall(r"\w+", str(claim).lower()))
            elif isinstance(claims_data, dict):
                for v in claims_data.values():
                    claims_terms.update(re.findall(r"\w+", str(v).lower()))
            claims_overlap = len(claims_terms & query_terms)
            score += 0.25 * (claims_overlap / max(len(query_terms), 1))

            # Methods match
            methods_terms = set()
            for method in paper.methods:
                methods_terms.update(re.findall(r"\w+", method.lower()))
            methods_overlap = len(methods_terms & query_terms)
            score += 0.15 * (methods_overlap / max(len(query_terms), 1))

            # Abstract match (if available)
            if paper.abstract:
                abstract_terms = set(re.findall(r"\w+", paper.abstract.lower()))
                abstract_overlap = len(abstract_terms & query_terms)
                score += 0.1 * (abstract_overlap / max(len(query_terms), 1))

            # Confidence boost
            score *= paper.confidence_score

            # Convert metadata to dict safely (handles both PaperMetadata dataclass and dict)
            if hasattr(paper, '__dataclass_fields__'):
                meta_dict = asdict(paper)
            elif isinstance(paper, dict):
                meta_dict = paper
            else:
                meta_dict = {k: v for k, v in paper.__dict__.items() if not k.startswith('_')}

            scored.append((score, SearchResult(
                result_id=paper.paper_id,
                source="external_paper",
                title=paper.title,
                snippet=f"{paper.title}: {paper.results_summary[:100]}..." if paper.results_summary else paper.title,
                relevance_score=round(score, 4),
                metadata=meta_dict,
                link=self._paper_link(paper),
            )))

        return [sr for _, sr in scored]

    def _search_papers_keyword(self, query: str) -> List[SearchResult]:
        """Keyword search against the inverted index."""
        if not self.keyword_index:
            return []

        query_terms = set(re.findall(r"\w+", query))
        if not query_terms:
            return []

        # Find papers matching any query term
        matching_papers: Dict[str, int] = {}  # paper_id -> match count
        for term in query_terms:
            if term in self.keyword_index:
                for paper_id in self.keyword_index[term]:
                    matching_papers[paper_id] = matching_papers.get(paper_id, 0) + 1

        scored: List[tuple[float, SearchResult]] = []
        for paper_id, match_count in matching_papers.items():
            paper = self.paper_index[paper_id]
            # Base score from match frequency
            base = 0.3 * (match_count / max(len(query_terms), 1))
            # Confidence boost
            boosted = base * paper.confidence_score
            # Convert metadata to dict safely (handles PaperMetadata dataclass and other types)
            if hasattr(paper, '__dataclass_fields__'):
                meta_dict = asdict(paper)
            elif isinstance(paper, dict):
                meta_dict = paper
            else:
                meta_dict = {k: v for k, v in paper.__dict__.items() if not k.startswith('_')}

            scored.append((boosted, SearchResult(
                result_id=paper_id,
                source="external_paper",
                title=paper.title,
                snippet=f"{paper.title}: related to '{query[:50]}'",
                relevance_score=round(boosted, 4),
                metadata=meta_dict,
                link=self._paper_link(paper),
            )))

        return [sr for _, sr in scored]

    def _search_experiments(self, query: str) -> List[SearchResult]:
        """Search internal experiment index."""
        if not self.experiment_index:
            return []

        query_terms = set(re.findall(r"\w+", query))
        if not query_terms:
            return []

        scored: List[tuple[float, SearchResult]] = []
        for exp_id, entry in self.experiment_index.items():
            score = 0.0
            snippet_terms = set(re.findall(r"\w+", entry["snippet"].lower()))
            title_terms = set(re.findall(r"\w+", entry["title"].lower()))

            # Title overlap
            title_overlap = len(title_terms & query_terms)
            score += 0.5 * (title_overlap / max(len(query_terms), 1))

            # Key findings overlap
            findings_terms = set()
            for finding in entry["key_findings"]:
                findings_terms.update(re.findall(r"\w+", finding.lower()))
            findings_overlap = len(findings_terms & query_terms)
            score += 0.3 * (findings_overlap / max(len(query_terms), 1))

            if score > 0:
                scored.append((score, SearchResult(
                    result_id=exp_id,
                    source="internal_experiment",
                    title=entry["title"],
                    snippet=entry["snippet"][:150] + "...",
                    relevance_score=round(score, 4),
                    metadata=entry,
                    link=f"experiment:{exp_id}",
                )))

        return [sr for _, sr in scored]

    def _search_research_graph(self, query: str) -> List[SearchResult]:
        """Search the typed research graph for related entities."""
        # Simple node label matching
        query_terms = set(re.findall(r"\w+", query))
        if not query_terms:
            return []

        scored: List[tuple[float, SearchResult]] = []
        for node in self.research_graph.nodes:
            node_label = node["label"].lower()
            node_type = node["type"]

            # Match against node labels
            label_overlap = len(set(re.findall(r"\w+", node_label)) & query_terms)
            if label_overlap > 0:
                # Higher score for certain node types
                type_weight = {"Question": 0.5, "Hypothesis": 0.7, "Circuit": 0.8, "Conclusion": 0.6, "Publication": 0.4, "Experiment": 0.7}.get(node_type, 0.5)
                score = 0.4 * (label_overlap / max(len(query_terms), 1)) * type_weight

                scored.append((score, SearchResult(
                    result_id=node["id"],
                    source=f"internal_{node_type.lower()}",
                    title=node["label"],
                    snippet=f"{node['label']} ({node_type})",
                    relevance_score=round(score, 4),
                    metadata={"id": node["id"], "type": node_type, "label": node["label"]},
                    link=f"graph:{node['id']}",
                )))

        return [sr for _, sr in scored]

    def _fuse_results(self, results: List[SearchResult], query: SearchQuery) -> List[SearchResult]:
        """Deduplicate and fuse results from multiple sources.

        - De-duplicates by result_id
        - Applies domain recency filtering if specified
        - Boosts results matching domain_filter
        """
        seen: Dict[str, SearchResult] = {}
        for r in results:
            if r.result_id not in seen:
                seen[r.result_id] = r
            else:
                # Merge metadata from duplicate sources
                existing = seen[r.result_id]
                existing.metadata = {**existing.metadata, **r.metadata}
                # Take higher relevance score
                if r.relevance_score > existing.relevance_score:
                    existing.relevance_score = r.relevance_score

        # Apply recency filter
        if query.recency_filter:
            min_year, max_year = query.recency_filter
            filtered = []
            for r in seen.values():
                meta = r.metadata
                # Try to extract year from various sources
                year = None
                if "year" in meta:
                    year = meta["year"]
                elif hasattr(meta.get("paper", {}), 'year') if isinstance(meta.get('paper'), dict) else False:
                    pass
                # If we can't determine year, include it (conservative)
                if year is None or (min_year <= year <= max_year):
                    filtered.append(r)
                # If year outside range, skip
            seen = {r.result_id: r for r in filtered}

        # Apply domain filter boost
        if query.domain_filter:
            for r in seen.values():
                meta = r.metadata
                # Check if metadata indicates matching domain
                text_to_check = str(meta).lower()
                for domain in query.domain_filter:
                    if domain.lower() in text_to_check:
                        r.relevance_score = min(1.0, r.relevance_score + 0.1)
                        break

        return list(seen.values())

    # ---- Hypothesis Enrichment ----

    def enrich_hypothesis(self, hypothesis: str, max_citations: int = 3) -> str:
        """Augment a hypothesis with related paper citations.

        Searches the RAG index for papers related to the hypothesis topic
        and appends relevant citations with brief rationales.
        """
        # Search for papers related to the hypothesis topic
        search_query = SearchQuery(query=hypothesis, semantic_weight=0.8)
        search_results = self.search(search_query)

        citations: List[str] = []
        for result in search_results[:max_citations]:
            meta = result.metadata
            # Determine what kind of metadata we have
            paper_meta = None
            if isinstance(meta, dict):
                # Check if it's a PaperMetadata dict (has paper_id, title, etc.)
                if "paper_id" in meta and "title" in meta and "claims" in meta:
                    paper_meta = meta
                # Check if it has 'title' at top level with 'claims'
                elif meta.get("claims"):
                    paper_meta = meta

            if paper_meta and paper_meta.get("claims"):
                # Pick the most relevant claim(s)
                relevant_claims = paper_meta["claims"][:1]
                citation_text = f"[{paper_meta.get('title', 'Unknown')}]: {relevant_claims[0]}"
                citations.append(citation_text)

        if citations:
            return hypothesis + "\n\nRelated work:\n" + "\n".join(f"  - {c}" for c in citations)
        return hypothesis

    # ---- Query Assistance ----

    def get_related_papers(self, paper_id: str, max_results: int = 5) -> List[SearchResult]:
        """Get papers related to a given paper ID via claims/methods overlap."""
        if paper_id not in self.paper_index:
            return []

        paper = self.paper_index[paper_id]
        # Build query from the paper's claims and methods
        query_terms: List[str] = []
        for claim in paper.claims:
            query_terms.extend(re.findall(r"\w+", claim.lower()))
        for method in paper.methods:
            query_terms.extend(re.findall(r"\w+", method.lower()))

        if not query_terms:
            return []

        search_query = SearchQuery(query=" ".join(set(query_terms)), semantic_weight=0.8)
        results = self.search(search_query)
        # Exclude the source paper itself
        filtered = [r for r in results if r.result_id != paper_id]
        return filtered[:max_results]

    # ---- Helpers ----

    @staticmethod
    def _extract_terms(*texts: Any) -> List[str]:
        """Extract search-friendly terms from multiple text sources.

        Handles both str and List[str] inputs.
        """
        terms: Set[str] = set()
        for text in texts:
            if text is None:
                continue
            if isinstance(text, str):
                if text:
                    terms.update(re.findall(r"\w+", text.lower()))
            elif isinstance(text, list):
                for item in text:
                    if item:
                        terms.update(re.findall(r"\w+", str(item).lower()))
            elif isinstance(text, (int, float)):
                if text:
                    terms.update(re.findall(r"\w+", str(text).lower()))
        return list(terms)

    @staticmethod
    def _paper_link(paper: PaperMetadata) -> str:
        """Generate a URL link for a paper based on its source."""
        if paper.source == "arxiv":
            return f"https://arxiv.org/abs/{paper.doi.split('/')[-1]}"
        elif paper.source == "github":
            return f"https://github.com/search?q={paper.doi}"
        elif paper.source == "context7":
            return f"https://context7.dev/p/{paper.paper_id}"
        return f"https://doi.org/{paper.doi}"


# Convenience function for quick integration with AutonomousResearchEngine
def integrate_rag_into_engine(
    engine: Any,
    rag: Optional[RagIntegration] = None,
) -> RagIntegration:
    """Wire a RagIntegration instance into an AutonomousResearchEngine.

    Adds the rag engine as an attribute and enriches hypothesis generation.
    """
    rag_instance = rag or RagIntegration()
    engine.rag = rag_instance  # type: ignore[attr-defined]

    # Wrap hypothesis generation to add citation enrichment
    original_generate = getattr(engine, 'generate_hypotheses', None)

    def enriched_hypothesis_generation(goal: str) -> List[str]:
        # Generate base hypotheses (would come from the original engine)
        hypotheses = engine.hypothesis_generator.generate_hypotheses(context_prompt=goal) if engine.hypothesis_generator else []

        # enrich each hypothesis with citations
        enriched = []
        for h in hypotheses:
            enriched_h = rag_instance.enrich_hypothesis(h)
            enriched.append(enriched_h)
        return enriched

    # Monkey-patch if hypothesis_generator has generate_hypotheses
    if original_generate and callable(original_generate):
        engine.hypothesis_generator.generate_hypotheses = lambda context_prompt: enriched_hypothesis_generation(context_prompt)  # type: ignore[method-assign,assignment]

    return rag_instance