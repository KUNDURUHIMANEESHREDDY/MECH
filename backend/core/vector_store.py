"""Abstract Vector Store Interface.

Decouples the application from a specific vector database provider.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

class VectorStore(ABC):
    @abstractmethod
    def add_embeddings(self, collection_name: str, ids: List[str], embeddings: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        pass

    @abstractmethod
    def search(self, collection_name: str, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        pass

class ChromaDBStore(VectorStore):
    """ChromaDB-backed vector storage.

    The client construction is commented out, so this class was never talking to
    a database. `search()` nevertheless returned `[{"id": "doc_1",
    "score": 0.95}]` -- the same document at the same confidence for every query,
    every collection, every embedding. Any caller ranking results by score got
    a confident, meaningless answer, and because the id was stable it looked
    like a real hit.

    Both methods now raise unless a real client is supplied.
    """

    def __init__(self, persist_directory: str = "./chroma_db",
                 client: Any = None) -> None:
        self.persist_directory = persist_directory
        self._client = client

    def _require_client(self) -> Any:
        if self._client is None:
            raise RuntimeError(
                "ChromaDBStore has no client. Construct it with "
                "client=chromadb.PersistentClient(path=...) or a compatible "
                "client. Returning fabricated search results is not an "
                "acceptable fallback: a fixed document at a fixed score reads "
                "as a real match to every caller."
            )
        return self._client

    def add_embeddings(
        self,
        collection_name: str,
        ids: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
    ) -> None:
        client = self._require_client()
        client.get_or_create_collection(collection_name).upsert(
            ids=list(ids), embeddings=list(embeddings),
            metadatas=list(metadata),
        )

    def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        client = self._require_client()
        result = client.get_or_create_collection(collection_name).query(
            query_embeddings=[list(query_embedding)], n_results=top_k,
        )
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        # Chroma returns a distance (lower is closer); callers ranking by
        # descending score want similarity, so convert rather than pass the
        # distance through under a field named "score".
        return [
            {"id": doc_id, "score": round(1.0 / (1.0 + float(dist)), 6)}
            for doc_id, dist in zip(ids, distances)
        ]
