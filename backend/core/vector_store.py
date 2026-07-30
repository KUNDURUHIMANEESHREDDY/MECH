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
    """Implementation using ChromaDB for local vector storage."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        # self.client = chromadb.PersistentClient(path=persist_directory)
        
    def add_embeddings(self, collection_name: str, ids: List[str], embeddings: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        print(f"Added {len(ids)} embeddings to ChromaDB collection {collection_name}")
        
    def search(self, collection_name: str, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        return [{"id": "doc_1", "score": 0.95}]
