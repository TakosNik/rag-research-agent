"""A vector store that keeps everything in memory, for tests and small demos."""

from rag_research_agent.embeddings import cosine_similarity
from rag_research_agent.models import Chunk, SearchResult
from rag_research_agent.vectorstores.base import VectorStore


class InMemoryVectorStore(VectorStore):
    """Keeps chunks and vectors in a Python dict and searches by brute force.

    Every search compares the query against every stored vector, and all data
    is lost when the program stops. Fine for tests and small demos, not for
    production.
    """

    def __init__(self) -> None:
        # Chunk id -> (chunk, vector)
        self._items: dict[str, tuple[Chunk, list[float]]] = {}

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"got {len(chunks)} chunks but {len(vectors)} vectors; they must match"
            )
        for chunk, vector in zip(chunks, vectors, strict=True):
            # Copy the vector so later changes to the caller's list don't affect us
            self._items[chunk.id] = (chunk, list(vector))

    def delete_document(self, document_id: str) -> int:
        ids = [id for id, (chunk, _) in self._items.items() if chunk.document_id == document_id]
        for id in ids:
            del self._items[id]
        return len(ids)

    def search(
        self, query_vector: list[float], top_k: int = 5, source: str | None = None
    ) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        results = [
            SearchResult(chunk=chunk, score=cosine_similarity(query_vector, vector))
            for chunk, vector in self._items.values()
            if source is None or chunk.source == source
        ]
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]

    def count(self) -> int:
        return len(self._items)
