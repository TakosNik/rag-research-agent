"""The shared interface that every vector store implements."""

from abc import ABC, abstractmethod

from rag_research_agent.models import Chunk, SearchResult


class VectorStore(ABC):
    """Base class for all vector stores.

    A vector store keeps chunks together with their vectors (from an
    `Embedder`). Given the vector of a question, it finds the stored chunks
    whose vectors are most similar, which are the chunks most likely to help
    answer it.
    """

    @abstractmethod
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Store each chunk with its vector, matched by position in the two lists.

        "Upsert" means update or insert: a chunk whose id is not stored yet is
        added, and a chunk whose id is already stored replaces the old one. So
        indexing the same document twice never creates duplicates.

        Raises ValueError if the two lists have different lengths.
        """

    @abstractmethod
    def delete_document(self, document_id: str) -> int:
        """Remove every chunk of a document and return how many were removed.

        This keeps answers up to date. When a document is edited it may get
        fewer chunks than before, and upserting the new chunks would leave the
        old extra ones behind; when a document is deleted at the source, none
        of its chunks should be found any more. Deleting a document's chunks
        before re-indexing it (or after it is removed) stops searches from
        returning outdated text. Returns 0 if the document has no chunks.
        """

    @abstractmethod
    def search(
        self, query_vector: list[float], top_k: int = 5, source: str | None = None
    ) -> list[SearchResult]:
        """Return up to `top_k` chunks most similar to `query_vector`, best first.

        If `source` is given (e.g. "local" or "s3"), only chunks from that
        source are considered. An empty store gives an empty list.
        """

    @abstractmethod
    def count(self) -> int:
        """Number of chunks currently stored."""
