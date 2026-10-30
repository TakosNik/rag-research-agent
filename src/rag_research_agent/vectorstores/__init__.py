"""Vector stores keep chunks with their vectors and find the most similar ones."""

from rag_research_agent.config import Settings
from rag_research_agent.embeddings import Embedder
from rag_research_agent.vectorstores.base import VectorStore
from rag_research_agent.vectorstores.memory import InMemoryVectorStore
from rag_research_agent.vectorstores.pgvector import PgVectorStore

STORES = ("pgvector", "memory")


def get_vector_store(settings: Settings, embedder: Embedder) -> VectorStore:
    """Create the vector store chosen by the `vector_store` setting.

    The embedder is needed because a pgvector table is built for one model's
    vector size and refuses vectors from any other model.
    """
    store = settings.vector_store.strip().lower()
    if store == "memory":
        return InMemoryVectorStore()
    if store == "pgvector":
        return PgVectorStore(
            database_url=settings.database_url.get_secret_value(),
            dimensions=embedder.dimensions,
            embedding_model=embedder.model_name,
        )
    raise ValueError(
        f"Unknown vector store {settings.vector_store!r}; choose one of: {', '.join(STORES)}"
    )


__all__ = ["InMemoryVectorStore", "PgVectorStore", "VectorStore", "get_vector_store"]
