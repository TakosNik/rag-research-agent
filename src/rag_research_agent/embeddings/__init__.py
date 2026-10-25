"""Embedders turn text into vectors of numbers, so similar texts can be found."""

from rag_research_agent.config import Settings
from rag_research_agent.embeddings.base import Embedder, cosine_similarity
from rag_research_agent.embeddings.fake import FakeEmbedder
from rag_research_agent.embeddings.fastembed import FastEmbedEmbedder

PROVIDERS = ("fake", "fastembed")


def get_embedder(settings: Settings) -> Embedder:
    """Create the embedder chosen by the `embedding_provider` setting."""
    provider = settings.embedding_provider.strip().lower()
    if provider == "fake":
        return FakeEmbedder()
    if provider == "fastembed":
        return FastEmbedEmbedder(model_name=settings.embedding_model)
    raise ValueError(
        f"Unknown embedding provider {settings.embedding_provider!r}; "
        f"choose one of: {', '.join(PROVIDERS)}"
    )


__all__ = ["Embedder", "FakeEmbedder", "FastEmbedEmbedder", "cosine_similarity", "get_embedder"]
