"""Embedders turn text into vectors of numbers, so similar texts can be found."""

from rag_research_agent.embeddings.base import Embedder, cosine_similarity
from rag_research_agent.embeddings.fake import FakeEmbedder

__all__ = ["Embedder", "FakeEmbedder", "cosine_similarity"]
