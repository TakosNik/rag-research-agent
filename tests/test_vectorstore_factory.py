import pytest

from rag_research_agent.config import Settings
from rag_research_agent.embeddings import FakeEmbedder
from rag_research_agent.vectorstores import InMemoryVectorStore, get_vector_store


def test_memory_setting_gives_in_memory_store():
    store = get_vector_store(Settings(_env_file=None, vector_store="memory"), FakeEmbedder())
    assert isinstance(store, InMemoryVectorStore)


def test_unknown_vector_store_raises():
    with pytest.raises(ValueError, match="Unknown vector store"):
        get_vector_store(Settings(_env_file=None, vector_store="nope"), FakeEmbedder())
