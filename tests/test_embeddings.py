import math

import fastembed
import numpy as np
import pytest

from rag_research_agent.config import Settings
from rag_research_agent.embeddings import (
    Embedder,
    FakeEmbedder,
    FastEmbedEmbedder,
    cosine_similarity,
    get_embedder,
)


def length(vector):
    return math.sqrt(sum(x * x for x in vector))


def test_embedder_cannot_be_created_directly():
    with pytest.raises(TypeError):
        Embedder()


def test_vectors_have_the_right_length():
    embedder = FakeEmbedder(dimensions=64)

    assert embedder.dimensions == 64
    assert len(embedder.embed_query("How do I reset my password?")) == 64
    vectors = embedder.embed_documents(["First chunk.", "Second chunk."])
    assert [len(vector) for vector in vectors] == [64, 64]


def test_vectors_are_normalised_to_length_one():
    vector = FakeEmbedder().embed_query("Reset your password in the settings page.")
    assert length(vector) == pytest.approx(1.0)


def test_same_text_gives_the_same_vector():
    text = "The VPN must be enabled before connecting."

    assert FakeEmbedder().embed_query(text) == FakeEmbedder().embed_query(text)
    assert FakeEmbedder().embed_documents([text]) == [FakeEmbedder().embed_query(text)]


def test_texts_sharing_words_are_more_similar_than_unrelated_texts():
    embedder = FakeEmbedder()
    query = embedder.embed_query("How do I reset my password?")
    related, unrelated = embedder.embed_documents(
        [
            "To reset your password, open the settings page.",
            "The office kitchen is cleaned every Friday afternoon.",
        ]
    )

    assert cosine_similarity(query, related) > cosine_similarity(query, unrelated)


def test_embed_documents_on_empty_list_returns_empty_list():
    assert FakeEmbedder().embed_documents([]) == []


def test_text_without_words_gives_a_zero_vector():
    assert FakeEmbedder(dimensions=8).embed_query("") == [0.0] * 8


def test_dimensions_must_be_positive():
    with pytest.raises(ValueError):
        FakeEmbedder(dimensions=0)


def test_cosine_of_a_vector_with_itself_is_one():
    vector = [0.5, -1.0, 2.0]
    assert cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_cosine_of_perpendicular_and_opposite_vectors():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 2.0], [-1.0, -2.0]) == pytest.approx(-1.0)


def test_cosine_ignores_vector_length():
    assert cosine_similarity([1.0, 2.0], [10.0, 20.0]) == pytest.approx(1.0)


def test_cosine_with_a_zero_vector_is_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_cosine_rejects_vectors_of_different_lengths():
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


class FakeTextEmbedding:
    """Stands in for fastembed.TextEmbedding, so these tests stay fast and offline."""

    created = 0

    def __init__(self, model_name):
        FakeTextEmbedding.created += 1
        self.model_name = model_name
        self.embedding_size = 3

    def passage_embed(self, texts):
        return (np.array([1.0, 0.0, 0.0], dtype=np.float32) for _ in texts)

    def query_embed(self, query):
        return (np.array([0.0, 1.0, 0.0], dtype=np.float32) for _ in query)


@pytest.fixture
def fake_text_embedding(monkeypatch):
    FakeTextEmbedding.created = 0
    monkeypatch.setattr(fastembed, "TextEmbedding", FakeTextEmbedding)
    return FakeTextEmbedding


def test_get_embedder_returns_fake_embedder_by_default(monkeypatch):
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    assert isinstance(get_embedder(Settings(_env_file=None)), FakeEmbedder)


def test_get_embedder_returns_fastembed_embedder_with_model_name():
    settings = Settings(
        _env_file=None, embedding_provider="fastembed", embedding_model="example/model"
    )

    embedder = get_embedder(settings)

    assert isinstance(embedder, FastEmbedEmbedder)
    assert embedder.model_name == "example/model"


def test_get_embedder_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown embedding provider 'nope'"):
        get_embedder(Settings(_env_file=None, embedding_provider="nope"))


def test_fastembed_embedder_does_not_load_model_on_creation(fake_text_embedding):
    FastEmbedEmbedder()
    assert fake_text_embedding.created == 0


def test_fastembed_embedder_loads_model_once_on_first_use(fake_text_embedding):
    embedder = FastEmbedEmbedder("example/model")

    embedder.embed_query("first")
    embedder.embed_query("second")

    assert fake_text_embedding.created == 1
    assert embedder._model.model_name == "example/model"


def test_fastembed_embedder_uses_passage_and_query_methods(fake_text_embedding):
    embedder = FastEmbedEmbedder()

    documents = embedder.embed_documents(["chunk one", "chunk two"])
    query = embedder.embed_query("question")

    assert documents == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert query == [0.0, 1.0, 0.0]
    # Plain Python floats, not numpy types
    assert all(type(x) is float for x in documents[0] + query)


def test_fastembed_embedder_dimensions_come_from_model(fake_text_embedding):
    assert FastEmbedEmbedder().dimensions == 3


def test_fastembed_embed_documents_on_empty_list_does_not_load_model(fake_text_embedding):
    assert FastEmbedEmbedder().embed_documents([]) == []
    assert fake_text_embedding.created == 0


@pytest.mark.integration
def test_real_model_finds_related_sentence():
    embedder = FastEmbedEmbedder()
    query = embedder.embed_query("How do I connect remotely?")
    vpn, cafeteria = embedder.embed_documents(
        [
            "Use the company VPN to access internal systems when working from home.",
            "The cafeteria serves lunch from noon until two o'clock.",
        ]
    )

    assert embedder.dimensions == 384
    assert len(query) == len(vpn) == 384
    assert cosine_similarity(query, vpn) > cosine_similarity(query, cafeteria)
