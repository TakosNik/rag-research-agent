import math

import pytest

from rag_research_agent.embeddings import Embedder, FakeEmbedder, cosine_similarity


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
