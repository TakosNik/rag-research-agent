"""Tests against a real Postgres + pgvector; start it first with: docker compose up -d"""

import psycopg
import pytest

from rag_research_agent.config import Settings
from rag_research_agent.embeddings import FakeEmbedder, cosine_similarity
from rag_research_agent.models import Chunk
from rag_research_agent.vectorstores import PgVectorStore

pytestmark = pytest.mark.integration

# A separate table, so the tests never touch data in the real "chunks" table
TABLE = "test_chunks"
MODEL = "fake"

embedder = FakeEmbedder()


@pytest.fixture
def database_url():
    url = Settings().database_url.get_secret_value()
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {TABLE}")
    yield url
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {TABLE}")


@pytest.fixture
def store(database_url):
    return PgVectorStore(database_url, embedder.dimensions, MODEL, table=TABLE)


def make_chunk(document_id, index, text, source="local"):
    return Chunk(
        id=f"{document_id}#{index}",
        document_id=document_id,
        source=source,
        title=f"{document_id}.txt",
        uri=f"/docs/{document_id}.txt",
        index=index,
        text=text,
        metadata={"author": "Example Author"},
    )


def add(store, chunks):
    store.upsert(chunks, embedder.embed_documents([chunk.text for chunk in chunks]))


def search(store, question, **kwargs):
    return store.search(embedder.embed_query(question), **kwargs)


def test_upsert_then_count(store):
    add(
        store,
        [
            make_chunk("vpn", 0, "Enable the VPN before connecting."),
            make_chunk("vpn", 1, "The VPN client is in the software centre."),
        ],
    )

    assert store.count() == 2


def test_search_returns_all_chunk_fields(store):
    chunk = make_chunk("vpn", 0, "Enable the VPN before connecting.")
    add(store, [chunk])

    [result] = search(store, "VPN")

    assert result.chunk == chunk
    assert 0 < result.score <= 1


def test_upserting_the_same_id_replaces_the_chunk(store):
    add(store, [make_chunk("vpn", 0, "Old text about the office kitchen.")])
    add(store, [make_chunk("vpn", 0, "New text about the VPN client.")])

    assert store.count() == 1
    [result] = search(store, "VPN client")
    assert result.chunk.text == "New text about the VPN client."


def test_search_returns_the_most_similar_chunk_first(store):
    add(
        store,
        [
            make_chunk("kitchen", 0, "The office kitchen is cleaned every Friday."),
            make_chunk("password", 0, "To reset your password, open the settings page."),
            make_chunk("vpn", 0, "Enable the VPN before connecting."),
        ],
    )

    results = search(store, "How do I reset my password?")

    assert len(results) == 3
    assert results[0].chunk.id == "password#0"
    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)


def test_scores_match_the_in_memory_cosine_similarity(store):
    text = "To reset your password, open the settings page."
    add(store, [make_chunk("password", 0, text)])
    query = embedder.embed_query("reset password")

    [result] = store.search(query)

    expected = cosine_similarity(query, embedder.embed_documents([text])[0])
    assert result.score == pytest.approx(expected, abs=1e-5)


def test_top_k_limits_the_number_of_results(store):
    add(store, [make_chunk("doc", i, f"Chunk number {i} about VPN.") for i in range(10)])

    assert len(search(store, "VPN", top_k=3)) == 3
    assert len(search(store, "VPN")) == 5


def test_top_k_below_one_raises_value_error(store):
    with pytest.raises(ValueError):
        search(store, "VPN", top_k=0)


def test_source_filter_only_returns_chunks_from_that_source(store):
    add(
        store,
        [
            make_chunk("local-vpn", 0, "Enable the VPN before connecting.", source="local"),
            make_chunk("s3-vpn", 0, "Enable the VPN before connecting.", source="s3"),
            make_chunk("s3-kitchen", 0, "The kitchen is cleaned on Friday.", source="s3"),
        ],
    )

    results = search(store, "VPN", source="s3")

    assert [result.chunk.source for result in results] == ["s3", "s3"]
    assert results[0].chunk.id == "s3-vpn#0"
    assert search(store, "VPN", source="jira") == []


def test_delete_document_removes_only_that_documents_chunks(store):
    add(
        store,
        [
            make_chunk("vpn", 0, "Enable the VPN before connecting."),
            make_chunk("vpn", 1, "The VPN client is in the software centre."),
            make_chunk("password", 0, "To reset your password, open the settings page."),
        ],
    )

    assert store.delete_document("vpn") == 2
    assert store.count() == 1
    assert [result.chunk.id for result in search(store, "VPN password")] == ["password#0"]
    assert store.delete_document("vpn") == 0


def test_searching_an_empty_store_returns_empty_list(store):
    assert search(store, "anything") == []


def test_mismatched_list_lengths_raise_value_error(store):
    chunks = [make_chunk("vpn", 0, "One."), make_chunk("vpn", 1, "Two.")]

    with pytest.raises(ValueError):
        store.upsert(chunks, embedder.embed_documents(["One."]))
    assert store.count() == 0


def test_data_survives_creating_a_new_store_object(store, database_url):
    add(store, [make_chunk("vpn", 0, "Enable the VPN before connecting.")])

    again = PgVectorStore(database_url, embedder.dimensions, MODEL, table=TABLE)

    assert again.count() == 1


def test_different_dimensions_raise_a_clear_error(store, database_url):
    with pytest.raises(ValueError, match="dimensions"):
        PgVectorStore(database_url, 128, MODEL, table=TABLE)


def test_different_embedding_model_raises_a_clear_error(store, database_url):
    add(store, [make_chunk("vpn", 0, "Enable the VPN before connecting.")])

    with pytest.raises(ValueError, match="embedding model"):
        PgVectorStore(database_url, embedder.dimensions, "another-model", table=TABLE)
