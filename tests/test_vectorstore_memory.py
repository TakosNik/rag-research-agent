import pytest

from rag_research_agent.embeddings import FakeEmbedder
from rag_research_agent.models import Chunk
from rag_research_agent.vectorstores import InMemoryVectorStore, VectorStore

embedder = FakeEmbedder()


def make_chunk(document_id, index, text, source="local"):
    return Chunk(
        id=f"{document_id}#{index}",
        document_id=document_id,
        source=source,
        title=f"{document_id}.txt",
        uri=f"/docs/{document_id}.txt",
        index=index,
        text=text,
    )


def store_with(chunks):
    store = InMemoryVectorStore()
    store.upsert(chunks, embedder.embed_documents([chunk.text for chunk in chunks]))
    return store


def search(store, question, **kwargs):
    return store.search(embedder.embed_query(question), **kwargs)


def test_vector_store_cannot_be_created_directly():
    with pytest.raises(TypeError):
        VectorStore()


def test_upsert_then_count():
    store = store_with(
        [
            make_chunk("vpn", 0, "Enable the VPN before connecting."),
            make_chunk("vpn", 1, "The VPN client is in the software centre."),
        ]
    )

    assert store.count() == 2


def test_upserting_the_same_id_replaces_the_chunk():
    store = store_with([make_chunk("vpn", 0, "Old text about the office kitchen.")])

    new_chunk = make_chunk("vpn", 0, "New text about the VPN client.")
    store.upsert([new_chunk], embedder.embed_documents([new_chunk.text]))

    assert store.count() == 1
    [result] = search(store, "VPN client")
    assert result.chunk.text == "New text about the VPN client."


def test_search_returns_the_most_similar_chunk_first():
    store = store_with(
        [
            make_chunk("kitchen", 0, "The office kitchen is cleaned every Friday."),
            make_chunk("password", 0, "To reset your password, open the settings page."),
            make_chunk("vpn", 0, "Enable the VPN before connecting."),
        ]
    )

    results = search(store, "How do I reset my password?")

    assert len(results) == 3
    assert results[0].chunk.id == "password#0"
    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_the_number_of_results():
    store = store_with([make_chunk("doc", i, f"Chunk number {i} about VPN.") for i in range(10)])

    assert len(search(store, "VPN", top_k=3)) == 3
    assert len(search(store, "VPN")) == 5


def test_top_k_below_one_raises_value_error():
    store = store_with([make_chunk("vpn", 0, "Enable the VPN.")])

    with pytest.raises(ValueError):
        search(store, "VPN", top_k=0)


def test_source_filter_only_returns_chunks_from_that_source():
    store = store_with(
        [
            make_chunk("local-vpn", 0, "Enable the VPN before connecting.", source="local"),
            make_chunk("s3-vpn", 0, "Enable the VPN before connecting.", source="s3"),
            make_chunk("s3-kitchen", 0, "The kitchen is cleaned on Friday.", source="s3"),
        ]
    )

    results = search(store, "VPN", source="s3")

    assert [result.chunk.source for result in results] == ["s3", "s3"]
    assert results[0].chunk.id == "s3-vpn#0"
    assert search(store, "VPN", source="jira") == []


def test_delete_document_removes_only_that_documents_chunks():
    store = store_with(
        [
            make_chunk("vpn", 0, "Enable the VPN before connecting."),
            make_chunk("vpn", 1, "The VPN client is in the software centre."),
            make_chunk("password", 0, "To reset your password, open the settings page."),
        ]
    )

    assert store.delete_document("vpn") == 2
    assert store.count() == 1
    assert [result.chunk.id for result in search(store, "VPN password")] == ["password#0"]
    assert store.delete_document("vpn") == 0


def test_searching_an_empty_store_returns_empty_list():
    assert search(InMemoryVectorStore(), "anything") == []


def test_mismatched_list_lengths_raise_value_error():
    store = InMemoryVectorStore()
    chunks = [make_chunk("vpn", 0, "One."), make_chunk("vpn", 1, "Two.")]

    with pytest.raises(ValueError):
        store.upsert(chunks, embedder.embed_documents(["One."]))
    assert store.count() == 0
