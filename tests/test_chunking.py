from datetime import UTC, datetime

import pytest

from rag_research_agent.chunking import chunk_document, chunk_text
from rag_research_agent.models import SourceDocument


def make_paragraphs(count):
    return "\n\n".join(
        f"Paragraph {n} talks about topic {n}. It has a second sentence too." for n in range(count)
    )


def words(text):
    return text.split()


def test_short_text_gives_one_chunk():
    assert chunk_text("Just a short paragraph.\n\nAnd another one.") == [
        "Just a short paragraph.\n\nAnd another one."
    ]


def test_long_text_gives_several_chunks_under_the_limit():
    text = make_paragraphs(20)

    chunks = chunk_text(text, chunk_size=200, overlap=40)

    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)
    # Nothing is lost: every word of the text is in some chunk
    assert set(words(text)) <= {word for chunk in chunks for word in words(chunk)}


def test_each_chunk_starts_with_the_end_of_the_previous_one():
    chunks = chunk_text(make_paragraphs(20), chunk_size=200, overlap=40)

    for previous, current in zip(chunks, chunks[1:], strict=False):
        # Some start of `current`, of 20 to 40 characters, is the end of `previous`
        assert any(previous.endswith(current[:n]) for n in range(20, 41))


def test_zero_overlap_repeats_nothing():
    chunks = chunk_text(make_paragraphs(20), chunk_size=200, overlap=0)

    assert sum(len(words(chunk)) for chunk in chunks) == len(words(make_paragraphs(20)))


def test_very_long_paragraph_is_split_at_sentence_ends():
    paragraph = " ".join(f"This is sentence number {n}." for n in range(50))

    chunks = chunk_text(paragraph, chunk_size=200, overlap=40)

    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)
    assert all(chunk.endswith(".") for chunk in chunks)


def test_word_longer_than_chunk_is_cut():
    chunks = chunk_text("x" * 250, chunk_size=100, overlap=20)

    assert all(len(chunk) <= 100 for chunk in chunks)
    assert "".join(chunks).count("x") >= 250


def test_page_break_is_a_paragraph_break():
    assert chunk_text("Page one.\fPage two.") == ["Page one.\n\nPage two."]


@pytest.mark.parametrize("text", ["", "   ", "\n\n\f\n"])
def test_empty_text_gives_no_chunks(text):
    assert chunk_text(text) == []


@pytest.mark.parametrize("overlap", [200, 300, -1])
def test_bad_overlap_raises(overlap):
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("Some text.", chunk_size=200, overlap=overlap)


def test_chunk_document_fills_in_ids_and_indexes():
    doc = SourceDocument(
        id="doc-1",
        source="local",
        title="example.txt",
        uri="/tmp/example.txt",
        content=b"",
        mime_type="text/plain",
        last_modified=datetime(2026, 1, 1, tzinfo=UTC),
        metadata={"author": "Example Author"},
    )

    chunks = chunk_document(doc, make_paragraphs(50))

    assert len(chunks) > 1
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert [chunk.id for chunk in chunks] == [f"doc-1#{i}" for i in range(len(chunks))]
    first = chunks[0]
    assert first.document_id == "doc-1"
    assert first.source == "local"
    assert first.title == "example.txt"
    assert first.uri == "/tmp/example.txt"
    assert first.text.startswith("Paragraph 0")
    assert first.metadata == {"author": "Example Author"}
    # Metadata is copied, so changing a chunk's metadata leaves the document alone
    first.metadata["tag"] = "changed"
    assert doc.metadata == {"author": "Example Author"}
