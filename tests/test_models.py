from datetime import UTC, datetime

from rag_research_agent.models import SourceDocument


def make_document(**overrides):
    fields = {
        "id": "doc-1",
        "source": "local",
        "title": "example.txt",
        "uri": "/tmp/example.txt",
        "content": b"Hello, world!",
        "mime_type": "text/plain",
        "last_modified": datetime(2026, 1, 1, tzinfo=UTC),
    }
    fields.update(overrides)
    return SourceDocument(**fields)


def test_create_source_document():
    doc = make_document(metadata={"author": "Example Author"})
    assert doc.id == "doc-1"
    assert doc.source == "local"
    assert doc.content == b"Hello, world!"
    assert doc.last_modified == datetime(2026, 1, 1, tzinfo=UTC)
    assert doc.metadata == {"author": "Example Author"}


def test_metadata_defaults_to_empty():
    doc = make_document()
    assert doc.metadata == {}


def test_metadata_not_shared_between_documents():
    first = make_document()
    second = make_document(id="doc-2")
    first.metadata["tag"] = "changed"
    assert second.metadata == {}
