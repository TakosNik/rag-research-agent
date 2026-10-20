from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

from rag_research_agent.connectors import Connector
from rag_research_agent.models import SourceDocument


class FakeConnector(Connector):
    """Tiny in-memory connector that holds one document."""

    def __init__(self):
        self.document = SourceDocument(
            id="doc-1",
            source="fake",
            title="example.txt",
            uri="fake://example.txt",
            content=b"Hello, world!",
            mime_type="text/plain",
            last_modified=datetime(2026, 1, 1, tzinfo=UTC),
        )

    @property
    def name(self) -> str:
        return "fake"

    def list_documents(self, since: datetime | None = None) -> Iterator[SourceDocument]:
        if since is None or self.document.last_modified > since:
            yield self.document


def test_connector_cannot_be_created_directly():
    with pytest.raises(TypeError):
        Connector()


def test_fake_connector_has_name():
    assert FakeConnector().name == "fake"


def test_fake_connector_lists_all_documents_without_since():
    docs = list(FakeConnector().list_documents())
    assert [doc.id for doc in docs] == ["doc-1"]


def test_fake_connector_respects_since():
    connector = FakeConnector()
    before = datetime(2025, 12, 31, tzinfo=UTC)
    after = datetime(2026, 1, 2, tzinfo=UTC)
    assert [doc.id for doc in connector.list_documents(since=before)] == ["doc-1"]
    assert list(connector.list_documents(since=after)) == []
