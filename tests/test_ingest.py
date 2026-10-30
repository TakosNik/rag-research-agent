import logging
import shutil
from pathlib import Path

from rag_research_agent.connectors import LocalFolderConnector
from rag_research_agent.embeddings import FakeEmbedder
from rag_research_agent.ingest import IngestReport, SkippedDocument, ingest
from rag_research_agent.vectorstores import InMemoryVectorStore

FIXTURES = Path(__file__).parent / "fixtures"

embedder = FakeEmbedder()

# Long enough to be cut into several chunks
LONG_TEXT = "\n\n".join(
    f"Paragraph {i} explains how the VPN client connects to the office network. " * 4
    for i in range(10)
)


def run(folder, store):
    return ingest(LocalFolderConnector(folder), embedder, store)


def search(store, question):
    return store.search(embedder.embed_query(question), top_k=50)


def test_documents_are_stored(tmp_path):
    (tmp_path / "vpn.txt").write_text("Enable the VPN before connecting.")
    (tmp_path / "guides").mkdir()
    (tmp_path / "guides" / "password.md").write_text("Reset your password in settings.")
    store = InMemoryVectorStore()

    report = run(tmp_path, store)

    assert report == IngestReport(documents_seen=2, documents_ingested=2, chunks_stored=2)
    assert store.count() == 2
    [result, *_] = search(store, "reset password")
    assert result.chunk.document_id == "guides/password.md"


def test_rerunning_does_not_duplicate_chunks(tmp_path):
    (tmp_path / "vpn.txt").write_text(LONG_TEXT)
    store = InMemoryVectorStore()

    first = run(tmp_path, store)
    second = run(tmp_path, store)

    assert first.chunks_stored > 1
    assert second == first
    assert store.count() == first.chunks_stored


def test_changed_file_replaces_its_old_chunks(tmp_path):
    path = tmp_path / "vpn.txt"
    path.write_text(LONG_TEXT)
    (tmp_path / "other.txt").write_text("The kitchen is cleaned on Friday.")
    store = InMemoryVectorStore()
    first = run(tmp_path, store)

    path.write_text("The VPN was replaced by a new remote access tool.")
    second = run(tmp_path, store)

    assert first.chunks_stored > second.chunks_stored == 2
    assert store.count() == 2
    texts = {result.chunk.text for result in search(store, "VPN")}
    assert texts == {
        "The VPN was replaced by a new remote access tool.",
        "The kitchen is cleaned on Friday.",
    }


def test_unsupported_file_is_skipped_with_a_reason(tmp_path, caplog):
    (tmp_path / "vpn.txt").write_text("Enable the VPN before connecting.")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG example bytes")
    store = InMemoryVectorStore()

    with caplog.at_level(logging.WARNING):
        report = run(tmp_path, store)

    assert report.documents_seen == 2
    assert report.documents_ingested == 1
    assert report.skipped == [
        SkippedDocument(document_id="logo.png", reason="unsupported file type: image/png")
    ]
    assert "logo.png" in caplog.text
    assert store.count() == 1


def test_broken_file_is_skipped_and_the_run_continues(tmp_path):
    (tmp_path / "broken.pdf").write_bytes(b"this is not really a PDF")
    (tmp_path / "vpn.txt").write_text("Enable the VPN before connecting.")
    store = InMemoryVectorStore()

    report = run(tmp_path, store)

    assert report.documents_ingested == 1
    [skipped] = report.skipped
    assert skipped.document_id == "broken.pdf"
    assert skipped.reason.startswith("text extraction failed")


def test_documents_without_text_are_skipped(tmp_path):
    shutil.copy(FIXTURES / "no_text.pdf", tmp_path / "scanned.pdf")
    (tmp_path / "empty.txt").write_text("  \n\n  ")
    store = InMemoryVectorStore()

    report = run(tmp_path, store)

    assert report.documents_ingested == 0
    assert [skipped.document_id for skipped in report.skipped] == ["empty.txt", "scanned.pdf"]
    assert all("no text found" in skipped.reason for skipped in report.skipped)
    assert store.count() == 0


def test_document_that_loses_its_text_has_old_chunks_removed(tmp_path):
    path = tmp_path / "vpn.txt"
    path.write_text("Enable the VPN before connecting.")
    store = InMemoryVectorStore()
    run(tmp_path, store)

    path.write_text("")
    report = run(tmp_path, store)

    assert [skipped.document_id for skipped in report.skipped] == ["vpn.txt"]
    assert store.count() == 0
