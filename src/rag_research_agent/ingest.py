"""Ingestion: fetch documents, cut them into chunks, embed them and store them."""

import logging

from pydantic import BaseModel, Field

from rag_research_agent.chunking import chunk_document
from rag_research_agent.connectors import Connector
from rag_research_agent.embeddings import Embedder
from rag_research_agent.parsing import UnsupportedFileTypeError, extract_text
from rag_research_agent.vectorstores import VectorStore

logger = logging.getLogger(__name__)


class SkippedDocument(BaseModel):
    """A document that was not stored, and why."""

    document_id: str
    reason: str


class IngestReport(BaseModel):
    """What happened during one ingestion run."""

    # Documents the connector returned
    documents_seen: int = 0
    # Documents whose chunks are now in the store
    documents_ingested: int = 0
    # Chunks written to the store in this run
    chunks_stored: int = 0
    # Documents that were not stored, with the reason for each
    skipped: list[SkippedDocument] = Field(default_factory=list)


def ingest(connector: Connector, embedder: Embedder, store: VectorStore) -> IngestReport:
    """Put every document from `connector` into `store`, replacing older versions.

    For each document: extract its text, split it into chunks, embed all
    chunks in one call, delete the document's old chunks and upsert the new
    ones. Running this again on unchanged documents gives the same store
    contents, so it is safe to repeat.

    A document that can't be read (unsupported type, broken file) or has no
    text (e.g. a scanned PDF) is logged, recorded in `skipped`, and any old
    chunks of it are removed so searches don't return outdated text. Errors
    from the embedder or store stop the run, since they would affect every
    document.
    """
    report = IngestReport()
    for doc in connector.list_documents():
        report.documents_seen += 1

        try:
            text = extract_text(doc)
        except UnsupportedFileTypeError as e:
            _skip(report, store, doc.id, f"unsupported file type: {e.mime_type}")
            continue
        except Exception as e:
            # Parsers raise various errors for broken files; one bad file
            # shouldn't stop the whole run
            _skip(report, store, doc.id, f"text extraction failed: {e}")
            continue

        chunks = chunk_document(doc, text)
        if not chunks:
            _skip(report, store, doc.id, "no text found (e.g. a scanned PDF with only images)")
            continue

        # Embed before deleting, so if embedding fails the old chunks are still there
        vectors = embedder.embed_documents([chunk.text for chunk in chunks])
        store.delete_document(doc.id)
        store.upsert(chunks, vectors)

        report.documents_ingested += 1
        report.chunks_stored += len(chunks)
        logger.info("Ingested %s (%d chunks)", doc.id, len(chunks))
    return report


def _skip(report: IngestReport, store: VectorStore, document_id: str, reason: str) -> None:
    logger.warning("Skipped %s: %s", document_id, reason)
    report.skipped.append(SkippedDocument(document_id=document_id, reason=reason))
    store.delete_document(document_id)
