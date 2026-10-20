"""Shared data models used across connectors and the rest of the pipeline."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    """One document fetched from a source, in a common shape for every connector."""

    # Unique ID of the document within its source
    id: str
    # Which connector it came from, e.g. "local" or "s3"
    source: str
    # Human-readable title, e.g. the file name
    title: str
    # Where the document lives, e.g. a file path or s3:// URL
    uri: str
    # Raw file contents, kept as bytes so binary formats (PDF, DOCX) also fit
    content: bytes
    # Type of the content, e.g. "text/plain" or "application/pdf"
    mime_type: str
    # When the document was last changed at the source
    last_modified: datetime
    # Any extra source-specific details (author, tags, ...); empty by default
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A small piece of a document's text, the unit we later search over."""

    # Unique ID of the chunk: "<document id>#<index>"
    id: str
    # ID of the document this chunk was cut from
    document_id: str
    # Source, title and uri are copied from the document, so a search result
    # can show where it came from without looking the document up again
    source: str
    title: str
    uri: str
    # Position of the chunk in its document, starting at 0
    index: int
    # The chunk's text
    text: str
    # Extra details copied from the document; empty by default
    metadata: dict[str, Any] = Field(default_factory=dict)
