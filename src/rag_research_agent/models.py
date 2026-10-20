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
