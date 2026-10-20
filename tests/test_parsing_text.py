from datetime import UTC, datetime

import pytest

from rag_research_agent.models import SourceDocument
from rag_research_agent.parsing import UnsupportedFileTypeError, extract_text


def make_document(content, mime_type="text/plain"):
    return SourceDocument(
        id="doc-1",
        source="local",
        title="example",
        uri="/tmp/example",
        content=content,
        mime_type=mime_type,
        last_modified=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_plain_text():
    assert extract_text(make_document(b"Hello, world!")) == "Hello, world!"


def test_markdown():
    content = b"# Title\n\nSome *text*"
    assert extract_text(make_document(content, "text/markdown")) == "# Title\n\nSome *text*"


def test_utf8_non_ascii():
    assert extract_text(make_document("café".encode())) == "café"


def test_non_utf8_bytes_fall_back_to_latin1():
    assert extract_text(make_document(b"caf\xe9")) == "café"


def test_windows_line_endings_are_normalised():
    assert extract_text(make_document(b"line one\r\nline two\r\n")) == "line one\nline two\n"


def test_unknown_mime_type_raises():
    with pytest.raises(UnsupportedFileTypeError, match="image/png"):
        extract_text(make_document(b"\x89PNG", "image/png"))
