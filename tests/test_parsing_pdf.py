from datetime import UTC, datetime
from pathlib import Path

import pytest

from rag_research_agent.models import SourceDocument
from rag_research_agent.parsing import PdfParsingError, extract_text

FIXTURES = Path(__file__).parent / "fixtures"


def make_pdf_document(content):
    return SourceDocument(
        id="doc-1",
        source="local",
        title="example",
        uri="/tmp/example.pdf",
        content=content,
        mime_type="application/pdf",
        last_modified=datetime(2026, 1, 1, tzinfo=UTC),
    )


def read_fixture(name):
    return (FIXTURES / name).read_bytes()


def test_two_page_pdf_text_is_split_by_form_feed():
    text = extract_text(make_pdf_document(read_fixture("two_pages.pdf")))

    pages = text.split("\f")
    assert len(pages) == 2
    assert pages[0].strip() == "Hello from page one"
    assert pages[1].strip() == "Hello from page two"


def test_pdf_without_text_returns_empty_string():
    assert extract_text(make_pdf_document(read_fixture("no_text.pdf"))) == ""


def test_broken_pdf_raises_clear_error():
    with pytest.raises(PdfParsingError, match="broken"):
        extract_text(make_pdf_document(b"this is not a pdf"))


def test_encrypted_pdf_raises_clear_error():
    with pytest.raises(PdfParsingError, match="encrypted"):
        extract_text(make_pdf_document(read_fixture("encrypted.pdf")))
