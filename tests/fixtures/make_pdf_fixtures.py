"""Generate the small PDF files used by tests/test_parsing_pdf.py.

The PDFs are committed, so this only needs to run again if they change:

    uv run python tests/fixtures/make_pdf_fixtures.py
"""

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen.canvas import Canvas

FIXTURES = Path(__file__).parent


def make_two_pages() -> bytes:
    """A normal PDF with one line of text on each of its two pages."""
    buffer = BytesIO()
    canvas = Canvas(buffer, invariant=True)  # invariant: same bytes on every run
    canvas.drawString(72, 720, "Hello from page one")
    canvas.showPage()
    canvas.drawString(72, 720, "Hello from page two")
    canvas.showPage()
    canvas.save()
    return buffer.getvalue()


def make_no_text() -> bytes:
    """A PDF with only a drawn shape and no text, like a scanned page."""
    buffer = BytesIO()
    canvas = Canvas(buffer, invariant=True)
    canvas.rect(72, 600, 200, 100, fill=1)
    canvas.showPage()
    canvas.save()
    return buffer.getvalue()


def make_encrypted() -> bytes:
    """The two-page PDF, locked with a (dummy) password."""
    writer = PdfWriter(clone_from=PdfReader(BytesIO(make_two_pages())))
    writer.encrypt("dummy-password", algorithm="RC4-128")
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


if __name__ == "__main__":
    (FIXTURES / "two_pages.pdf").write_bytes(make_two_pages())
    (FIXTURES / "no_text.pdf").write_bytes(make_no_text())
    (FIXTURES / "encrypted.pdf").write_bytes(make_encrypted())
