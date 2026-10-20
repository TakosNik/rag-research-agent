"""Parser for PDF files."""

from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

# Separates pages in the extracted text, so page numbers can be recovered later
PAGE_SEPARATOR = "\f"


class PdfParsingError(Exception):
    """Raised when a PDF cannot be read because it is encrypted or broken."""


def parse_pdf(content: bytes) -> str:
    """Extract the text of a PDF, page by page, joined with form feeds.

    Returns an empty string if the PDF has no extractable text (e.g. a scanned
    PDF that only contains images).
    """
    try:
        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted and not reader.decrypt(""):
            # Some PDFs are encrypted with an empty password (e.g. only to
            # restrict printing) and can still be read; anything else can't.
            raise PdfParsingError("PDF is encrypted and needs a password")
        pages = [page.extract_text() for page in reader.pages]
    except PdfParsingError:
        raise
    except (PdfReadError, ValueError, KeyError, TypeError) as e:
        raise PdfParsingError(f"PDF is broken or not a valid PDF: {e}") from e

    if not any(page.strip() for page in pages):
        return ""
    return PAGE_SEPARATOR.join(pages)
