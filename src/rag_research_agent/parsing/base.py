"""Pick the right parser for a document based on its mime type."""

from collections.abc import Callable

from rag_research_agent.models import SourceDocument
from rag_research_agent.parsing.text import parse_text


class UnsupportedFileTypeError(Exception):
    """Raised when no parser is registered for a document's mime type."""

    def __init__(self, mime_type: str):
        self.mime_type = mime_type
        super().__init__(f"No parser for mime type: {mime_type}")


# A parser takes a file's raw bytes and returns its text
Parser = Callable[[bytes], str]

# Registry of supported formats: mime type -> parser. Add new formats here.
PARSERS: dict[str, Parser] = {
    "text/plain": parse_text,
    "text/markdown": parse_text,
}


def extract_text(doc: SourceDocument) -> str:
    """Return the text of a document, using the parser for its mime type."""
    parser = PARSERS.get(doc.mime_type)
    if parser is None:
        raise UnsupportedFileTypeError(doc.mime_type)
    return parser(doc.content)
