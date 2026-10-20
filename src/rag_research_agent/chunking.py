"""Split a document's text into small, overlapping chunks."""

import re

from rag_research_agent.models import Chunk, SourceDocument

# One or more blank lines between paragraphs
PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
PARAGRAPH_SEPARATOR = "\n\n"

# Ways to split a paragraph that is too long, from best to worst. Each entry
# is (separator used when gluing pieces back together, where to split).
SPLITTERS = [
    (" ", re.compile(r"(?<=[.!?])\s+")),  # sentence ends
    ("\n", re.compile(r"\n")),  # line breaks, e.g. table rows
    (" ", re.compile(r"\s+")),  # between words
]


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Split text into chunks of at most `chunk_size` characters.

    Paragraphs are packed together while they fit. Each chunk after the first
    starts with roughly the last `overlap` characters of the previous chunk.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be at least 0 and smaller than chunk_size")

    # Leave room for the overlap, so every piece fits into a chunk after it
    pieces = _split_into_pieces(text, max_length=chunk_size - overlap)

    chunks: list[str] = []
    current = ""
    for separator, piece in pieces:
        if not current:
            current = piece
        elif len(current) + len(separator) + len(piece) <= chunk_size:
            current += separator + piece
        else:
            chunks.append(current)
            room = chunk_size - len(separator) - len(piece)
            tail = _tail(current, min(overlap, room))
            current = tail + separator + piece if tail else piece
    if current:
        chunks.append(current)
    return chunks


def chunk_document(doc: SourceDocument, text: str) -> list[Chunk]:
    """Split a document's extracted text into Chunk objects."""
    return [
        Chunk(
            id=f"{doc.id}#{index}",
            document_id=doc.id,
            source=doc.source,
            title=doc.title,
            uri=doc.uri,
            index=index,
            text=chunk,
            metadata=dict(doc.metadata),
        )
        for index, chunk in enumerate(chunk_text(text))
    ]


def _split_into_pieces(text: str, max_length: int) -> list[tuple[str, str]]:
    """Split text into (separator, piece) pairs, each piece at most `max_length` long.

    The separator is what goes between the piece and the one before it.
    """
    # Treat page breaks as paragraph breaks for now
    text = text.replace("\f", PARAGRAPH_SEPARATOR)
    pieces = []
    for paragraph in PARAGRAPH_BREAK.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        parts = _split_long(paragraph, max_length)
        pieces.append((PARAGRAPH_SEPARATOR, parts[0][1]))
        pieces.extend(parts[1:])
    return pieces


def _split_long(text: str, max_length: int, level: int = 0) -> list[tuple[str, str]]:
    """Split text that is too long using SPLITTERS, cutting mid-word as a last resort."""
    if len(text) <= max_length:
        return [("", text)]
    if level == len(SPLITTERS):
        # Last resort: cut every max_length characters, even mid-word.
        # The pieces are glued back with "" since they were one word.
        return [("", text[i : i + max_length]) for i in range(0, len(text), max_length)]

    separator, pattern = SPLITTERS[level]
    pieces = []
    for part in pattern.split(text):
        part = part.strip()
        if not part:
            continue
        for i, (inner_separator, piece) in enumerate(_split_long(part, max_length, level + 1)):
            pieces.append((inner_separator if i else separator, piece))
    return pieces


def _tail(text: str, length: int) -> str:
    """Return roughly the last `length` characters of text, starting at a whole word."""
    if length <= 0:
        return ""
    if len(text) <= length:
        return text
    tail = text[-length:]
    if not text[-length - 1].isspace():
        # The cut landed mid-word: start at the next word instead
        space = re.search(r"\s", tail)
        if space:
            tail = tail[space.end() :]
    return tail.strip()
