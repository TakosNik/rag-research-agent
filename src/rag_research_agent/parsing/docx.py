"""Parser for Word (.docx) files."""

from io import BytesIO
from zipfile import BadZipFile

from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from docx.table import Table

# Separates paragraphs and tables in the extracted text
BLOCK_SEPARATOR = "\n\n"
# Separates the cells of one table row
CELL_SEPARATOR = " | "


class DocxParsingError(Exception):
    """Raised when bytes are not a valid .docx file."""


def parse_docx(content: bytes) -> str:
    """Extract paragraphs and tables of a .docx file, in document order.

    Paragraphs are separated by a blank line and empty ones are skipped.
    Each table row becomes one line with its cells joined by " | ".
    """
    try:
        document = Document(BytesIO(content))
    except (BadZipFile, KeyError, ValueError, PackageNotFoundError) as e:
        raise DocxParsingError(f"File is broken or not a valid .docx file: {e}") from e

    blocks = []
    for item in document.iter_inner_content():
        text = _table_to_text(item) if isinstance(item, Table) else item.text.strip()
        if text:
            blocks.append(text)
    return BLOCK_SEPARATOR.join(blocks)


def _table_to_text(table: Table) -> str:
    """Write each row as one line, skipping rows where every cell is empty."""
    lines = []
    for row in table.rows:
        cells = []
        for cell in row.cells:
            # A merged cell is returned once per column it spans; keep it once
            if cells and cell._tc is cells[-1]._tc:
                continue
            cells.append(cell)
        texts = [cell.text.strip() for cell in cells]
        if any(texts):
            lines.append(CELL_SEPARATOR.join(texts))
    return "\n".join(lines)
