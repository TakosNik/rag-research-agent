"""Connector that reads files from a folder on the local disk."""

import mimetypes
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from rag_research_agent.connectors.base import Connector
from rag_research_agent.models import SourceDocument


class LocalFolderConnector(Connector):
    """Reads every file in a folder (and its subfolders) as a SourceDocument.

    Hidden files and anything inside hidden folders (names starting with ".")
    are skipped. File contents are returned as raw bytes; nothing is parsed.
    """

    def __init__(self, folder: str | Path):
        self.folder = Path(folder).resolve()
        if not self.folder.is_dir():
            raise FileNotFoundError(f"Folder does not exist: {self.folder}")

    @property
    def name(self) -> str:
        return "local"

    def list_documents(self, since: datetime | None = None) -> Iterator[SourceDocument]:
        """Yield one SourceDocument per file, sorted by relative path."""
        # rglob("*") finds everything below the folder; sort for a stable order
        for path in sorted(self.folder.rglob("*")):
            relative = path.relative_to(self.folder)
            if not path.is_file() or any(part.startswith(".") for part in relative.parts):
                continue

            last_modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
            if since is not None and last_modified <= since:
                continue

            mime_type, _ = mimetypes.guess_type(path.name)
            yield SourceDocument(
                id=relative.as_posix(),
                source=self.name,
                title=path.name,
                uri=str(path),
                content=path.read_bytes(),
                mime_type=mime_type or "application/octet-stream",
                last_modified=last_modified,
            )
