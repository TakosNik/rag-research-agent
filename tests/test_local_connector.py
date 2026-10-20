import os
from datetime import UTC, datetime

import pytest

from rag_research_agent.connectors import LocalFolderConnector


def write(path, content=b"example"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def ids(connector, **kwargs):
    return [doc.id for doc in connector.list_documents(**kwargs)]


def test_name_is_local(tmp_path):
    assert LocalFolderConnector(tmp_path).name == "local"


def test_finds_files_in_subfolders_with_relative_ids(tmp_path):
    write(tmp_path / "b.txt")
    write(tmp_path / "sub" / "deeper" / "a.txt")
    assert ids(LocalFolderConnector(tmp_path)) == ["b.txt", "sub/deeper/a.txt"]


def test_document_fields(tmp_path):
    path = write(tmp_path / "notes" / "hello.txt", b"Hello, world!")
    [doc] = LocalFolderConnector(tmp_path).list_documents()
    assert doc.id == "notes/hello.txt"
    assert doc.source == "local"
    assert doc.title == "hello.txt"
    assert doc.uri == str(path.resolve())
    assert doc.content == b"Hello, world!"
    assert doc.last_modified.tzinfo is not None


def test_skips_hidden_files_and_folders(tmp_path):
    write(tmp_path / "visible.txt")
    write(tmp_path / ".hidden.txt")
    write(tmp_path / ".git" / "config")
    write(tmp_path / "sub" / ".cache" / "file.txt")
    assert ids(LocalFolderConnector(tmp_path)) == ["visible.txt"]


def test_guesses_mime_types(tmp_path):
    write(tmp_path / "a.txt")
    write(tmp_path / "b.pdf")
    write(tmp_path / "c.unknownext")
    docs = LocalFolderConnector(tmp_path).list_documents()
    assert [doc.mime_type for doc in docs] == [
        "text/plain",
        "application/pdf",
        "application/octet-stream",
    ]


def test_since_filters_by_modification_time(tmp_path):
    old = write(tmp_path / "old.txt")
    new = write(tmp_path / "new.txt")
    old_time = datetime(2026, 1, 1, tzinfo=UTC).timestamp()
    new_time = datetime(2026, 6, 1, tzinfo=UTC).timestamp()
    os.utime(old, (old_time, old_time))
    os.utime(new, (new_time, new_time))
    connector = LocalFolderConnector(tmp_path)

    assert ids(connector) == ["new.txt", "old.txt"]
    assert ids(connector, since=datetime(2026, 3, 1, tzinfo=UTC)) == ["new.txt"]
    assert ids(connector, since=datetime(2026, 6, 1, tzinfo=UTC)) == []


def test_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        LocalFolderConnector(tmp_path / "missing")


def test_empty_folder_yields_nothing(tmp_path):
    assert ids(LocalFolderConnector(tmp_path)) == []
