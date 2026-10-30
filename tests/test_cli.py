import pytest
from typer.testing import CliRunner

from rag_research_agent import cli
from rag_research_agent.vectorstores import InMemoryVectorStore

runner = CliRunner()


@pytest.fixture(autouse=True)
def offline_settings(monkeypatch):
    # Environment variables win over any local .env file
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("VECTOR_STORE", "memory")


@pytest.fixture
def shared_store(monkeypatch):
    # The memory store is empty in every new command, so share one between
    # `ingest` and `search` (the real pgvector store keeps data on its own)
    store = InMemoryVectorStore()
    monkeypatch.setattr(cli, "get_vector_store", lambda settings, embedder: store)
    return store


def test_ingest_prints_the_report(tmp_path):
    (tmp_path / "vpn.txt").write_text("Enable the VPN before connecting.")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG example bytes")

    result = runner.invoke(cli.app, ["ingest", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "Documents seen:     2" in result.output
    assert "Documents ingested: 1" in result.output
    assert "Chunks stored:      1" in result.output
    assert "logo.png: unsupported file type: image/png" in result.output


def test_ingest_missing_folder_fails(tmp_path):
    result = runner.invoke(cli.app, ["ingest", str(tmp_path / "missing")])

    assert result.exit_code != 0


def test_search_on_empty_store_says_no_results():
    result = runner.invoke(cli.app, ["search", "anything"])

    assert result.exit_code == 0, result.output
    assert "No results." in result.output


def test_search_prints_best_match_first_with_score_and_title(tmp_path, shared_store):
    (tmp_path / "password.txt").write_text("To reset your password, open the settings page.")
    (tmp_path / "kitchen.txt").write_text("The office kitchen is cleaned every Friday.")
    runner.invoke(cli.app, ["ingest", str(tmp_path)])

    result = runner.invoke(cli.app, ["search", "How do I reset my password?"])

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].startswith("1. [")
    assert "password.txt" in lines[0]
    assert "To reset your password" in lines[1]
    assert "2. [" in result.output


def test_search_top_k_and_source_options(tmp_path, shared_store):
    for i in range(4):
        (tmp_path / f"vpn{i}.txt").write_text(f"VPN guide number {i}.")
    runner.invoke(cli.app, ["ingest", str(tmp_path)])

    limited = runner.invoke(cli.app, ["search", "VPN", "--top-k", "2"])
    other_source = runner.invoke(cli.app, ["search", "VPN", "--source", "jira"])

    assert limited.output.count("vpn") == 2
    assert "No results." in other_source.output


def test_search_shortens_long_text(tmp_path, shared_store):
    (tmp_path / "long.txt").write_text("VPN " * 200)
    runner.invoke(cli.app, ["ingest", str(tmp_path)])

    result = runner.invoke(cli.app, ["search", "VPN"])

    preview = result.output.splitlines()[1].strip()
    assert preview.endswith("...")
    assert len(preview) <= cli.PREVIEW_LENGTH + 3
