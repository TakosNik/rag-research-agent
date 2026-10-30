"""Command-line tool: `rag ingest <folder>` and `rag search "<question>"`."""

import logging
from pathlib import Path
from typing import Annotated

import typer

from rag_research_agent.config import Settings
from rag_research_agent.connectors import LocalFolderConnector
from rag_research_agent.embeddings import get_embedder
from rag_research_agent.ingest import ingest
from rag_research_agent.vectorstores import get_vector_store

# How much of each result's text `rag search` shows
PREVIEW_LENGTH = 200

app = typer.Typer(help="Index company documents and search them.", no_args_is_help=True)


@app.callback()
def main() -> None:
    # Show warnings (e.g. skipped documents) from the pipeline on stderr
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


@app.command("ingest")
def ingest_command(
    folder: Annotated[Path, typer.Argument(help="Folder to index.", exists=True, file_okay=False)],
) -> None:
    """Index every file in a local folder (and its subfolders)."""
    settings = Settings()
    embedder = get_embedder(settings)
    store = get_vector_store(settings, embedder)

    report = ingest(LocalFolderConnector(folder), embedder, store)

    typer.echo(f"Documents seen:     {report.documents_seen}")
    typer.echo(f"Documents ingested: {report.documents_ingested}")
    typer.echo(f"Chunks stored:      {report.chunks_stored}")
    typer.echo(f"Skipped:            {len(report.skipped)}")
    for skipped in report.skipped:
        typer.echo(f"  - {skipped.document_id}: {skipped.reason}")


@app.command("search")
def search_command(
    question: Annotated[str, typer.Argument(help="What to search for.")],
    top_k: Annotated[int, typer.Option(min=1, help="How many results to show.")] = 5,
    source: Annotated[
        str | None, typer.Option(help='Only search one source, e.g. "local".')
    ] = None,
) -> None:
    """Find the stored chunks most similar to a question."""
    settings = Settings()
    embedder = get_embedder(settings)
    store = get_vector_store(settings, embedder)

    results = store.search(embedder.embed_query(question), top_k=top_k, source=source)

    if not results:
        typer.echo("No results.")
        return
    for number, result in enumerate(results, start=1):
        chunk = result.chunk
        page = chunk.metadata.get("page")
        location = f" (page {page})" if page is not None else ""
        typer.echo(f"{number}. [{result.score:.3f}] {chunk.title}{location}")
        typer.echo(f"   {_preview(chunk.text)}")
        typer.echo()


def _preview(text: str) -> str:
    """The start of a chunk's text on one line, cut to about PREVIEW_LENGTH characters."""
    text = " ".join(text.split())
    if len(text) <= PREVIEW_LENGTH:
        return text
    return text[:PREVIEW_LENGTH].rstrip() + "..."
