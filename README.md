# rag-research-agent

A Python RAG and research agent for company knowledge. It will pull from
document storage (AWS S3), Jira, ServiceNow and SearXNG web search through
pluggable connectors, with a configurable LLM and embedding provider.

Status: early skeleton. Only configuration loading exists so far.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

## Install

```bash
uv sync
cp .env.example .env   # fill in values locally; .env is git-ignored
```

## Run tests and lint

```bash
uv run pytest
uv run ruff check
uv run ruff format --check
```


