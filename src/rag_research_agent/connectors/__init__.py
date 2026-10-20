"""Connectors fetch documents from a source (S3, Jira, ...) as SourceDocuments."""

from rag_research_agent.connectors.base import Connector
from rag_research_agent.connectors.local import LocalFolderConnector

__all__ = ["Connector", "LocalFolderConnector"]
