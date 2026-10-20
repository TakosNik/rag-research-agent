"""Parsing turns a SourceDocument's raw bytes into plain text."""

from rag_research_agent.parsing.base import UnsupportedFileTypeError, extract_text

__all__ = ["UnsupportedFileTypeError", "extract_text"]
