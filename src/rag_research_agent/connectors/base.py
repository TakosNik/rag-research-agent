"""The shared interface that every connector implements."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import datetime

from rag_research_agent.models import SourceDocument


class Connector(ABC):
    """Base class for all connectors.

    A connector knows how to talk to one source (e.g. S3, Jira or ServiceNow)
    and turns whatever that source stores into SourceDocument objects. The rest
    of the pipeline only uses this interface, so it never needs to know which
    source a document came from.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, unique name of the source, e.g. "s3" or "jira"."""

    @abstractmethod
    def list_documents(self, since: datetime | None = None) -> Iterator[SourceDocument]:
        """Yield the documents from this source, one at a time.

        If `since` is given, only yield documents changed after that time
        (their `last_modified` is later than `since`). This lets us fetch just
        the new or updated documents instead of everything again. If `since`
        is None, yield all documents.
        """
