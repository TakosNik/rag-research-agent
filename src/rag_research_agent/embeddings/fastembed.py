"""A real embedder that runs a small model locally with FastEmbed (no API key needed)."""

from typing import TYPE_CHECKING

from rag_research_agent.embeddings.base import Embedder

if TYPE_CHECKING:
    from fastembed import TextEmbedding

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class FastEmbedEmbedder(Embedder):
    """Embeds text with a FastEmbed model running on this machine (CPU is fine).

    The model is loaded the first time it is needed, not when this object is
    created. Loading can download the model (tens to hundreds of MB, cached
    afterwards) and takes a moment, so creating the embedder and importing
    this module stay fast.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self._model_name = model_name
        self._model: TextEmbedding | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._get_model().embedding_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # passage_embed adds whatever document prefix the model expects
        return [vector.tolist() for vector in self._get_model().passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        # query_embed adds whatever query prefix the model expects
        (vector,) = self._get_model().query_embed([text])
        return vector.tolist()

    def _get_model(self) -> "TextEmbedding":
        if self._model is None:
            # Imported here so importing this module doesn't load FastEmbed either
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self._model_name)
        return self._model
