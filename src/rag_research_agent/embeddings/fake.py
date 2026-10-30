"""A fake embedder for tests and offline development. It does NOT understand meaning."""

import hashlib
import math
import re

from rag_research_agent.embeddings.base import Embedder

WORD = re.compile(r"\w+")


class FakeEmbedder(Embedder):
    """Turns text into a vector by counting words, with no model and no network.

    Each word is hashed into one of `dimensions` buckets and the bucket counts
    form the vector, which is then scaled to length 1. The same text always
    gives the same vector, and texts that share words get more similar vectors.

    This only matches words, it does NOT understand meaning: "car" and
    "automobile" look unrelated to it. Use it only for tests and offline
    development, never as a real embedder.
    """

    def __init__(self, dimensions: int = 256):
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return "fake"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for word in WORD.findall(text.lower()):
            vector[self._bucket(word)] += 1.0
        length = math.sqrt(sum(x * x for x in vector))
        if length == 0:
            # No words, e.g. an empty string: nothing to normalise
            return vector
        return [x / length for x in vector]

    def _bucket(self, word: str) -> int:
        # Python's built-in hash() changes between runs, so use a stable hash
        digest = hashlib.sha256(word.encode()).digest()
        return int.from_bytes(digest[:8], "big") % self._dimensions
