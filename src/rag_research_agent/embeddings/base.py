"""The shared interface that every embedder implements, plus a similarity helper."""

import math
from abc import ABC, abstractmethod


class Embedder(ABC):
    """Base class for all embedders.

    An embedder turns text into a vector: a fixed-length list of numbers that
    describes the text. Texts with similar meaning get vectors that point in a
    similar direction, so we can find chunks related to a question by
    comparing vectors (see `cosine_similarity`).

    Documents and queries have separate methods because some models treat them
    differently. A chunk is usually a longer passage and a question is short,
    and some models expect a different instruction or prefix for each (e.g.
    "passage: ..." vs "query: ..."), or even use two different models. Keeping
    the methods apart lets each embedder do the right thing, while models that
    don't care can simply use the same code for both.
    """

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Length of every vector this embedder returns."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per text, in the same order, for storing and searching.

        Use this for the chunks we index. Taking a list lets real models embed
        many chunks in one request, which is much faster than one at a time.
        An empty list gives an empty list.
        """

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Return the vector for a user's question, to compare against document vectors."""


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """How closely two vectors point in the same direction, from -1 to 1.

    1 means the same direction (very similar), 0 means unrelated and -1 means
    opposite. Only the direction counts, not the length of the vectors. If
    either vector is all zeros it has no direction, so we return 0.0.
    """
    if len(a) != len(b):
        raise ValueError(f"vectors have different lengths: {len(a)} and {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
