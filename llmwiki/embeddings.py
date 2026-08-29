"""Embedding interface (Phase 3 placeholder).

Phase 1 only needs the interface stub so the package shape matches
the plan. The real implementation in Phase 3 will wrap FastEmbed
and expose a small, typed surface that the rest of the RAG depends on.

Design decisions captured here so Phase 3 has a stable contract:

- The interface is a class, not a function. We may need to hold
  state (model handle, cache, batch settings) across calls.
- The interface returns numpy arrays, not Python lists. The vector
  store wrapper can convert as needed but the cost of converting
  back-and-forth on every call is real.
- We return embeddings with shape ``(dim,)`` for a single text
  and ``(n, dim)`` for a batch. No weird in/out shape games.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class Embedder(ABC):
    """Abstract embedding backend.

    Implementations: :class:`FastEmbedEmbedder` (Phase 3), and
    later a local-server backend (Ollama) and an OpenAI-compatible
    backend for benchmarking.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier (e.g. ``BAAI/bge-small-en-v1.5``)."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the embedding vectors this embedder produces."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per input.

        The order of the returned list matches the order of the input.
        Empty input returns an empty list.
        """


__all__ = ["Embedder"]
