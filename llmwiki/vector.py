"""Vector store interface (Phase 3 placeholder).

Phase 3 will implement this against ``sqlite-vec``. The contract is
captured here so the rest of the RAG has a stable target.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class VectorStore(ABC):
    """Abstract vector store.

    Implementations: :class:`SqliteVecStore` (Phase 3). Benchmarking
    may add a ``NumpyMemmapStore`` for sanity testing.
    """

    @abstractmethod
    def upsert(self, ids: Sequence[int], vectors: Sequence[Sequence[float]]) -> None:
        """Insert or update vectors by row id. Idempotent."""

    @abstractmethod
    def search(self, query: Sequence[float], top_k: int) -> list[tuple[int, float]]:
        """Return ``(id, score)`` pairs ranked by similarity (descending)."""

    @abstractmethod
    def delete(self, ids: Sequence[int]) -> None:
        """Remove vectors by id. No-op for unknown ids."""


__all__ = ["VectorStore"]
