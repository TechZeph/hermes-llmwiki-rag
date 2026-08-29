"""Lexical (BM25) retrieval interface (Phase 4 placeholder).

Phase 4 will implement this against SQLite FTS5. The contract is
captured here so the rest of the RAG has a stable target.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class LexicalIndex(ABC):
    """Abstract lexical index (BM25 over chunk text)."""

    @abstractmethod
    def upsert(self, ids: Sequence[int], texts: Sequence[str]) -> None:
        """Insert or update chunk text by row id. Idempotent."""

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Return ``(id, bm25_score)`` pairs ranked by relevance (descending)."""

    @abstractmethod
    def delete(self, ids: Sequence[int]) -> None:
        """Remove documents from the index. No-op for unknown ids."""


__all__ = ["LexicalIndex"]
