"""Reranker interface (Phase 6 placeholder).

Phase 6 will implement this against a local cross-encoder (BGE
reranker, Jina reranker, or similar). The contract is captured
here so hybrid retrieval (Phase 5) can plan for it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class Reranker(ABC):
    """Abstract cross-encoder reranker."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier (e.g. ``BAAI/bge-reranker-base``)."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """Return ``(index_into_documents, score)`` pairs ranked by relevance.

        ``index_into_documents`` is the position in the input list.
        If ``top_k`` is given, only the top-k results are returned.
        """
