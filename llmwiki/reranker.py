"""Cross-encoder reranking interface and the local FastEmbed implementation.

Reranking is conditional: :class:`FastEmbedReranker` exists so the
evaluation harness can measure it, and it is enabled in production only
when the held-out gate in ``docs/evaluation.md`` passes.
"""

from __future__ import annotations

import os
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
        """Return ``(index_into_documents, score)`` pairs, best first.

        ``index_into_documents`` is the position in the input list. If
        ``top_k`` is given, only the top-k results are returned. Scores
        are model logits; larger is more relevant. They are comparable
        within one call only.
        """


class FastEmbedReranker(Reranker):
    """Local cross-encoder wrapping ``fastembed.rerank.cross_encoder.TextCrossEncoder``."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        *,
        cache_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self._model_name = model_name
        if cache_path is not None:
            os.environ.setdefault("FASTEMBED_CACHE_PATH", str(cache_path))
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._impl = TextCrossEncoder(model_name=model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        if not documents:
            return []
        scores = [float(s) for s in self._impl.rerank(query, list(documents))]
        ranked = sorted(enumerate(scores), key=lambda t: (-t[1], t[0]))
        if top_k is not None:
            ranked = ranked[: max(top_k, 0)]
        return ranked


__all__ = ["FastEmbedReranker", "Reranker"]
