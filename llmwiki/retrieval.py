"""End-to-end retrieval (Phase 5+ placeholder)."""

from __future__ import annotations

from .models import Chunk, Document


def retrieve(query: str, *, top_k: int = 10) -> list[tuple[Document, Chunk]]:
    """Placeholder. Returns no results.

    Phase 5+ wires together the embedder, vector store, lexical index,
    hybrid fusion, reranker, and graph expander behind this single
    entry point. Tests will mock the components.
    """
    return []
