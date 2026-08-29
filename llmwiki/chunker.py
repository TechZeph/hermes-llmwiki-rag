"""Structural chunking of markdown documents (Phase 2 placeholder).

Phase 1 does not chunk; it indexes whole documents. Phase 2 will
split each document into structural chunks that preserve heading
hierarchy, section names, and the relationships to the parent
document.

This module exists in Phase 1 only as a no-op stub so the package
shape matches the plan. The real implementation lands in Phase 2.
"""

from __future__ import annotations

from .models import Chunk, Document

__all__ = ["Chunk", "Document", "chunk_document"]


def chunk_document(doc: Document, *, max_chunk_chars: int = 2000) -> list[Chunk]:
    """Placeholder. Returns a single whole-document chunk.

    Phase 2 will replace this with a real structural chunker that
    splits at heading boundaries, preserves heading paths, and
    never splits a paragraph in half unnecessarily.
    """
    return [
        Chunk(
            id=None,
            document_id=doc.id if doc.id is not None else 0,
            heading_path=(),
            text="",  # real text lives in the document; Phase 2 reads the body
            position=0,
        )
    ]
