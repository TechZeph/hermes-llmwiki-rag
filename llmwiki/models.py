"""Dataclasses shared across the RAG core.

These are the value types that flow between components. They are
deliberately small (no behaviour, only data) so they can be pickled,
serialised, and reasoned about without dragging dependencies across
the package.

Phase 1 only uses :class:`Document` and :class:`IndexRunStats`.
:class:`Chunk` lands in Phase 2. The other types are placeholders so
the interface stubs in this package are honest about the shape of
future data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Document:
    """A single indexed markdown file.

    The ``frontmatter``, ``tags``, ``wikilinks``, ``aliases`` and
    ``headings`` fields mirror the corresponding columns in the
    ``documents`` table. They are stored as JSON in the database
    and as Python lists/dicts here.
    """

    id: int | None
    path: str
    absolute_path: str
    title: str
    mtime_ns: int
    size_bytes: int
    content_hash: str
    frontmatter: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    wikilinks: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    headings: tuple[dict[str, Any], ...] = ()  # [{"level": int, "text": str}, ...]


@dataclass(frozen=True, slots=True)
class IndexRunStats:
    """Summary of one indexing run.

    Returned by :class:`llmwiki.indexer.Indexer.run` so the CLI can
    report what happened and the eval harness can later correlate
    runs with retrieval quality.
    """

    mode: str  # "full" | "incremental"
    documents_seen: int = 0
    documents_added: int = 0
    documents_updated: int = 0
    documents_removed: int = 0
    documents_skipped: int = 0
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Chunk:
    """A structural chunk of a document (Phase 2 placeholder)."""

    id: int | None
    document_id: int
    heading_path: tuple[str, ...]  # e.g. ("Hosp-core", "Plan", "Phase 0")
    text: str
    position: int  # ordinal within the document


__all__ = ["Chunk", "Document", "IndexRunStats"]
