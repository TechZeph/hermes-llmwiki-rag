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

    Phase 3 adds ``embeddings_built`` and ``embeddings_rebuilt`` —
    counts of vector rows written by the run. ``built`` covers new
    vectors from added/updated chunks; ``rebuilt`` covers chunks
    whose embedding was rewritten because the configured embedding
    model changed since last run.
    """

    mode: str  # "incremental" | "full"
    documents_seen: int = 0
    documents_added: int = 0
    documents_updated: int = 0
    documents_removed: int = 0
    documents_skipped: int = 0
    chunks_added: int = 0
    chunks_updated: int = 0
    chunks_removed: int = 0
    embeddings_built: int = 0
    embeddings_rebuilt: int = 0
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Chunk:
    """A structural chunk of a document.

    Attributes
    ----------
    id:
        Database row id. ``None`` for unsaved chunks.
    document_id:
        Row id of the parent :class:`Document`.
    heading_path:
        Breadcrumb of heading names, starting with the document
        title. Empty tuple only for documents with no title and no
        headings. Example: ``("Hosp-core", "Plan", "Phase 0")``.
    section_name:
        The text of the most specific heading that owns this chunk
        (the last element of ``heading_path``). Empty string if the
        chunk is a preamble.
    text:
        The chunk's body text. The chunker guarantees that text is
        never split mid-paragraph (paragraphs are atomic).
    position:
        Ordinal within the parent document, starting at 0. Used to
        preserve document order for retrieval.
    """

    id: int | None
    document_id: int
    heading_path: tuple[str, ...]
    section_name: str
    text: str
    position: int


__all__ = ["Chunk", "Document", "IndexRunStats"]
