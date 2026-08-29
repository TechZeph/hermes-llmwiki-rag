"""Structural chunking of markdown documents (Phase 2).

The chunker splits a parsed document into chunks that preserve
heading hierarchy, section names, and the parent document's
identity. The strategy is **one chunk per heading section**:

- A "frontmatter / H1" chunk if the file has a title or frontmatter
  but no H1.
- A "preamble" chunk for content that appears before the first
  heading.
- One chunk per heading section, with the heading text and all
  body text up to the next heading of equal or higher level.

Within a section, if the text exceeds ``max_chunk_chars``, it is
greedily split at paragraph boundaries (blank lines) so we never
cut a paragraph in half. This is a safety net, not the primary
splitting rule.

A document with the structure::

    # Title
    intro
    ## A
    a body
    ## B
    b body
    ### B.1
    nested body

produces three chunks::

    Chunk(heading_path=("Title",), section_name="A", text="a body")
    Chunk(heading_path=("Title", "B"), section_name="B", text="b body")
    Chunk(heading_path=("Title", "B", "B.1"), section_name="B.1", text="nested body")

The "Title" segment is the parent document's title (or H1 text) so
that every chunk carries the breadcrumb it was found under.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from .models import Chunk, Document
from .parser import ParsedDocument

__all__ = ["Chunk", "Document", "ParsedDocument", "chunk_document", "split_oversized"]

# ATX headings. The same regex the parser uses, kept in sync.
_HEADING_RE: Final = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)

# Paragraph boundary: one or more blank lines.
_PARAGRAPH_RE: Final = re.compile(r"\n\s*\n+")


@dataclass(frozen=True, slots=True)
class _Section:
    """One heading section: its breadcrumb path and the body text under it."""

    heading_path: tuple[str, ...]
    section_name: str
    text: str


def _slice_sections(body: str) -> list[_Section]:
    """Slice the body into heading sections in document order.

    Returns an empty list if the body is empty or whitespace-only.
    """
    body = body.strip()
    if not body:
        return []

    # Find all headings with their character offsets.
    matches = list(_HEADING_RE.finditer(body))
    if not matches:
        # No headings at all — the whole body is one section with an
        # empty breadcrumb. Callers attach a parent title when needed.
        return [_Section(heading_path=(), section_name="", text=body)]

    sections: list[_Section] = []
    # Running heading stack: index i is the current heading at level i+1.
    stack: list[tuple[int, str]] = []

    for i, match in enumerate(matches):
        level = len(match.group(1))
        text = match.group(2).strip()
        # Promote/demote the stack to match this heading's level.
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, text))
        # Body runs from end-of-heading-line to start-of-next-heading (or EOF).
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        section_text = body[start:end].strip()
        sections.append(
            _Section(
                heading_path=tuple(name for _, name in stack),
                section_name=text,
                text=section_text,
            )
        )

    # If there is body before the first heading, attach it as a preamble
    # whose heading_path is empty and section_name is "".
    preamble = body[: matches[0].start()].strip()
    if preamble:
        sections.insert(0, _Section(heading_path=(), section_name="", text=preamble))

    return sections


def split_oversized(text: str, max_chars: int) -> list[str]:
    """Greedy paragraph-level split. Never cuts a paragraph in half.

    If the text fits, returns ``[text]``. Otherwise splits at the
    last paragraph boundary before ``max_chars`` and recurses on
    the remainder. If a single paragraph is itself over the limit
    (rare — markdown paragraphs that long usually indicate a code
    block or pasted log), it is kept as a single chunk so we never
    lose content.
    """
    if len(text) <= max_chars or max_chars <= 0:
        return [text] if text else []
    # Find paragraph boundaries: matches the blank line(s) between paragraphs.
    # We use the start of the separator, not the end, so the head
    # doesn't include the trailing "\n\n" (which would be a half-cut
    # boundary marker in the chunk's text).
    boundaries = [m.start() for m in _PARAGRAPH_RE.finditer(text)]
    if not boundaries:
        return [text]
    # Pick the largest boundary <= max_chars.
    cut = max((b for b in boundaries if b <= max_chars), default=-1)
    if cut <= 0:
        # No boundary before the limit: keep one giant chunk rather
        # than fragmenting a single paragraph.
        return [text]
    head = text[:cut].rstrip()
    tail = text[cut:].lstrip()
    return [head, *split_oversized(tail, max_chars)]


def _section_to_chunks(
    section: _Section,
    *,
    document_id: int,
    parent_title: str,
    start_position: int,
    max_chunk_chars: int,
) -> list[Chunk]:
    """Convert one section into one or more Chunks.

    The breadcrumb is the heading path exactly as the parser saw it:
    no prepending of the document title, no deduplication. The H1
    that matches the document title appears as the first element
    of the path naturally. The document title is metadata, not
    part of the structural tree.

    Preamble sections (body before the first heading) are filed
    under the document title as a single-element breadcrumb so they
    can still be found by retrieval.

    Oversized sections are split at paragraph boundaries; the
    split pieces share the same heading path and section_name but
    get distinct positions.
    """
    if not section.text:
        return []

    if section.heading_path:
        breadcrumb = section.heading_path
    elif parent_title:
        breadcrumb = (parent_title,)
    else:
        breadcrumb = ()

    pieces = split_oversized(section.text, max_chunk_chars)
    chunks: list[Chunk] = []
    for offset, piece in enumerate(pieces):
        chunks.append(
            Chunk(
                id=None,
                document_id=document_id,
                heading_path=breadcrumb,
                section_name=section.section_name,
                text=piece,
                position=start_position + offset,
            )
        )
    return chunks


def chunk_document(
    parsed: ParsedDocument,
    *,
    document_id: int = 0,
    max_chunk_chars: int = 2000,
) -> list[Chunk]:
    """Chunk a parsed document.

    Parameters
    ----------
    parsed:
        The output of :func:`llmwiki.parser.parse_markdown`. The
        ``body`` field must be populated (it is, by default).
    document_id:
        The row id of the parent document. ``0`` for unsaved
        documents (e.g. in unit tests).
    max_chunk_chars:
        Safety-net upper bound on chunk size in characters. The
        chunker first splits at heading boundaries (the primary
        rule); if any single section exceeds this size, it is
        further split at paragraph boundaries. Set to ``0`` to
        disable the safety net (not recommended).
    """
    parent_title = parsed.title or ""
    sections = _slice_sections(parsed.body)
    if not sections:
        return []
    chunks: list[Chunk] = []
    position = 0
    for section in sections:
        section_chunks = _section_to_chunks(
            section,
            document_id=document_id,
            parent_title=parent_title,
            start_position=position,
            max_chunk_chars=max_chunk_chars,
        )
        chunks.extend(section_chunks)
        position += len(section_chunks)
    return chunks


# Backwards-compatible signature: accept a Document and re-parse the
# body from disk. This is the slow path; callers should prefer
# chunk_document(parsed) when they already have a ParsedDocument.
def chunk_document_from_disk(
    doc: Document,
    *,
    path: str,
    max_chunk_chars: int = 2000,
) -> list[Chunk]:
    """Re-read the file at ``path`` and chunk it.

    Provided for callers that only have a :class:`Document` and
    don't want to plumb the body through. The indexer uses the
    fast path (parsed body) and doesn't need this.
    """
    # Local import to avoid a cycle (parser imports models).
    from .parser import parse_markdown

    parsed = parse_markdown(path)
    return chunk_document(
        parsed,
        document_id=doc.id if doc.id is not None else 0,
        max_chunk_chars=max_chunk_chars,
    )


__all__ = [
    "Chunk",
    "Document",
    "ParsedDocument",
    "chunk_document",
    "chunk_document_from_disk",
    "split_oversized",
]
