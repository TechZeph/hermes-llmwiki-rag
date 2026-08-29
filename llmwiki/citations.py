"""Phase 7 placeholders for citations and LLM-ready context blocks."""

from __future__ import annotations

from .models import Chunk, Document


def build_citation(doc: Document, chunk: Chunk) -> str:
    """Placeholder. Returns a one-line human-readable citation."""
    heading = " > ".join(chunk.heading_path) if chunk.heading_path else doc.title
    return f"{doc.path} :: {heading}"


def build_context(*, query: str, results: list[tuple[Document, Chunk]]) -> str:
    """Placeholder. Joins citations with the chunk text."""
    parts: list[str] = [f"# Retrieved context for: {query}\n"]
    for doc, chunk in results:
        parts.append(f"## {build_citation(doc, chunk)}")
        parts.append(chunk.text)
        parts.append("")
    return "\n".join(parts)


__all__ = ["build_citation", "build_context"]
