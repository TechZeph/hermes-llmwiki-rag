"""Versioned input recipes for chunking and dense embeddings."""

from __future__ import annotations

from collections.abc import Sequence

CHUNKER_RECIPE_VERSION = "chunker-v1-heading-char-2000"
DOCUMENT_EMBEDDING_RECIPE_VERSION = "document-v1-structural"
QUERY_EMBEDDING_RECIPE_VERSION = "query-v1-raw"
CORPUS_POLICY_VERSION = "corpus-v1-path-profiles"


def format_document_embedding_input(
    *,
    title: str,
    heading_path: Sequence[str],
    aliases: Sequence[str],
    tags: Sequence[str],
    body: str,
) -> str:
    """Return the v1 structural document input without authority metadata."""
    lines = [f"Title: {title}"]
    if heading_path:
        lines.append(f"Heading: {' > '.join(heading_path)}")
    if aliases:
        lines.append(f"Aliases: {', '.join(aliases)}")
    if tags:
        lines.append(f"Tags: {', '.join(tags)}")
    return "\n".join(lines) + "\n\n" + body


def format_query_embedding_input(query: str) -> str:
    """Return the v1 raw-query baseline for independent future comparison."""
    return query.strip()


def embedding_recipe_state(*, model_name: str, dimension: int) -> dict[str, str]:
    """Return the persisted compatibility state for the active dense projection."""
    return {
        "recipe.chunker": CHUNKER_RECIPE_VERSION,
        "recipe.document_embedding": DOCUMENT_EMBEDDING_RECIPE_VERSION,
        "recipe.query_embedding": QUERY_EMBEDDING_RECIPE_VERSION,
        "recipe.corpus_policy": CORPUS_POLICY_VERSION,
        "recipe.embedding_model": model_name,
        "recipe.embedding_dimension": str(dimension),
    }


__all__ = [
    "CHUNKER_RECIPE_VERSION",
    "CORPUS_POLICY_VERSION",
    "DOCUMENT_EMBEDDING_RECIPE_VERSION",
    "QUERY_EMBEDDING_RECIPE_VERSION",
    "embedding_recipe_state",
    "format_document_embedding_input",
    "format_query_embedding_input",
]
