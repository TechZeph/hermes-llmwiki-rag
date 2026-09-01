"""Versioned input recipes for chunking and dense embeddings."""

from __future__ import annotations

from collections.abc import Sequence

CHUNKER_RECIPE_VERSION = "chunker-v1-heading-char-2000"
DOCUMENT_EMBEDDING_RECIPE_VERSION = "document-v1-structural"
QUERY_EMBEDDING_RECIPE_VERSION = "query-v2-bge-instruction"
CORPUS_POLICY_VERSION = "corpus-v1-path-profiles"

# Query-side recipes are evaluated independently of the document side; a
# query recipe change never requires re-embedding documents.
QUERY_RECIPES: dict[str, str] = {
    "query-v1-raw": "",
    # BGE v1.5 model card: short-query -> passage retrieval instruction.
    "query-v2-bge-instruction": "Represent this sentence for searching relevant passages: ",
}


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


def format_query_embedding_input(
    query: str, *, recipe: str = QUERY_EMBEDDING_RECIPE_VERSION
) -> str:
    """Return the query text for ``recipe`` (raw baseline or instruction-prefixed)."""
    if recipe not in QUERY_RECIPES:
        raise ValueError(f"unknown query recipe {recipe!r}; known: {sorted(QUERY_RECIPES)}")
    return QUERY_RECIPES[recipe] + query.strip()


def embedding_recipe_state(*, model_name: str, dimension: int) -> dict[str, str]:
    """Return the persisted compatibility state for the active dense projection.

    Only document-side keys participate in the re-embed decision; see
    :func:`compatibility_keys`.
    """
    return {
        "recipe.chunker": CHUNKER_RECIPE_VERSION,
        "recipe.document_embedding": DOCUMENT_EMBEDDING_RECIPE_VERSION,
        "recipe.query_embedding": QUERY_EMBEDDING_RECIPE_VERSION,
        "recipe.corpus_policy": CORPUS_POLICY_VERSION,
        "recipe.embedding_model": model_name,
        "recipe.embedding_dimension": str(dimension),
    }


def compatibility_state(state: dict[str, str]) -> dict[str, str]:
    """Subset of recipe state whose change invalidates stored document vectors."""
    keys = (
        "recipe.chunker",
        "recipe.document_embedding",
        "recipe.embedding_model",
        "recipe.embedding_dimension",
    )
    return {k: state[k] for k in keys if k in state}


__all__ = [
    "CHUNKER_RECIPE_VERSION",
    "CORPUS_POLICY_VERSION",
    "DOCUMENT_EMBEDDING_RECIPE_VERSION",
    "QUERY_EMBEDDING_RECIPE_VERSION",
    "QUERY_RECIPES",
    "compatibility_state",
    "embedding_recipe_state",
    "format_document_embedding_input",
    "format_query_embedding_input",
]
