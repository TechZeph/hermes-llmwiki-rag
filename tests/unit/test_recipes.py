"""Tests for versioned chunk and embedding input recipes."""

from __future__ import annotations

import pytest

from llmwiki.embeddings import model_provenance
from llmwiki.recipes import (
    CHUNKER_RECIPE_VERSION,
    DOCUMENT_EMBEDDING_RECIPE_VERSION,
    QUERY_EMBEDDING_RECIPE_VERSION,
    embedding_recipe_state,
    format_document_embedding_input,
    format_query_embedding_input,
)


def test_document_recipe_preserves_structural_context_before_chunk_body() -> None:
    """Document vectors include the title, breadcrumb, aliases, and tags."""
    assert DOCUMENT_EMBEDDING_RECIPE_VERSION == "document-v1-structural"
    assert format_document_embedding_input(
        title="RAG Architecture",
        heading_path=("RAG Architecture", "Chunking and embeddings"),
        aliases=("Retrieval architecture",),
        tags=("rag", "retrieval"),
        body="Chunk bodies remain the canonical source text.",
    ) == (
        "Title: RAG Architecture\n"
        "Heading: RAG Architecture > Chunking and embeddings\n"
        "Aliases: Retrieval architecture\n"
        "Tags: rag, retrieval\n"
        "\n"
        "Chunk bodies remain the canonical source text."
    )


def test_document_recipe_omits_empty_optional_metadata() -> None:
    """Sparse notes do not gain empty labels that add embedding noise."""
    assert (
        format_document_embedding_input(
            title="Untitled",
            heading_path=(),
            aliases=(),
            tags=(),
            body="body",
        )
        == "Title: Untitled\n\nbody"
    )


def test_query_recipe_is_independently_versioned() -> None:
    """Query recipes are selectable; v1 stays a measurable raw baseline, v2 is the default."""
    assert CHUNKER_RECIPE_VERSION == "chunker-v1-heading-char-2000"
    assert QUERY_EMBEDDING_RECIPE_VERSION == "query-v2-bge-instruction"
    assert (
        format_query_embedding_input("  What is the current RAG state?  ", recipe="query-v1-raw")
        == "What is the current RAG state?"
    )
    assert format_query_embedding_input("  q ").startswith(
        "Represent this sentence for searching relevant passages: q"
    )
    with pytest.raises(ValueError):
        format_query_embedding_input("q", recipe="query-v9")


def test_embedding_recipe_state_records_versions_model_and_actual_dimension() -> None:
    """The projection state contains every compatibility component used by dense search."""
    assert embedding_recipe_state(model_name="BAAI/bge-small-en-v1.5", dimension=384) == {
        "recipe.chunker": "chunker-v1-heading-char-2000",
        "recipe.document_embedding": "document-v1-structural",
        "recipe.query_embedding": "query-v2-bge-instruction",
        "recipe.corpus_policy": "corpus-v1-path-profiles",
        "recipe.embedding_model": "BAAI/bge-small-en-v1.5",
        "recipe.embedding_dimension": "384",
    }


def test_model_provenance_records_the_fastembed_package_and_artifact_source() -> None:
    """A projection records enough local-model provenance to be reproduced offline."""
    provenance = model_provenance("BAAI/bge-small-en-v1.5")

    assert provenance["embedding.backend"] == "fastembed"
    assert provenance["embedding.backend_version"]
    assert provenance["embedding.artifact_source"] == "qdrant/bge-small-en-v1.5-onnx-q"
