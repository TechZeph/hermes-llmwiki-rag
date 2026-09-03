"""Integration test for Phase 3 end-to-end semantic search (golden question).

This test does NOT use the real FastEmbed model — it uses a deterministic
fake embedder that produces cosine-aligned vectors from keywords. That
gives us a precise assertion: a query whose query vector is identical to
a chunk's stored vector must surface that chunk as the top hit.

The shape of this test mirrors what the Phase 13 eval harness will do
against a real model: index a small synthetic vault, ask a question,
verify the right document surfaces.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from llmwiki import db as dbmod
from llmwiki.config import Settings
from llmwiki.embeddings import Embedder
from llmwiki.indexer import Indexer
from llmwiki.vector import SqliteVecStore

# The production schema hardcodes float[384]. The fake embedder below
# uses 384-dim vectors with one-hot-at-keyword-index semantics.
_PROD_DIM = 384


class KeywordEmbedder(Embedder):
    """A toy embedder that places each known keyword at a unique dim.

    A text containing keyword K[i] gets a 1.0 at index i (and 0.0
    elsewhere). The resulting vector is unit-length, so cosine
    distance is well-defined and a query for "apple" exactly
    matches the embedding of any text containing "apple".

    The dimension is fixed at 384 to match the production schema.
    """

    def __init__(self, keywords: Sequence[str], dim: int = _PROD_DIM) -> None:
        if len(keywords) > dim:
            raise ValueError(
                f"too many keywords ({len(keywords)}) for dim={dim}; "
                f"raise dim or use fewer keywords"
            )
        self._keywords = tuple(keywords)
        self._index = {kw: i for i, kw in enumerate(self._keywords)}
        self._dim = dim

    @property
    def model_name(self) -> str:
        return "keyword-embedder"

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            v = [0.0] * self._dim
            lower = t.lower()
            for kw, idx in self._index.items():
                if kw in lower:
                    v[idx] = 1.0
            out.append(v)
        return out


class RecordingEmbedder(KeywordEmbedder):
    """Keyword embedder that exposes the exact text passed to embedding."""

    def __init__(self, keywords: Sequence[str], dim: int = _PROD_DIM) -> None:
        super().__init__(keywords, dim)
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(tuple(texts))
        return super().embed(texts)


class FailingRecordingEmbedder(RecordingEmbedder):
    """Fails when a changed embedding input carries the test sentinel."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if any("FAIL_EMBED" in text for text in texts):
            raise RuntimeError("injected embedding failure")
        return super().embed(texts)


def test_indexer_applies_configured_batch_limit_to_new_document_chunks(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "sections.md").write_text(
        "# Sections\n\nalpha\n\n## Two\n\nbeta\n\n## Three\n\ngamma\n",
        encoding="utf-8",
    )
    embedder = RecordingEmbedder(["alpha", "beta", "gamma"])
    settings = Settings(
        vault_path=vault,
        db_path=tmp_path / "test.sqlite",
        embedding_batch_size=1,
    )

    stats = Indexer(settings, embedder=embedder).run()

    assert stats.embeddings_built >= 3
    assert embedder.calls and all(len(call) <= 1 for call in embedder.calls)


def _make_vault(root: Path) -> None:
    """Three markdown files about distinct topics."""
    (root / "apples.md").write_text(
        "# Apples\n\n"
        "Apples are a popular fruit. They grow on trees.\n\n"
        "## Varieties\n\n"
        "There are many varieties of apples including Granny Smith.\n"
    )
    (root / "oranges.md").write_text(
        "# Oranges\n\nOranges are citrus fruits. They are rich in vitamin C.\n"
    )
    (root / "cars.md").write_text(
        "# Cars\n\nCars are motor vehicles with four wheels and an engine.\n"
    )


def _path_for_chunk(conn, chunk_id: int) -> str:
    row = conn.execute(
        """
        SELECT d.path FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.id = ?
        """,
        (chunk_id,),
    ).fetchone()
    assert row is not None
    return str(row[0])


def test_semantic_search_finds_relevant_document(tmp_path: Path) -> None:
    """A query about apples should rank the apples document first.

    The toy embedder gives exact one-hot vectors, so a query vector
    for a keyword K is identical to any chunk vector that mentions
    K. The top hit must come from the matching document.
    """
    vault = tmp_path / "vault"
    vault.mkdir()
    _make_vault(vault)

    keywords = ["apple", "orange", "fruit", "car", "wheel", "engine"]
    db_path = tmp_path / "test.sqlite"
    settings = Settings(vault_path=vault, db_path=db_path)
    embedder = KeywordEmbedder(keywords=keywords)

    indexer = Indexer(settings, embedder=embedder)
    stats = indexer.run(mode="incremental")
    assert stats.documents_added == 3
    assert stats.chunks_added >= 3
    assert stats.errors == ()

    with dbmod.connect(db_path) as conn:
        dbmod.init_schema(conn)
        store = SqliteVecStore(conn)
        assert store.count() == stats.chunks_added

        for keyword, expected_path in (
            ("apple", "apples.md"),
            ("orange", "oranges.md"),
            ("car", "cars.md"),
        ):
            q = embedder.embed([f"a query about {keyword}"])[0]
            hits = store.search(q, top_k=1)
            assert hits, f"expected at least one hit for {keyword!r}"
            assert _path_for_chunk(conn, hits[0][0]) == expected_path


def test_search_returns_top_k_in_ascending_distance_order(tmp_path: Path) -> None:
    """Sanity: a small multi-chunk vault returns ranked results."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\n\napple apple apple\n")
    (vault / "b.md").write_text("# B\n\norange orange\n")
    keywords = ["apple", "orange"]
    embedder = KeywordEmbedder(keywords=keywords)
    db_path = tmp_path / "test.sqlite"
    settings = Settings(vault_path=vault, db_path=db_path)
    Indexer(settings, embedder=embedder).run(mode="incremental")

    with dbmod.connect(db_path) as conn:
        store = SqliteVecStore(conn)
        hits = store.search(embedder.embed(["apple"])[0], top_k=2)
        assert len(hits) == 2
        # Smaller distance (more similar) comes first.
        assert hits[0][1] <= hits[1][1]


def test_indexed_run_is_incremental(tmp_path: Path) -> None:
    """A second pass over an unchanged vault does no work."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\n\nhello world\n")
    keywords = ["hello", "world"]
    embedder = KeywordEmbedder(keywords=keywords)
    db_path = tmp_path / "test.sqlite"
    settings = Settings(vault_path=vault, db_path=db_path)

    s1 = Indexer(settings, embedder=embedder).run(mode="incremental")
    assert s1.documents_added == 1
    assert s1.embeddings_built >= 1

    s2 = Indexer(settings, embedder=embedder).run(mode="incremental")
    assert s2.documents_added == 0
    assert s2.documents_updated == 0
    assert s2.documents_skipped == 1
    assert s2.embeddings_built == 0
    assert s2.embeddings_rebuilt == 0


def test_changed_document_reuses_vectors_for_byte_identical_embedding_inputs(
    tmp_path: Path,
) -> None:
    """An edit re-embeds only the structurally changed chunk, not its siblings."""
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "note.md"
    note.write_text(
        "# Note\n\n"
        "## First\n\n"
        "apple unchanged\n\n"
        "## Second\n\n"
        "orange original\n\n"
        "## Third\n\n"
        "pear unchanged\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "test.sqlite"
    embedder = RecordingEmbedder(keywords=["apple", "orange", "pear"])
    settings = Settings(vault_path=vault, db_path=db_path)

    Indexer(settings, embedder=embedder).run()
    with dbmod.connect(db_path) as conn:
        before = dict(conn.execute("SELECT section_name, id FROM chunks").fetchall())

    embedder.calls.clear()
    note.write_text(
        "# Note\n\n"
        "## First\n\n"
        "apple unchanged\n\n"
        "## Second\n\n"
        "orange changed\n\n"
        "## Third\n\n"
        "pear unchanged\n",
        encoding="utf-8",
    )
    stats = Indexer(settings, embedder=embedder).run()

    assert stats.embeddings_built == 1
    assert embedder.calls == [("Title: Note\nHeading: Note > Second\n\norange changed",)]
    with dbmod.connect(db_path) as conn:
        after = dict(conn.execute("SELECT section_name, id FROM chunks").fetchall())
        assert after["First"] == before["First"]
        assert after["Third"] == before["Third"]
        store = SqliteVecStore(conn)
        assert store.count() == 3
        for query, expected_section in (
            ("apple", "First"),
            ("orange", "Second"),
            ("pear", "Third"),
        ):
            hit = store.search(embedder.embed([query])[0], top_k=1)
            section = conn.execute(
                "SELECT section_name FROM chunks WHERE id = ?", (hit[0][0],)
            ).fetchone()
            assert section == (expected_section,)


def test_metadata_change_reembeds_every_affected_structural_input(tmp_path: Path) -> None:
    """Changing aliases invalidates every chunk because aliases are embedded input."""
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "note.md"
    note.write_text(
        "---\n"
        "aliases: [Old note]\n"
        "---\n\n"
        "# Note\n\n"
        "## First\n\n"
        "apple body\n\n"
        "## Second\n\n"
        "orange body\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "test.sqlite"
    embedder = RecordingEmbedder(keywords=["apple", "orange"])
    settings = Settings(vault_path=vault, db_path=db_path)

    Indexer(settings, embedder=embedder).run()
    embedder.calls.clear()
    note.write_text(
        "---\n"
        "aliases: [Renamed note]\n"
        "---\n\n"
        "# Note\n\n"
        "## First\n\n"
        "apple body\n\n"
        "## Second\n\n"
        "orange body\n",
        encoding="utf-8",
    )

    stats = Indexer(settings, embedder=embedder).run()

    assert stats.embeddings_built == 2
    assert embedder.calls == [
        (
            "Title: Note\nHeading: Note > First\nAliases: Renamed note\n\napple body",
            "Title: Note\nHeading: Note > Second\nAliases: Renamed note\n\norange body",
        )
    ]


def test_inserted_preceding_chunk_keeps_later_vector_ids(tmp_path: Path) -> None:
    """Position shifts do not invalidate later byte-identical embedding inputs."""
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "note.md"
    note.write_text(
        "# Note\n\n## First\n\napple body\n\n## Second\n\norange body\n", encoding="utf-8"
    )
    db_path = tmp_path / "test.sqlite"
    embedder = RecordingEmbedder(keywords=["apple", "orange", "pear"])
    settings = Settings(vault_path=vault, db_path=db_path)

    Indexer(settings, embedder=embedder).run()
    with dbmod.connect(db_path) as conn:
        before = dict(conn.execute("SELECT section_name, id FROM chunks").fetchall())

    embedder.calls.clear()
    note.write_text(
        "# Note\n\n## Intro\n\npear body\n\n## First\n\napple body\n\n## Second\n\norange body\n",
        encoding="utf-8",
    )
    stats = Indexer(settings, embedder=embedder).run()

    assert stats.embeddings_built == 1
    with dbmod.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT section_name, id, position FROM chunks ORDER BY position"
        ).fetchall()
        assert [(row[0], row[2]) for row in rows] == [("Intro", 0), ("First", 1), ("Second", 2)]
        ids = {str(row[0]): int(row[1]) for row in rows}
        assert ids["First"] == before["First"]
        assert ids["Second"] == before["Second"]
        assert SqliteVecStore(conn).count() == 3


def test_partial_reuse_update_rolls_back_when_changed_chunk_embedding_fails(tmp_path: Path) -> None:
    """A failed changed chunk leaves retained siblings and all prior rows untouched."""
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "note.md"
    note.write_text(
        "# Note\n\n## Retained\n\napple unchanged\n\n## Changed\n\norange original\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "test.sqlite"
    embedder = FailingRecordingEmbedder(keywords=["apple", "orange"])
    settings = Settings(vault_path=vault, db_path=db_path)

    Indexer(settings, embedder=embedder).run()
    with dbmod.connect(db_path) as conn:
        before_document = conn.execute(
            "SELECT content_hash FROM documents WHERE path = 'note.md'"
        ).fetchone()
        before_chunks = conn.execute(
            "SELECT id, section_name, text FROM chunks ORDER BY id"
        ).fetchall()
        before_vectors = conn.execute(
            "SELECT chunk_id FROM chunk_embeddings ORDER BY chunk_id"
        ).fetchall()

    note.write_text(
        "# Note\n\n## Retained\n\napple unchanged\n\n## Changed\n\nFAIL_EMBED orange changed\n",
        encoding="utf-8",
    )
    stats = Indexer(settings, embedder=embedder).run()

    assert len(stats.errors) == 1
    with dbmod.connect(db_path) as conn:
        assert (
            conn.execute("SELECT content_hash FROM documents WHERE path = 'note.md'").fetchone()
            == before_document
        )
        assert (
            conn.execute("SELECT id, section_name, text FROM chunks ORDER BY id").fetchall()
            == before_chunks
        )
        assert (
            conn.execute("SELECT chunk_id FROM chunk_embeddings ORDER BY chunk_id").fetchall()
            == before_vectors
        )
        assert dbmod.inspect_integrity(db_path, vault_path=vault)["orphan_vectors"] == 0


def test_no_embed_update_removes_superseded_vectors(tmp_path: Path) -> None:
    """A metadata-only reindex must not leave vectors for replaced chunks behind."""
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "note.md"
    note.write_text("# Note\n\napple first version\n", encoding="utf-8")
    db_path = tmp_path / "test.sqlite"
    settings = Settings(vault_path=vault, db_path=db_path)
    embedder = KeywordEmbedder(keywords=["apple"])

    Indexer(settings, embedder=embedder).run()
    note.write_text("# Note\n\napple replacement version\n", encoding="utf-8")
    Indexer(settings).run()

    report = dbmod.inspect_integrity(db_path, vault_path=vault)
    assert report["orphan_vectors"] == 0


def test_indexer_embeds_structural_document_recipe(tmp_path: Path) -> None:
    """The document embedding input includes structural metadata, not body text alone."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text(
        """---
aliases: [RAG architecture]
tags: [rag, retrieval]
---

# Architecture

## Embedding recipe

apple body text
""",
        encoding="utf-8",
    )
    db_path = tmp_path / "test.sqlite"
    embedder = RecordingEmbedder(keywords=["apple"])

    Indexer(Settings(vault_path=vault, db_path=db_path), embedder=embedder).run()

    assert embedder.calls == [
        (
            "Title: Architecture\n"
            "Heading: Architecture > Embedding recipe\n"
            "Aliases: RAG architecture\n"
            "Tags: rag, retrieval\n"
            "\n"
            "apple body text",
        )
    ]
    with dbmod.connect(db_path) as conn:
        state = dict(
            conn.execute(
                "SELECT key, value FROM projection_meta WHERE key LIKE 'recipe.%' ORDER BY key"
            ).fetchall()
        )
    assert state == {
        "recipe.chunker": "chunker-v1-heading-char-2000",
        "recipe.corpus_policy": "corpus-v1-path-profiles",
        "recipe.document_embedding": "document-v1-structural",
        "recipe.embedding_dimension": "384",
        "recipe.embedding_model": "BAAI/bge-small-en-v1.5",
        "recipe.query_embedding": "query-v2-bge-instruction",
    }


def test_indexer_rejects_embedder_dimension_that_cannot_fit_vector_schema(tmp_path: Path) -> None:
    """A configured model must prove its actual output dimension before indexing."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("# Note\n\napple body\n", encoding="utf-8")

    with pytest.raises(
        ValueError, match="embedder dimension does not match configured vector schema"
    ):
        Indexer(
            Settings(vault_path=vault, db_path=tmp_path / "test.sqlite"),
            embedder=KeywordEmbedder(keywords=["apple"], dim=3),
        ).run()
